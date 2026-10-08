#!/usr/bin/env python3

"""
Prerequisites:
  1. An app at https://developer.spotify.com/dashboard
  2. In the app settings add as Redirect URI EXACTLY:
         http://127.0.0.1:8888/callback
     (if you change doors with --port, also change the URI)
  3. Since 2026 apps in Development Mode require that the owner
     have Spotify Premium and that your user is on the allowlist
     (User Management in the dashboard, max 5 users).

Usage:
  python rainbow_playlist.py <playlist url|uri|id> --client-id XXX --client-secret YYY
  (or environment variables SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET)

Options:
  --copy creates a NEW sorted playlist instead of reordering the original
  --dry-run calculates and shows order without writing anything
  --grays-first puts the grey/white/black covers at the beginning instead of the bottom
  --port N local door for OAuth callback (default 8888)
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import io
import json
import os
import re
import secrets
import sys
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import numpy as np
import requests
from PIL import Image

API = "https://api.spotify.com/v1"
AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SCOPE = ("playlist-read-private playlist-read-collaborative "
         "playlist-modify-public playlist-modify-private")
TOKEN_CACHE = os.path.join(os.path.expanduser("~"), ".spotify_rainbow_token.json")

PAGE_SIZE = 50          # maximum for reading playlist items
WRITE_CHUNKS = (100, 50, 10)   # write chunk sizes (fallback if Spotify rejects)
IMG_WORKERS = 32        # cover downloads in parallel
API_WORKERS = 8         # playlist pages read in parallel
HUE_SHIFT = 15          # moves "pinkish" reds (345-360 degrees) to the beginning with the other reds
CHROMA_MIN_FRACTION = 0.12   # below this fraction of colored pixels the cover is "gray"


# --------------------------------------------------------------------------
# Authentication (Authorization Code flow con client secret)
# --------------------------------------------------------------------------
class CallbackHandler(BaseHTTPRequestHandler):
    params: dict = {}

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return
        CallbackHandler.params = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write("<h2>Authorization completed, you can close this tab.</h2>".encode())

    def log_message(self, *args):
        pass


def token_request(client_id: str, client_secret: str, **data) -> dict:
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    r = requests.post(TOKEN_URL, data=data, headers={"Authorization": f"Basic {basic}"}, timeout=30)
    if not r.ok:
        raise RuntimeError(f"Token endpoint -> {r.status_code}: {r.text[:300]}")
    return r.json()


def save_cache(client_id: str, refresh_token: str) -> None:
    try:
        with open(TOKEN_CACHE, "w", encoding="utf-8") as f:
            json.dump({"client_id": client_id, "scope": SCOPE, "refresh_token": refresh_token}, f)
        os.chmod(TOKEN_CACHE, 0o600)
    except OSError:
        pass


def get_access_token(client_id: str, client_secret: str, port: int) -> str:
    # 1) try the saved refresh token (no repeated login)
    try:
        with open(TOKEN_CACHE, encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("client_id") == client_id and cache.get("scope") == SCOPE and cache.get("refresh_token"):
            tok = token_request(client_id, client_secret,
                                grant_type="refresh_token", refresh_token=cache["refresh_token"])
            save_cache(client_id, tok.get("refresh_token", cache["refresh_token"]))
            return tok["access_token"]
    except (OSError, ValueError, RuntimeError):
        pass

    # 2) interactive login in the browser
    redirect = f"http://127.0.0.1:{port}/callback"
    state = secrets.token_urlsafe(16)
    url = AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": client_id, "response_type": "code", "redirect_uri": redirect,
        "scope": SCOPE, "state": state,
    })
    CallbackHandler.params = {}
    server = HTTPServer(("127.0.0.1", port), CallbackHandler)
    print("Opening the browser for authorization... (if it does not open, copy this link):")
    print(url)
    webbrowser.open(url)
    while "code" not in CallbackHandler.params and "error" not in CallbackHandler.params:
        server.handle_request()
    server.server_close()

    p = CallbackHandler.params
    if "error" in p:
        raise RuntimeError(f"Authorization denied: {p['error']}")
    if p.get("state") != state:
        raise RuntimeError("OAuth state mismatch, please try again.")
    tok = token_request(client_id, client_secret,
                        grant_type="authorization_code", code=p["code"], redirect_uri=redirect)
    save_cache(client_id, tok["refresh_token"])
    return tok["access_token"]


# --------------------------------------------------------------------------
# Minimal API client with retry/rate-limit
# --------------------------------------------------------------------------
class Spotify:
    def __init__(self, token: str):
        self.s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
        self.s.mount("https://", adapter)
        self.s.headers["Authorization"] = f"Bearer {token}"

    def call(self, method: str, path: str, **kw) -> dict:
        url = API + path
        for attempt in range(6):
            r = self.s.request(method, url, timeout=30, **kw)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", "2")) + 1)
                continue
            if r.status_code >= 500:
                time.sleep(1 + attempt)
                continue
            if not r.ok:
                raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
            return r.json() if r.content else {}
        raise RuntimeError(f"{method} {path}: troppi tentativi falliti")


# --------------------------------------------------------------------------
# Playlist reading
# --------------------------------------------------------------------------
@dataclass
class Track:
    idx: int
    uri: str
    name: str
    artists: str
    album_id: str
    cover: Optional[str]
    local: bool
    key: tuple = field(default_factory=tuple)


def parse_playlist_id(value: str) -> str:
    m = re.search(r"playlist[/:]([A-Za-z0-9]+)", value)
    return m.group(1) if m else value.strip()


def fetch_entries(api: Spotify, pid: str) -> list:
    path = f"/playlists/{pid}/items"
    base = {"limit": PAGE_SIZE, "additional_types": "track"}
    first = api.call("GET", path, params={**base, "offset": 0})
    total = first.get("total", 0)
    pages = {0: first.get("items", [])}

    def get(offset: int):
        return offset, api.call("GET", path, params={**base, "offset": offset}).get("items", [])

    with cf.ThreadPoolExecutor(API_WORKERS) as ex:
        for offset, items in ex.map(get, range(PAGE_SIZE, total, PAGE_SIZE)):
            pages[offset] = items
    return [e for off in sorted(pages) for e in pages[off]]


def parse_tracks(entries: list) -> list[Track]:
    tracks = []
    for i, e in enumerate(entries):
        t = e.get("item") or e.get("track")      # "item" dalla revisione di febbraio 2026
        if not t or not t.get("uri"):
            continue
        uri = t["uri"]
        album = t.get("album") or {}
        imgs = album.get("images") or t.get("images") or []
        cover = min(imgs, key=lambda im: im.get("width") or 10 ** 6)["url"] if imgs else None
        tracks.append(Track(
            idx=i, uri=uri, name=t.get("name", "?"),
            artists=", ".join(a.get("name", "") for a in t.get("artists", [])),
            album_id=album.get("id") or album.get("uri") or "",
            cover=cover,
            local=bool(t.get("is_local")) or uri.startswith("spotify:local:"),
        ))
    return tracks


# --------------------------------------------------------------------------
# Color analysis
# --------------------------------------------------------------------------
def analyze(img_bytes: bytes):
    """Returns ('c', hue_degrees, brightness) for colored covers, ('g', 0, brightness) for gray covers."""
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((32, 32))
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32).reshape(-1, 3)
    h = hsv[:, 0] * (360.0 / 255.0)
    s = hsv[:, 1] / 255.0
    v = hsv[:, 2] / 255.0

    mask = (s > 0.25) & (v > 0.20)
    if mask.mean() < CHROMA_MIN_FRACTION:
        return ("g", 0.0, float(v.mean()))

    hm, w = h[mask], (s * v)[mask]          # vivid pixels have more weight
    hist = np.bincount((hm // 10).astype(int) % 36, weights=w, minlength=36)
    smooth = hist + np.roll(hist, 1) + np.roll(hist, -1)
    peak = int(np.argmax(smooth)) * 10 + 5   # dominant band (degrees)

    near = np.abs((hm - peak + 180) % 360 - 180) <= 25
    rad = np.radians(hm[near])
    hue = float(np.degrees(np.arctan2((np.sin(rad) * w[near]).sum(),
                                      (np.cos(rad) * w[near]).sum())) % 360)
    return ("c", hue, float(v.mean()))


def load_color(sess: requests.Session, url: str):
    for attempt in range(3):
        try:
            r = sess.get(url, timeout=20)
            r.raise_for_status()
            return analyze(r.content)
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None


def compute_colors(tracks: list[Track]) -> dict:
    urls = sorted({t.cover for t in tracks if t.cover})      # one request per cover
    sess = requests.Session()
    sess.mount("https://", requests.adapters.HTTPAdapter(pool_connections=IMG_WORKERS, pool_maxsize=IMG_WORKERS))
    colors = {}
    with cf.ThreadPoolExecutor(IMG_WORKERS) as ex:
        for n, (url, col) in enumerate(zip(urls, ex.map(lambda u: load_color(sess, u), urls)), 1):
            colors[url] = col
            if n % 100 == 0 or n == len(urls):
                print(f"\r  covers analyzed: {n}/{len(urls)}", end="", flush=True)
    print()
    return colors


def rainbow_sort(tracks: list[Track], colors: dict, grays_first: bool) -> list[Track]:
    g_color, g_gray = (1, 0) if grays_first else (0, 1)
    for t in tracks:
        c = colors.get(t.cover) if t.cover else None
        if c is None:
            t.key = (2, 0.0, t.album_id, t.idx)                       # without cover: at the bottom
        elif c[0] == "c":
            t.key = (g_color, (c[1] + HUE_SHIFT) % 360, t.album_id, t.idx)
        else:
            lum = c[2] if grays_first else -c[2]                        # gray: light->dark (or vice versa)
            t.key = (g_gray, lum, t.album_id, t.idx)
    return sorted(tracks, key=lambda t: t.key)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def write_order(api: Spotify, pid: str, uris: list[str], replace: bool) -> None:
    """replace=True: replaces the content (PUT on the first chunk); False: appends."""
    last_err = None
    for size in WRITE_CHUNKS:
        try:
            chunks = [uris[i:i + size] for i in range(0, len(uris), size)] or [[]]
            for n, chunk in enumerate(chunks):
                method = "PUT" if (replace and n == 0) else "POST"
                api.call(method, f"/playlists/{pid}/items", json={"uris": chunk})
                print(f"\r  written {min((n + 1) * size, len(uris))}/{len(uris)}", end="", flush=True)
            print()
            return
        except RuntimeError as e:
            last_err = e
            if " 400" not in str(e):
                raise
            print(f"\n  chunks of {size} rejected, retrying with smaller chunks...")
    raise last_err  # type: ignore[misc]


def main() -> int:
    ap = argparse.ArgumentParser(description="Sort a Spotify playlist in rainbow order.")
    ap.add_argument("playlist", help="playlist URL, URI, or ID")
    ap.add_argument("--client-id", default=os.environ.get("SPOTIFY_CLIENT_ID"))
    ap.add_argument("--client-secret", default=os.environ.get("SPOTIFY_CLIENT_SECRET"))
    ap.add_argument("--copy", action="store_true", help="create a new playlist instead of modifying the original")
    ap.add_argument("--dry-run", action="store_true", help="do not write anything")
    ap.add_argument("--grays-first", action="store_true", help="gray/white/black at the beginning")
    ap.add_argument("--port", type=int, default=8888)
    a = ap.parse_args()

    if not a.client_id or not a.client_secret:
        ap.error("--client-id and --client-secret are required (or the SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET environment variables)")

    t0 = time.time()
    pid = parse_playlist_id(a.playlist)
    api = Spotify(get_access_token(a.client_id, a.client_secret, a.port))

    print("Reading the playlist...")
    tracks = parse_tracks(fetch_entries(api, pid))
    locals_ = [t for t in tracks if t.local]
    playable = [t for t in tracks if not t.local]
    print(f"  {len(tracks)} tracks ({len(locals_)} local files)")
    if not playable:
        print("No tracks to sort.")
        return 1

    print("Analyzing the covers...")
    colors = compute_colors(playable)
    ordered = rainbow_sort(playable, colors, a.grays_first)
    uris = [t.uri for t in ordered]

    print("\nOrder preview:")
    for t in (ordered[:5] + [None] + ordered[-5:]) if len(ordered) > 12 else ordered:
        print("   ..." if t is None else f"   {t.artists} - {t.name}")

    if a.dry_run:
        print(f"\nDry-run completed in {time.time() - t0:.1f}s, nothing was modified.")
        return 0

    if a.copy:
        name = api.call("GET", f"/playlists/{pid}", params={"fields": "name"}).get("name", "Playlist")
        new = api.call("POST", "/me/playlists",
                       json={"name": f"{name} (rainbow)", "public": False,
                             "description": "Sorted by cover color"})
        print(f"\nCreated new playlist: {new.get('external_urls', {}).get('spotify', new['id'])}")
        write_order(api, new["id"], uris, replace=False)
    else:
        if locals_:
            print("\nLa playlist contiene local files, che l'API non puo' riscrivere: "
                  "rerun with --copy to avoid losing them (they will still be excluded from the copy).")
            return 1
        backup = f"backup_{pid}.json"
        with open(backup, "w", encoding="utf-8") as f:
            json.dump([t.uri for t in sorted(playable, key=lambda t: t.idx)], f)
        print(f"\nBackup of the original order: {backup}")
        write_order(api, pid, uris, replace=True)

    print(f"\nDone in {time.time() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
