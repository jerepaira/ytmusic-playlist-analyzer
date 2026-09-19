"""Local web UI for the Jev playlist meter and scanner.

    uv run --no-project uvicorn webui:app --host 127.0.0.1 --port 8020

Then open http://127.0.0.1:8020/
"""

from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from jev_local import JevError, JevLocal
from rate import (
    BACKENDS,
    add_profile,
    combined_profile,
    get_active,
    get_threshold,
    list_profiles,
    remove_profile,
    resolve,
    resolve_profile,
    score,
)

app = FastAPI(title="jev playlist meter")
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
CONCURRENCY = int(os.environ.get("JEV_CONCURRENCY", "3"))
UI = (Path(__file__).with_name("ui.html")).read_text()


def playlist_id_from(value: str) -> str:
    value = value.strip()
    if value.startswith("http"):
        vid = (parse_qs(urlparse(value).query).get("list") or [None])[0]
        if not vid:
            raise HTTPException(status_code=400, detail="no 'list=' parameter in URL")
        return vid
    return value


def fetch_tracks(playlist_id: str, max_tracks: int) -> tuple[str, list[dict]]:
    from ytmusicapi import YTMusic

    yt = YTMusic()
    limit = None if max_tracks <= 0 else max_tracks
    try:
        data = yt.get_playlist(playlist_id, limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"cannot read playlist: {exc}") from exc
    tracks: list[dict] = []
    seen: set[str] = set()
    for t in data.get("tracks", []):
        title = t.get("title")
        if not title or title in seen:
            continue
        seen.add(title)
        tracks.append({
            "title": title,
            "artists": [a["name"] for a in t.get("artists", [])],
            "album": (t.get("album") or {}).get("name") if isinstance(t.get("album"), dict) else t.get("album"),
            "year": t.get("year"),
            "videoId": t.get("videoId"),
        })
        if limit and len(tracks) >= limit:
            break
    return data.get("title", "playlist"), tracks


class AddProfileBody(BaseModel):
    playlist_id: str


class RateBody(BaseModel):
    query: str
    raw: bool = False
    backend: str = "lmstudio"
    model: str | None = None
    profile: str | None = None


class ScanBody(BaseModel):
    playlist: str
    max_tracks: int = 50
    backend: str = "lmstudio"
    model: str | None = None
    profile: str | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UI


@app.get("/api/profiles")
def api_profiles() -> dict:
    profiles = list_profiles()
    out = {"profiles": profiles, "active": get_active(), "combined": None}
    if profiles:
        c = combined_profile()
        out["combined"] = {"title": c["title"], "count": c["count"], "top_artists": c["top_artists"]}
    return out


@app.post("/api/profiles")
def api_add_profile(body: AddProfileBody) -> dict:
    pid = playlist_id_from(body.playlist_id)
    profile = add_profile(pid)
    return {"playlist_id": pid, "title": profile["title"], "count": profile["count"]}


@app.delete("/api/profiles/{pid}")
def api_remove_profile(pid: str) -> dict:
    remove_profile(pid)
    return {"removed": pid}


@app.post("/api/rate")
def api_rate(body: RateBody) -> dict:
    try:
        profile = resolve_profile(body.profile)
    except SystemExit as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    base_url, model = BACKENDS.get(body.backend, BACKENDS["lmstudio"])
    jev = JevLocal(base_url=base_url, model=body.model or model)
    track = {"title": body.query} if body.raw else resolve(body.query)
    try:
        result = score(jev, profile, track, include_genre=True)
        result["threshold"] = get_threshold(jev, profile, resolved=not body.raw)
        return result
    except JevError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _run_scan(job_id: str, playlist_id: str, max_tracks: int, base_url: str, model: str,
              selector: str | None) -> None:
    job = JOBS[job_id]
    try:
        title, tracks = fetch_tracks(playlist_id, max_tracks)
        profile = resolve_profile(selector)
        job["playlist_title"] = title
        job["profile_title"] = profile["title"]
        job["total"] = len(tracks)
        jev = JevLocal(base_url=base_url, model=model)
        job["threshold"] = get_threshold(jev, profile, resolved=True)

        def one(track: dict) -> dict:
            try:
                return score(jev, profile, track, include_genre=True)
            except Exception as exc:  # noqa: BLE001
                return {**track, "error": str(exc)[:200], "match": None}

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {pool.submit(one, t): t for t in tracks}
            for fut in as_completed(futures):
                with JOBS_LOCK:
                    job["results"].append(fut.result())
                    job["done"] += 1
        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(exc)


@app.post("/api/scan")
def api_scan(body: ScanBody) -> dict:
    try:
        resolve_profile(body.profile)
    except SystemExit as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pid = playlist_id_from(body.playlist)
    base_url, model = BACKENDS.get(body.backend, BACKENDS["lmstudio"])
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "done": 0, "total": 0, "results": [],
                    "playlist_title": None, "profile_title": None, "threshold": None, "error": None}
    threading.Thread(target=_run_scan,
                     args=(job_id, pid, body.max_tracks, base_url, body.model or model, body.profile),
                     daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/scan/{job_id}")
def api_scan_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job")
    with JOBS_LOCK:
        results = sorted([r for r in job["results"] if r.get("match") is not None],
                         key=lambda r: r["match"], reverse=True)
        done = job["done"]
    return {
        "status": job["status"],
        "done": done,
        "total": job["total"],
        "playlist_title": job["playlist_title"],
        "profile_title": job["profile_title"],
        "threshold": job["threshold"],
        "error": job["error"],
        "results": results,
    }
