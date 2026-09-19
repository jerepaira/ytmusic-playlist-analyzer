"""Classify candidate tracks against a YouTube Music playlist using the local Jev layer.

Suggests tracks; never modifies the playlist unless --apply is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jev_local import JevLocal

JEV = JevLocal(
    base_url="http://127.0.0.1:11434/v1",
    model="qwen2.5-coder:7b",
)

BACKENDS = {
    "ollama": ("http://127.0.0.1:11434/v1", "qwen2.5-coder:7b"),
    "lmstudio": ("http://127.0.0.1:1234/v1", "qwen2.5-coder-14b-instruct"),
    "atomic": ("http://127.0.0.1:1337/v1", "qwen2.5-coder-7b"),
}


def track_state(track: dict) -> str:
    artists = ", ".join(track.get("artists") or []) or track.get("artist")
    fields = [
        ("Title", track.get("title")),
        ("Artist(s)", artists),
        ("Album", track.get("album")),
        ("Year", track.get("year")),
        ("Duration", track.get("duration")),
        ("Plays", track.get("plays")),
    ]
    return "\n".join(f"{k}: {v}" for k, v in fields if v)


def playlist_profile(tracks: list[dict]) -> str:
    lines = []
    for t in tracks[:25]:
        artists = ", ".join(t.get("artists") or []) or t.get("artist", "?")
        lines.append(f"- {t.get('title', '?')} — {artists} ({t.get('album', '?')})")
    return "\n".join(lines)


def classify(playlist: list[dict], candidate: dict, min_fit: float) -> dict | None:
    state = (
        "PLAYLIST (existing tracks):\n"
        f"{playlist_profile(playlist)}\n\n"
        "CANDIDATE TRACK:\n"
        f"{track_state(candidate)}"
    )
    answers = JEV.decide(state, {
        "genre_fit": {
            "type": "choice",
            "instructions": "Which genre best matches the playlist's genre?",
            "criteria": {
                "same": "same genre / subgenre as the playlist",
                "related": "related genre, plausible fit",
                "different": "different genre",
            },
        },
        "fit": {
            "type": "score",
            "instructions": "How well does the candidate fit this playlist's style and mood?",
            "criteria": ["poor fit", "weak fit", "decent fit", "strong fit"],
        },
        "add": {
            "type": "noul",
            "instructions": "Should this track be added to the playlist?",
        },
    })["answers"]
    fit_norm = answers["fit"]["score"] / 3.0
    add_p = answers["add"]["noul"]
    if fit_norm < min_fit:
        return None
    return {
        "title": candidate.get("title"),
        "artists": candidate.get("artists"),
        "fit": answers["fit"]["score"],
        "fit_norm": round(fit_norm, 4),
        "add_probability": add_p,
        "genre_fit": answers["genre_fit"]["choice"],
        "genre_probabilities": answers["genre_fit"]["probabilities"],
    }


def load_demo() -> tuple[list[dict], list[dict]]:
    playlist = [
        {"title": "Bohemian Rhapsody", "artists": ["Queen"], "album": "A Night at the Opera", "year": "1975"},
        {"title": "Stairway to Heaven", "artists": ["Led Zeppelin"], "album": "Led Zeppelin IV", "year": "1971"},
        {"title": "Hotel California", "artists": ["Eagles"], "album": "Hotel California", "year": "1976"},
    ]
    candidates = [
        {"title": "Back In Black", "artists": ["AC/DC"], "album": "Back in Black", "year": "1980"},
        {"title": "Shape of You", "artists": ["Ed Sheeran"], "album": "Divide", "year": "2017"},
        {"title": "Wish You Were Here", "artists": ["Pink Floyd"], "album": "Wish You Were Here", "year": "1975"},
        {"title": "bad guy", "artists": ["Billie Eilish"], "album": "When We All Fall Asleep", "year": "2019"},
    ]
    return playlist, candidates


def from_ytmusic(playlist_id: str, seeds: int = 8, max_candidates: int = 40) -> tuple[list[dict], list[dict]]:
    from ytmusicapi import YTMusic

    yt = YTMusic()
    data = yt.get_playlist(playlist_id, limit=None)
    playlist = [
        {
            "title": t.get("title"),
            "artists": [a["name"] for a in t.get("artists", [])],
            "album": (t.get("album") or {}).get("name") if isinstance(t.get("album"), dict) else t.get("album"),
            "year": t.get("year"),
        }
        for t in data.get("tracks", [])
        if t.get("title")
    ]
    seen = {t["title"] for t in playlist}
    candidates: dict[str, dict] = {}
    for t in data.get("tracks", [])[:seeds]:
        if not t.get("videoId"):
            continue
        try:
            radio = yt.get_watch_playlist(t["videoId"], limit=15)
        except Exception:  # noqa: BLE001
            continue
        for r in radio.get("tracks", []):
            title = r.get("title")
            if not title or title in seen or title in candidates:
                continue
            candidates[title] = {
                "title": title,
                "artists": [a["name"] for a in r.get("artists", [])],
                "album": (r.get("album") or {}).get("name") if isinstance(r.get("album"), dict) else r.get("album"),
                "year": r.get("year"),
                "duration": r.get("duration"),
                "plays": r.get("views"),
                "videoId": r.get("videoId"),
            }
            if len(candidates) >= max_candidates:
                return playlist, list(candidates.values())
    return playlist, list(candidates.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--playlist-id", help="YouTube Music playlist id (list=... param)")
    ap.add_argument("--seeds", type=int, default=8, help="how many playlist tracks seed the candidate radio")
    ap.add_argument("--max-candidates", type=int, default=40, help="cap on generated candidates")
    ap.add_argument("--demo", action="store_true", help="run with built-in offline sample")
    ap.add_argument("--min-fit", type=float, default=0.5, help="minimum normalized fit score (0-1) to suggest")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="ollama", help="preset: ollama | lmstudio | atomic")
    ap.add_argument("--base-url", help="override OpenAI-compatible base URL")
    ap.add_argument("--model", help="override model name")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    base_url, model = BACKENDS[args.backend]
    JEV.base_url = args.base_url or base_url
    JEV.model = args.model or model
    print(f"backend: {JEV.base_url} | model: {JEV.model}", file=sys.stderr)
    if args.demo:
        playlist, candidates = load_demo()
    elif args.playlist_id:
        playlist, candidates = from_ytmusic(args.playlist_id, args.seeds, args.max_candidates)
    else:
        ap.error("pass --demo or --playlist-id")
        return 2

    print(f"playlist: {len(playlist)} tracks | candidates: {len(candidates)}", file=sys.stderr)
    suggestions = []
    for c in candidates:
        try:
            result = classify(playlist, c, args.min_fit)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {c.get('title')}: {exc}", file=sys.stderr)
            continue
        if result:
            suggestions.append(result)
            print(f"  + {result['title']} — {', '.join(result['artists'] or [])} "
                  f"(fit={result['fit_norm']:.2f}, P(add)={result['add_probability']:.2f})", file=sys.stderr)

    suggestions.sort(key=lambda s: s["fit_norm"], reverse=True)
    if args.json:
        print(json.dumps(suggestions, indent=2, ensure_ascii=False))
    else:
        print(f"\n{len(suggestions)} suggestions:")
        for s in suggestions:
            print(f"  fit={s['fit_norm']:.2f}  P(add)={s['add_probability']:.2f}  "
                  f"{s['title']} — {', '.join(s['artists'] or [])}  [{s['genre_fit']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
