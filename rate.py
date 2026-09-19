"""Rate any track you are looking at against your playlist, instantly.

Builds and caches a compact profile of the playlist once, then scores any
track you throw at it with the local Jev layer.

    uv run --no-project python rate.py "Guy J - Metal Dreams"
    uv run --no-project python rate.py "https://music.youtube.com/watch?v=XXXX"
    uv run --no-project python rate.py "Tame Impala - Let It Happen" --raw
    uv run --no-project python rate.py --playlist-id PL... --refresh   # rebuild profile

Output is a 0-100 match score from the calibrated fit distribution, plus the
raw P(add) and the genre call, so you can decide whether to spend a listen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from jev_local import JevLocal

PROFILES_FILE = Path(__file__).with_name(".profiles.json")
CALIB_FILE = Path(__file__).with_name(".calibration.json")
SEPS = (" – ", " — ", " - ", " | ", " @ ")
NEG_ANCHORS = [
    "Ed Sheeran - Shape of You",
    "Metallica - Enter Sandman",
    "Bad Bunny - Titi Me Pregunto",
    "Taylor Swift - Anti-Hero",
    "AC/DC - Thunderstruck",
]

BACKENDS = {
    "ollama": ("http://127.0.0.1:11434/v1", "qwen2.5-coder:7b"),
    "lmstudio": ("http://127.0.0.1:1234/v1", "qwen2.5-coder-14b-instruct"),
    "atomic": ("http://127.0.0.1:1337/v1", "qwen2.5-coder-7b"),
}


def extract_artist(title: str) -> str | None:
    for sep in SEPS:
        if sep in title:
            part = title.split(sep)[0].strip()
            if 2 <= len(part) <= 40 and any(c.isalpha() for c in part):
                return part
    return None


def build_profile(playlist_id: str) -> dict:
    from ytmusicapi import YTMusic

    yt = YTMusic()
    data = yt.get_playlist(playlist_id, limit=None)
    tracks = [t for t in data.get("tracks", []) if t.get("title")]
    titles = [t["title"] for t in tracks]

    counts: dict[str, int] = {}
    for title in titles:
        artist = extract_artist(title)
        if artist:
            counts[artist] = counts.get(artist, 0) + 1
    top = [a for a, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:15]]

    step = max(1, len(titles) // 24)
    sample = titles[::step][:24]
    return {
        "playlist_id": playlist_id,
        "title": data.get("title", "playlist"),
        "count": len(titles),
        "top_artists": top,
        "sample": sample,
    }


def _registry() -> dict:
    if PROFILES_FILE.exists():
        return json.loads(PROFILES_FILE.read_text())
    legacy = Path(__file__).with_name(".profile_cache.json")
    if legacy.exists():
        old = json.loads(legacy.read_text())
        pid = old.get("playlist_id")
        reg = {"profiles": ({pid: old} if pid else {}), "active": pid}
        PROFILES_FILE.write_text(json.dumps(reg, indent=2))
        return reg
    return {"profiles": {}, "active": None}


def _save(reg: dict) -> None:
    PROFILES_FILE.write_text(json.dumps(reg, indent=2))


def add_profile(playlist_id: str, refresh: bool = False) -> dict:
    reg = _registry()
    if playlist_id in reg["profiles"] and not refresh:
        return reg["profiles"][playlist_id]
    profile = build_profile(playlist_id)
    reg["profiles"][playlist_id] = profile
    reg["active"] = playlist_id
    _save(reg)
    return profile


def list_profiles() -> list[dict]:
    reg = _registry()
    return [{"playlist_id": pid, "title": p["title"], "count": p["count"]}
            for pid, p in reg["profiles"].items()]


def get_active() -> str | None:
    return _registry().get("active")


def remove_profile(playlist_id: str) -> None:
    reg = _registry()
    reg["profiles"].pop(playlist_id, None)
    if reg.get("active") == playlist_id:
        reg["active"] = next(iter(reg["profiles"]), None)
    _save(reg)


def combined_profile() -> dict:
    from collections import Counter

    profiles = list(_registry()["profiles"].values())
    if not profiles:
        raise SystemExit("no profiles yet")
    if len(profiles) == 1:
        return profiles[0]
    artist_score: Counter = Counter()
    samples: list[str] = []
    for p in profiles:
        for rank, artist in enumerate(p["top_artists"]):
            artist_score[artist] += max(1, 15 - rank)
        samples += p["sample"][:10]
    return {
        "playlist_id": "*",
        "title": "todos los géneros",
        "count": sum(p["count"] for p in profiles),
        "top_artists": [a for a, _ in artist_score.most_common(20)],
        "sample": samples[:30],
    }


def resolve_profile(selector: str | None) -> dict:
    reg = _registry()
    if selector in (None, "", "active"):
        selector = reg.get("active")
    if selector == "*":
        return combined_profile()
    profile = reg["profiles"].get(selector)
    if not profile:
        raise SystemExit(f"unknown profile {selector!r}; add one with --playlist-id")
    return profile


def profile_state(profile: dict) -> str:
    lines = [
        f'PLAYLIST: "{profile["title"]}" — {profile["count"]} tracks',
        "Recurring artists: " + ", ".join(profile["top_artists"]),
        "Sample of tracks:",
    ]
    lines += [f"- {t}" for t in profile["sample"]]
    return "\n".join(lines)


def resolve(query: str) -> dict:
    from ytmusicapi import YTMusic

    yt = YTMusic()
    video_id = None
    if query.startswith("http"):
        qs = parse_qs(urlparse(query).query)
        video_id = (qs.get("v") or [None])[0]
    if video_id:
        try:
            track = yt.get_watch_playlist(video_id, limit=1)["tracks"][0]
        except Exception:  # noqa: BLE001
            track = None
        if track:
            return {
                "title": track.get("title"),
                "artists": [a["name"] for a in track.get("artists", [])],
                "album": (track.get("album") or {}).get("name") if isinstance(track.get("album"), dict) else track.get("album"),
                "year": track.get("year"),
            }
    for flt in ("songs", None):
        try:
            res = yt.search(query, filter=flt, limit=1) if flt else yt.search(query, limit=1)
        except Exception:  # noqa: BLE001
            res = []
        if res:
            r = res[0]
            return {
                "title": r.get("title"),
                "artists": [a["name"] for a in r.get("artists", [])],
                "album": (r.get("album") or {}).get("name") if isinstance(r.get("album"), dict) else r.get("album"),
                "year": r.get("year"),
            }
    return {"title": query}


def get_threshold(jev: JevLocal, profile: dict, resolved: bool = False) -> int:
    """Auto-calibrate the listen/skip cutoff for this profile+model+format.

    Anchors: a few of the playlist's own tracks (known fit) vs fixed off-genre
    tracks (known non-fit). The threshold is the midpoint, so 55-ish always
    means the same thing regardless of model or how rich the input is.
    """
    key = f"{profile.get('playlist_id')}|{jev.model}|{'res' if resolved else 'raw'}"
    cache = json.loads(CALIB_FILE.read_text()) if CALIB_FILE.exists() else {}
    if key in cache:
        return cache[key]

    def prep(text: str) -> dict:
        if resolved:
            try:
                return resolve(text)
            except Exception:  # noqa: BLE001
                pass
        return {"title": text}

    positives = [score(jev, profile, prep(t), include_genre=False)["match"] for t in profile["sample"][:5]]
    negatives = [score(jev, profile, prep(t), include_genre=False)["match"] for t in NEG_ANCHORS]
    pos_mean = sum(positives) / len(positives)
    neg_mean = sum(negatives) / len(negatives)
    threshold = max(45, min(70, round((pos_mean + neg_mean) / 2)))
    cache[key] = threshold
    CALIB_FILE.write_text(json.dumps(cache, indent=2))
    return threshold


def track_state(track: dict) -> str:
    fields = [
        ("Title", track.get("title")),
        ("Artist(s)", ", ".join(track.get("artists") or [])),
        ("Album", track.get("album")),
        ("Year", track.get("year")),
    ]
    return "\n".join(f"{k}: {v}" for k, v in fields if v)


def score(jev: JevLocal, profile: dict, track: dict, include_genre: bool = True) -> dict:
    state = profile_state(profile) + "\n\nCANDIDATE TRACK:\n" + track_state(track)
    questions = {
        "fit": {
            "type": "score",
            "instructions": "How well would this track sit in this playlist?",
            "criteria": ["no fit", "weak", "decent", "strong"],
        },
        "add": {"type": "noul", "instructions": "Would you add this track to this playlist?"},
    }
    if include_genre:
        questions["genre"] = {
            "type": "choice",
            "instructions": "How closely does the candidate's genre match this playlist?",
            "criteria": {"same": "same subgenre", "adjacent": "adjacent, could work", "off": "different style"},
        }
    answers = jev.decide(state, questions)["answers"]
    fit_norm = answers["fit"]["score"] / 3.0
    result = {
        "match": round(fit_norm * 100),
        "add_probability": answers["add"]["noul"],
        "fit": answers["fit"]["score"],
        "fit_distribution": answers["fit"]["probabilities"],
        "title": track.get("title"),
        "artists": track.get("artists"),
        "videoId": track.get("videoId"),
    }
    if include_genre:
        result["genre"] = answers["genre"]["choice"]
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="?", help="track name or YouTube Music URL")
    ap.add_argument("--playlist-id", help="add this playlist to your profiles (mine or any public one)")
    ap.add_argument("--profile", help="profile to use: a playlist id, or '*' for the combined general profile")
    ap.add_argument("--list", action="store_true", help="list saved profiles")
    ap.add_argument("--refresh", action="store_true", help="rebuild the profile if it already exists")
    ap.add_argument("--raw", action="store_true", help="do not resolve the query on YouTube, use the text as-is")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="lmstudio")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.list:
        for p in list_profiles():
            print(f"  {p['playlist_id']}  {p['title']}  ({p['count']} tracks)")
        return 0
    if args.playlist_id:
        add_profile(args.playlist_id, args.refresh)
    profile = resolve_profile(args.profile)
    if not args.query:
        print(f'profile: "{profile["title"]}" — {profile["count"]} tracks '
              f'({len(profile["top_artists"])} artists cached)', file=sys.stderr)
        return 0

    base_url, model = BACKENDS[args.backend]
    jev = JevLocal(base_url=args.base_url or base_url, model=args.model or model)
    track = {"title": args.query} if args.raw else resolve(args.query)
    result = score(jev, profile, track)
    threshold = get_threshold(jev, profile, resolved=not args.raw)

    if args.json:
        result["threshold"] = threshold
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    verdict = "ESCUCHAR" if result["match"] >= threshold else "SALTAR"
    print(f"{result['title']} — {', '.join(result['artists'] or []) or '?'}")
    print(f"  MATCH {result['match']}/100  (umbral {threshold})   P(add)={result['add_probability']:.2f}   "
          f"genero={result['genre']}   -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
