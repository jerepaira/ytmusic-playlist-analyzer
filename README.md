# ytmusic-playlist-analyzer

Score any track against your YouTube Music playlists using a local decision model
([local-jev](https://github.com/jerepaira/local-jev)), with an auto-calibrated
listen/skip threshold and a small web UI.

## What it does

- builds a compact **taste profile** from one or more of your playlists
- **rates any track** (name or YouTube Music URL) against a profile: `0-100` match + verdict
- **scans a public playlist** and ranks its tracks against your profile
- **auto-calibrates** the pass threshold per profile/model, so the cutoff means
  the same thing across genres
- lets you keep several profiles (one per genre) and a combined "all genres" one

## Run the UI

```bash
pip install fastapi uvicorn ytmusicapi requests
uvicorn webui:app --port 8020
# open http://127.0.0.1:8020
```

Paste one of your playlist URLs to build a profile, then either measure a single
track or scan someone else's public playlist.

## CLI

```bash
python rate.py --playlist-id <playlist-id>          # add a profile
python rate.py "Guy J - Metal Dreams" --profile <id> # rate a track
python rate.py --list                                # list profiles
```

## How the score works

It reuses `local-jev`: the model answers a `score` (fit 0-3) and a `noul` (add?)
decision read from its logprobs. The displayed **MATCH** comes from the fit
distribution; the threshold is auto-calibrated from anchors (a few of your own
tracks vs fixed off-genre tracks). It is a **ranking signal**, not a calibrated
probability.

## Requires

A local Jev backend (Ollama or LM Studio) — see [local-jev](https://github.com/jerepaira/local-jev).
`jev_local.py` is included here.
