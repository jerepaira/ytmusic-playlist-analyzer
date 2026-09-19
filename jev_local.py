"""Local Jev-style decision layer over any OpenAI-compatible endpoint (Ollama, LM Studio, Atomic).

System One style: no text generation. For every question we build a lettered
multiple-choice prompt, request exactly ONE token with logprobs, and read the
model's probability mass on each option letter. That gives typed decisions
(choice / score / noul) with a probability distribution in a single pass.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request

OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SYSTEM_PROMPT = (
    "You are a strict decision function. You receive a STATE and one QUESTION "
    "with lettered options. Reply with ONLY the single letter of the best option. "
    "No words, no punctuation, no explanation."
)


class JevError(RuntimeError):
    pass


class JevLocal:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        model: str = "qwen2.5-coder:7b",
        api_key: str = "local",
        timeout: int = 180,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _chat(self, prompt: str, top_logprobs: int) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": max(top_logprobs, 1),
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise JevError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:400]}") from exc
        except Exception as exc:  # noqa: BLE001
            raise JevError(f"cannot reach {self.base_url}: {exc}") from exc

    @staticmethod
    def _letter_logprobs(data: dict, n_options: int) -> list[float] | None:
        try:
            entry = data["choices"][0]["logprobs"]["content"][0]
        except (KeyError, IndexError, TypeError):
            return None
        letters = OPTION_LETTERS[:n_options]
        logp: dict[str, float] = {}
        for cand in entry.get("top_logprobs") or []:
            tok = cand.get("token", "").strip().upper()
            if len(tok) == 1 and tok in letters and tok not in logp:
                logp[tok] = float(cand["logprob"])
        if not logp:
            return None
        # options the model did not surface get a floor below the lowest seen
        floor = min(logp.values()) - 25.0
        return [logp.get(L, floor) for L in letters]

    @staticmethod
    def _normalize(logprobs: list[float]) -> list[float]:
        m = max(logprobs)
        exps = [math.exp(v - m) for v in logprobs]
        total = sum(exps)
        return [e / total for e in exps]

    @staticmethod
    def _render(state: str, instructions: str, options: list[str]) -> str:
        lines = [f"STATE:\n{state}", "", f"QUESTION: {instructions}"]
        for i, opt in enumerate(options):
            lines.append(f"{OPTION_LETTERS[i]}) {opt}")
        lines.append("")
        lines.append("Answer with one letter:")
        return "\n".join(lines)

    def _ask(self, state: str, instructions: str, options: list[str]) -> list[float]:
        prompt = self._render(state, instructions, options)
        data = self._chat(prompt, top_logprobs=min(len(options) + 12, 20))
        logprobs = self._letter_logprobs(data, len(options))
        if logprobs is None:
            raise JevError(
                "backend returned no usable logprobs for option letters; "
                "enable logprobs (LM Studio: enable it and reload the model)"
            )
        return self._normalize(logprobs)

    def decide_choice(self, state: str, instructions: str, criteria: dict[str, str]) -> dict:
        keys = list(criteria.keys())
        options = [f"{k}: {criteria[k]}" for k in keys]
        probs = self._ask(state, instructions, options)
        best = max(range(len(keys)), key=lambda i: probs[i])
        return {
            "type": "choice",
            "choice": keys[best],
            "probabilities": {k: round(p, 6) for k, p in zip(keys, probs)},
            "confidence": round(probs[best], 6),
        }

    def decide_score(self, state: str, instructions: str, criteria: list[str]) -> dict:
        probs = self._ask(state, instructions, list(criteria))
        score = sum(i * p for i, p in enumerate(probs))
        best = max(range(len(criteria)), key=lambda i: probs[i])
        return {
            "type": "score",
            "score": round(score, 6),
            "legend": {str(i): c for i, c in enumerate(criteria)},
            "probabilities": {str(i): round(p, 6) for i, p in enumerate(probs)},
            "confidence": round(probs[best], 6),
        }

    def decide_noul(self, state: str, instructions: str) -> dict:
        probs = self._ask(state, instructions, ["yes", "no"])
        return {"type": "noul", "noul": round(probs[0], 6), "probabilities": {"yes": round(probs[0], 6), "no": round(probs[1], 6)}}

    def decide(self, state: str, questions: dict) -> dict:
        answers: dict[str, dict] = {}
        for qid, q in questions.items():
            qtype = q.get("type")
            instructions = q.get("instructions", "")
            if qtype == "choice":
                answers[qid] = self.decide_choice(state, instructions, q["criteria"])
            elif qtype == "score":
                answers[qid] = self.decide_score(state, instructions, q["criteria"])
            elif qtype == "noul":
                answers[qid] = self.decide_noul(state, instructions)
            else:
                raise JevError(f"unknown question type: {qtype!r}")
        return {"model": self.model, "answers": answers}


if __name__ == "__main__":
    jev = JevLocal()
    state = (
        "Track: 'Bohemian Rhapsody' by Queen. Album: A Night at the Opera (1975). "
        "Genre: progressive rock, art rock. Mood: operatic, dynamic, dramatic."
    )
    print(json.dumps(jev.decide(state, {
        "genre": {"type": "choice", "instructions": "Which genre best fits this track?",
                  "criteria": {"rock": "rock", "pop": "pop", "electronic": "electronic", "classical": "classical"}},
        "energy": {"type": "score", "instructions": "How energetic is this track?",
                   "criteria": ["calm", "mid", "high energy"]},
        "fits_rock_playlist": {"type": "noul", "instructions": "Would this track fit a classic rock playlist?"},
    }), indent=2))
