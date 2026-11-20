from __future__ import annotations

import asyncio
import re

from exprft.conversation import Trajectory
from exprft.prompts import judge_prompt

SCORE_KEYS = ["accuracy", "reasoning", "comprehensiveness", "pedagogical", "confidence", "overall"]
_LABELS = {
    "accuracy": "Accuracy",
    "reasoning": "Reasoning ability",
    "comprehensiveness": "Comprehensiveness",
    "pedagogical": "Pedagogical value",
    "confidence": "Confidence calibration",
    "overall": "Overall",
}
_LABEL_LINE = {k: re.compile(rf"^\s*\**{re.escape(v)}\**\s*:(.*)$", re.I | re.M) for k, v in _LABELS.items()}
_VALUE = re.compile(r"^\s*\**\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?\s*\**\s*$")


def parse_scores(text: str) -> dict[str, float] | None:
    """Return the six scores as floats in [0, 10], or None if any is missing / invalid."""
    scores = {}
    for key, pat in _LABEL_LINE.items():
        lines = pat.findall(text)
        if len(lines) != 1:
            return None
        m = _VALUE.match(lines[0])
        if not m:
            return None
        val = float(m.group(1))
        if not 0.0 <= val <= 10.0:
            return None
        scores[key] = val
    return scores


async def _score_one(client, semaphore, traj: Trajectory, model: str, max_retries: int,
                     errors: list[str]) -> dict | None:
    system, user = judge_prompt(traj.context, traj.transcript())
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                        temperature=0.0,
                    ),
                    timeout=90.0,
                )
                raw = resp.choices[0].message.content or ""
                scores = parse_scores(raw)
                if scores is not None:
                    return {"model": model, "response_model": getattr(resp, "model", None), "raw": raw, **scores}
                errors.append(f"unparseable judge output: {raw[:200]!r}")
            except Exception as e:  # noqa: BLE001 - recorded and retried
                errors.append(f"{type(e).__name__}: {str(e)[:200]}")
                if "401" in str(e) or "invalid_api_key" in str(e):
                    return None
            await asyncio.sleep(2.0 * (2 ** attempt))
    return None


async def _score_all(trajectories, model, max_concurrent, max_retries):
    from openai import AsyncOpenAI

    from exprft.env import require

    require("OPENAI_API_KEY")
    for t in trajectories:
        if t.reward is not None and t.metadata.get("judge", {}).get("model") != model:
            raise RuntimeError(
                f"{t.context_id}: already scored by judge {t.metadata.get('judge', {}).get('model')!r}; "
                f"a trajectories file holds rewards from one judge model only")
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(max_concurrent)
    pending = [t for t in trajectories if t.reward is None]
    errors: list[str] = []
    results = await asyncio.gather(
        *[_score_one(client, sem, t, model, max_retries, errors) for t in pending])
    n_failed = 0
    for t, res in zip(pending, results):
        if res is None:
            n_failed += 1
            continue
        t.reward = res["overall"] / 10.0
        t.metadata["judge"] = res
    if pending and n_failed == len(pending):
        raise RuntimeError(f"every judge call failed; last error: {errors[-1] if errors else 'unknown'}")
    if errors:
        print(f"judge: {n_failed} trajectories unscored; last error: {errors[-1]}")
    return n_failed


def score_trajectories(trajectories: list[Trajectory], model: str = "gpt-4o",
                       max_concurrent: int = 30, max_retries: int = 4) -> int:
    """Fill ``reward`` in place for trajectories that have none. Returns #failures."""
    return asyncio.run(_score_all(trajectories, model, max_concurrent, max_retries))
