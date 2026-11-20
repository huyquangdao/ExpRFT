import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exprft.contexts import _norm, place_gold, validate_choices

COSQL_PROMPT = """\
You are an expert SQL instructor. Given a natural language question, database schema, and the correct SQL query, generate exactly 3 plausible but INCORRECT SQL queries.

Requirements for each distractor:
- Must use tables/columns from the SAME schema (syntactically valid for this database)
- Must be structurally similar to the correct query (similar length, uses JOINs if correct does, etc.)
- Must return WRONG results (e.g., wrong columns selected, wrong JOIN condition, wrong WHERE filter, wrong aggregation, missing GROUP BY, etc.)
- Each distractor should have a DIFFERENT type of error

Question: {question}
Correct SQL: {gold}

Respond with ONLY a JSON array of exactly 3 incorrect SQL strings, no explanation:
["SELECT ...", "SELECT ...", "SELECT ..."]"""

MATHDIAL_PROMPT = """\
You are a math tutor. Given a math question and the correct step-by-step solution, generate exactly 3 plausible but INCORRECT solutions.

Requirements for each distractor:
- Must address the SAME question (not a different problem)
- Must follow a similar step-by-step reasoning structure
- Must arrive at a WRONG final answer
- Each distractor should contain a DIFFERENT type of error (e.g., wrong operation, arithmetic mistake, misread the problem, wrong unit conversion, etc.)
- Keep each solution roughly the same length as the correct one

Question: {question}
Correct solution: {gold}

Respond with ONLY a JSON array of exactly 3 incorrect solution strings, no explanation:
["solution1...", "solution2...", "solution3..."]"""


def _prompt(ctx: dict) -> str:
    tpl = COSQL_PROMPT if ctx["dataset"] == "cosql" else MATHDIAL_PROMPT
    return tpl.format(question=ctx["question"], gold=ctx["metadata"]["gold"])


async def _distractors(client, sem, prompt, model, max_retries=3) -> list[str]:
    async with sem:
        for _ in range(max_retries):
            try:
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    temperature=0.7, max_tokens=1024)
                text = (r.choices[0].message.content or "").strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                arr = json.loads(text)
                if isinstance(arr, list) and len(arr) >= 3:
                    return [str(d) for d in arr[:3]]
            except Exception:
                pass
    return []


async def _rewrite(paths: dict[str, Path], model: str, max_concurrent: int, rng: random.Random) -> dict:
    from openai import AsyncOpenAI

    from exprft.env import require

    require("OPENAI_API_KEY")
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(max_concurrent)
    data = {split: json.loads(p.read_text()) for split, p in paths.items()}
    all_golds = {_norm(c["metadata"]["gold"]) for cs in data.values() for c in cs}
    failures: dict[str, list[str]] = {}
    for split, contexts in data.items():
        results = await asyncio.gather(*[_distractors(client, sem, _prompt(c), model) for c in contexts])
        failures[split] = []
        for ctx, ds in zip(contexts, results):
            gold = ctx["metadata"]["gold"]
            other_golds = all_golds - {_norm(gold)}
            if not validate_choices(gold, ds) or any(_norm(d) in other_golds for d in ds):
                failures[split].append(ctx["id"])
                continue
            ctx["choices"], ctx["correct_answer"] = place_gold(gold, ds, rng)
            ctx["metadata"]["distractors"] = model
        paths[split].write_text(json.dumps(contexts, indent=2))
    return failures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cosql", "mathdial"], required=True)
    p.add_argument("--contexts", default="data/contexts")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--max_concurrent", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d = Path(args.contexts) / args.dataset
    paths = {s: d / f"{s}.json" for s in ("train", "test")}
    failures = asyncio.run(_rewrite(paths, args.model, args.max_concurrent, random.Random(args.seed)))
    (d / "hard_distractor_failures.json").write_text(json.dumps(failures, indent=2))
    for split, ids in failures.items():
        print(f"{args.dataset}/{split}: {len(ids)} contexts kept pool distractors (listed in hard_distractor_failures.json)")


if __name__ == "__main__":
    main()
