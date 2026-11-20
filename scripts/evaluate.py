import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exprft.answer import parse_answer
from exprft.contexts import SUPPORTED, load_split
from exprft.generation import VLLMGenerator


def _mean_se(xs: list[float]) -> dict:
    n = len(xs)
    mean = statistics.mean(xs) if xs else 0.0
    sd = statistics.stdev(xs) if n > 1 else 0.0
    return {"mean": mean, "se": sd / math.sqrt(n) if n > 1 else 0.0, "n": n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=SUPPORTED, required=True)
    p.add_argument("--contexts", default="data/contexts")
    p.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for the base model")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num_turns", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge", default=None, help="judge model name, e.g. gpt-4o (needs OPENAI_API_KEY)")
    p.add_argument("--limit", type=int, default=None, help="use only the first N contexts")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    contexts = load_split(args.contexts, args.dataset, "test")[: args.limit]
    gen = VLLMGenerator(args.base_model, adapter_path=args.adapter, max_new_tokens=args.max_new_tokens,
                        tensor_parallel_size=args.tensor_parallel_size,
                        gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed)
    trajs = gen.generate(contexts, temperature=args.temperature, num_turns=args.num_turns)

    if args.judge:
        from exprft.judge import score_trajectories
        score_trajectories(trajs, model=args.judge)

    per_context = []
    for ctx, traj in zip(contexts, trajs):
        pred = parse_answer(traj.final_response(), valid_letters=ctx.choices.keys())
        row = {
            "context_id": ctx.id,
            "predicted": pred,
            "correct": ctx.correct_answer,
            "accuracy": float(pred is not None and pred == ctx.correct_answer.upper()),
            "truncated": traj.truncated,
            "turns": [{"teacher": t, "assistant": a} for t, a in traj.turns],
        }
        if args.judge:
            row["judge"] = traj.metadata.get("judge")
        per_context.append(row)

    summary = {
        "dataset": args.dataset, "split": "test", "adapter": args.adapter,
        "base_model": args.base_model, "temperature": args.temperature, "seed": args.seed,
        "accuracy": _mean_se([r["accuracy"] for r in per_context]),
        "unparsed_fraction": sum(r["predicted"] is None for r in per_context) / len(per_context),
        "truncated_turn_fraction": sum(sum(r["truncated"]) for r in per_context)
        / (len(per_context) * args.num_turns),
    }
    if args.judge:
        for k in ["overall", "accuracy", "reasoning", "comprehensiveness", "pedagogical", "confidence"]:
            vals = [r["judge"][k] for r in per_context if r.get("judge")]
            summary[f"judge_{k}"] = _mean_se(vals)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "per_context": per_context}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
