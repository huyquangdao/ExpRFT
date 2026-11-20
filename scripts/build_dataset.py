import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exprft.contexts import load_split
from exprft.conversation import load_trajectories
from exprft.tokenization import encode_example, expand_trajectory
from exprft.weighting import SIGMA_FLOOR, effective_sample_size, exprft_weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", required=True)
    p.add_argument("--contexts", default="data/contexts")
    p.add_argument("--dataset", required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--sigma_floor", type=float, default=SIGMA_FLOOR)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="tokenizer source")
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--keep_truncated", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from datasets import Dataset
    from transformers import AutoTokenizer

    keep_ids = {c.id for c in load_split(args.contexts, args.dataset, "train")}
    trajs = [t for t in load_trajectories(args.trajectories) if t.context_id in keep_ids]
    if any(t.reward is None for t in trajs):
        raise SystemExit("some trajectories have no reward; run compute_rewards.py first")

    weights = exprft_weights([t.reward for t in trajs], [t.context_id for t in trajs],
                             beta=args.beta, sigma_floor=args.sigma_floor)

    tok = AutoTokenizer.from_pretrained(args.model)
    rows, n_trunc_dropped, n_len_dropped = [], 0, 0
    for traj, w in zip(trajs, weights):
        if traj.any_truncated and not args.keep_truncated:
            n_trunc_dropped += 1
            continue
        for turn, (prefix, response) in enumerate(expand_trajectory(traj.messages()), 1):
            enc = encode_example(tok, prefix, response, args.max_seq_length)
            if enc is None:
                n_len_dropped += 1
                continue
            rows.append({**enc, "weight": w, "context_id": traj.context_id, "turn": turn})

    out = Path(args.out)
    Dataset.from_list(rows).save_to_disk(str(out))
    stats = {
        "dataset": args.dataset, "beta": args.beta, "sigma_floor": args.sigma_floor,
        "trajectories": len(trajs), "trajectories_dropped_truncated": n_trunc_dropped,
        "examples": len(rows), "examples_dropped_too_long": n_len_dropped,
        "weight_mean": sum(weights) / len(weights), "weight_max": max(weights),
        "weight_ess_fraction": effective_sample_size(weights) / len(weights),
        "response_tokens_total": sum(r["n_response_tokens"] for r in rows),
    }
    (out / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
