import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exprft.contexts import SUPPORTED, load_split
from exprft.conversation import save_trajectories
from exprft.generation import VLLMGenerator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=SUPPORTED, required=True)
    p.add_argument("--contexts", default="data/contexts")
    p.add_argument("--split", default="train")
    p.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.3, 0.7, 1.0])
    p.add_argument("--num_turns", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="use only the first N contexts")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    contexts = load_split(args.contexts, args.dataset, args.split)[: args.limit]
    gen = VLLMGenerator(args.base_model, max_new_tokens=args.max_new_tokens,
                        tensor_parallel_size=args.tensor_parallel_size,
                        gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed)
    trajectories = []
    for temp in args.temperatures:
        batch = gen.generate(contexts, temperature=temp, num_turns=args.num_turns)
        n_trunc = sum(t.any_truncated for t in batch)
        print(f"T={temp}: {len(batch)} trajectories, {n_trunc} with a truncated turn")
        trajectories.extend(batch)
    save_trajectories(trajectories, args.out)
    print(f"saved {len(trajectories)} trajectories -> {args.out}")


if __name__ == "__main__":
    main()
