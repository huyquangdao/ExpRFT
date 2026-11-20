import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exprft.conversation import load_trajectories, save_trajectories
from exprft.judge import score_trajectories


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", required=True)
    p.add_argument("--judge_model", default="gpt-4o")
    p.add_argument("--max_concurrent", type=int, default=30)
    args = p.parse_args()

    trajs = load_trajectories(args.trajectories)
    n_failed = score_trajectories(trajs, model=args.judge_model, max_concurrent=args.max_concurrent)
    save_trajectories(trajs, args.trajectories)
    scored = [t.reward for t in trajs if t.reward is not None]
    print(f"{len(scored)}/{len(trajs)} scored ({n_failed} failed this run); "
          f"mean reward {sum(scored) / max(len(scored), 1):.3f} -> {args.trajectories}")


if __name__ == "__main__":
    main()
