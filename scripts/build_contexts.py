import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exprft.contexts import SUPPORTED, load_raw, sample_and_split, save_contexts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=SUPPORTED + ["all"], required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="data/contexts")
    p.add_argument("--n_train", type=int, default=400)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for ds in (SUPPORTED if args.dataset == "all" else [args.dataset]):
        raw = load_raw(ds, args.data_dir)
        train, test, splits = sample_and_split(raw, args.n_train, args.n_test, args.seed)
        save_contexts(args.out, ds, train, test, splits)
        print(f"{ds}: {len(raw)} raw -> train {len(train)}, test {len(test)} -> {Path(args.out) / ds}")


if __name__ == "__main__":
    main()
