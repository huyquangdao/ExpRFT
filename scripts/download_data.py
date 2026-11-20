import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HF_SOURCES = {
    "openbookqa": [("allenai/openbookqa", "main", "train+validation+test")],
    "arc": [("allenai/ai2_arc", "ARC-Easy", "train+validation+test"),
            ("allenai/ai2_arc", "ARC-Challenge", "train+validation+test")],
    "sciqa": [("derek-thomas/ScienceQA", None, "train+validation+test")],
    "mmlu": [("cais/mmlu", "all", "test+validation")],
    "mathdial": [("eth-nlped/mathdial", None, "train+test")],
}


def _jsonable(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                pass  # drop non-serialisable fields
        return out
    return obj


def _without_image_decoding(ds):
    """Replace image columns by Image(decode=False) so iteration needs no image
    library; returns (dataset, image column names)."""
    from datasets import Image

    cols = [k for k, f in ds.features.items() if isinstance(f, Image)]
    for k in cols:
        ds = ds.cast_column(k, Image(decode=False))
    return ds, cols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--datasets", nargs="+", default=list(HF_SOURCES))
    args = p.parse_args()

    from datasets import load_dataset

    for name in args.datasets:
        if name == "cosql":
            print("cosql: manual download required (see data/README.md)")
            continue
        out = Path(args.data_dir) / "raw" / name / "data.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(out, "w") as f:
            for repo, config, split in HF_SOURCES[name]:
                ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
                ds, image_cols = _without_image_decoding(ds)
                for item in ds:
                    rec = dict(item)
                    for k in image_cols:
                        rec[k] = rec[k] is not None  # True if the item has an image
                    rec = _jsonable(rec)
                    if name == "arc":
                        rec["difficulty"] = config
                    f.write(json.dumps(rec) + "\n")
                    n += 1
        print(f"{name}: {n} records -> {out}")


if __name__ == "__main__":
    main()
