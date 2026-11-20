from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED = ["openbookqa", "arc", "sciqa", "mmlu", "cosql", "mathdial"]
LETTERS = "ABCD"


@dataclass
class Context:
    id: str
    dataset: str
    question: str
    choices: dict[str, str]
    correct_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Context":
        return Context(
            id=d["id"],
            dataset=d["dataset"],
            question=d["question"],
            choices=dict(d["choices"]),
            correct_answer=d["correct_answer"],
            metadata=dict(d.get("metadata", {})),
        )


# --------------------------------------------------------------------------- #
# Raw loaders                                                                  #
# --------------------------------------------------------------------------- #

def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_openbookqa(raw_dir: Path) -> list[Context]:
    out = []
    for i, item in enumerate(_read_jsonl(raw_dir / "openbookqa" / "data.jsonl")):
        choices = dict(zip(item["choices"]["label"], item["choices"]["text"]))
        out.append(Context(f"openbookqa_{i}", "openbookqa", item["question_stem"], choices,
                           item["answerKey"], {"fact1": item.get("fact1", "")}))
    return out


def _load_arc(raw_dir: Path) -> list[Context]:
    """Items with exactly four options."""
    out = []
    for i, item in enumerate(_read_jsonl(raw_dir / "arc" / "data.jsonl")):
        labels, texts = item["choices"]["label"], item["choices"]["text"]
        if len(labels) != 4 or item["answerKey"] not in labels:
            continue
        choices = {LETTERS[j]: t for j, t in enumerate(texts)}
        correct = LETTERS[labels.index(item["answerKey"])]
        diff = item.get("difficulty", "ARC")
        out.append(Context(f"arc_{diff}_{i}", "arc", item["question"], choices, correct,
                           {"difficulty": diff}))
    return out


def _load_sciqa(raw_dir: Path) -> list[Context]:
    """Text-only items with exactly four options; the text context (hint) is appended
    to the question."""
    out = []
    for i, item in enumerate(_read_jsonl(raw_dir / "sciqa" / "data.jsonl")):
        # download_data.py stores "image" as a boolean: True if the item has an image.
        if item.get("image") is not False:
            continue
        opts = item.get("choices") or []
        if len(opts) != 4:
            continue
        idx = int(item.get("answer", -1))
        if not 0 <= idx < 4:
            continue
        hint = (item.get("hint") or "").strip()
        question = item["question"] + (f"\nContext: {hint}" if hint else "")
        out.append(Context(f"sciqa_{i}", "sciqa", question, {LETTERS[j]: c for j, c in enumerate(opts)},
                           LETTERS[idx], {"subject": item.get("subject", ""), "topic": item.get("topic", "")}))
    return out


def _load_mmlu(raw_dir: Path) -> list[Context]:
    out = []
    for i, item in enumerate(_read_jsonl(raw_dir / "mmlu" / "data.jsonl")):
        choices = {LETTERS[j]: c for j, c in enumerate(item["choices"])}
        out.append(Context(f"mmlu_{i}", "mmlu", item["question"], choices,
                           LETTERS[int(item["answer"])], {"subject": item["subject"]}))
    return out


def _load_cosql(raw_dir: Path) -> list[Context]:
    """CoSQL final-turn text-to-SQL; choices are assigned after the split."""
    data_file = raw_dir / "cosql" / "sql_state_tracking" / "cosql_train.json"
    tables_file = raw_dir / "cosql" / "cosql_dataset" / "tables.json"
    if not data_file.exists() or not tables_file.exists():
        raise FileNotFoundError(
            f"CoSQL must be downloaded manually from https://yale-lily.github.io/cosql into "
            f"{raw_dir / 'cosql'} (needs sql_state_tracking/cosql_train.json and cosql_dataset/tables.json)")
    data = json.load(open(data_file))
    schemas: dict[str, dict] = {}
    for t in json.load(open(tables_file)):
        names = t["table_names_original"]
        cols = t["column_names_original"]
        columns = {n: [] for n in names}
        for ti, cname in cols:
            if 0 <= ti < len(names):
                columns[names[ti]].append(cname)
        fks = []
        for a, b in t["foreign_keys"]:
            if a < len(cols) and b < len(cols):
                (ta, ca), (tb, cb) = cols[a], cols[b]
                if 0 <= ta < len(names) and 0 <= tb < len(names):
                    fks.append(f"{names[ta]}.{ca} = {names[tb]}.{cb}")
        schemas[t["db_id"]] = {"tables": names, "columns": columns, "foreign_keys": fks}

    out = []
    for i, item in enumerate(data):
        db = item.get("database_id", "")
        gold = item.get("final", {}).get("query", "").strip()
        nl = item.get("final", {}).get("utterance", "").strip()
        if not gold or not nl or db not in schemas:
            continue
        s = schemas[db]
        schema_str = "; ".join(f"{t}({', '.join(cs)})" for t, cs in s["columns"].items())
        question = (
            "Your task is to generate SQL query based on the following question and "
            f"information provided. {nl}\n"
            "Hint: Consider relationships between multiple tables.\n"
            f"The relevant schema is: {schema_str}.\n"
            f"The optional foreign keys are: {s['foreign_keys'] or '[]'}."
        )
        out.append(Context(f"cosql_{i}", "cosql", question, {}, "",
                           {"db_id": db, "gold": gold, "nl_question": nl}))
    return out


def _load_mathdial(raw_dir: Path) -> list[Context]:
    """MathDial problems; choices are assigned after the split."""
    out = []
    for i, item in enumerate(_read_jsonl(raw_dir / "mathdial" / "data.jsonl")):
        gold = (item.get("ground_truth") or "").strip()
        if not gold:
            continue
        out.append(Context(f"mathdial_{i}", "mathdial", item["question"], {}, "", {"gold": gold}))
    return out


def load_raw(dataset: str, data_dir: str | Path) -> list[Context]:
    raw_dir = Path(data_dir) / "raw"
    loaders = {"openbookqa": _load_openbookqa, "arc": _load_arc, "sciqa": _load_sciqa,
               "mmlu": _load_mmlu, "cosql": _load_cosql, "mathdial": _load_mathdial}
    if dataset not in loaders:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {SUPPORTED}")
    return loaders[dataset](raw_dir)


# --------------------------------------------------------------------------- #
# Distractors from a within-partition pool                                     #
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.strip().lower())


def needs_distractors(ctx: Context) -> bool:
    return not ctx.choices


def validate_choices(gold: str, distractors: list[str]) -> bool:
    """Exactly three non-empty distractors, pairwise distinct and different from the gold."""
    if len(distractors) != 3 or any(not d.strip() for d in distractors):
        return False
    keys = {_norm(d) for d in distractors}
    return len(keys) == 3 and _norm(gold) not in keys


def place_gold(gold: str, distractors: list[str], rng: random.Random) -> tuple[dict[str, str], str]:
    pos = rng.randint(0, 3)
    options = distractors[:pos] + [gold] + distractors[pos:]
    return {LETTERS[j]: options[j] for j in range(4)}, LETTERS[pos]


def assign_pool_distractors(contexts: list[Context], rng: random.Random) -> list[Context]:
    """Give each choice-less context three distractors drawn from the other contexts
    of this same list (same-DB first for CoSQL)."""
    if not any(needs_distractors(c) for c in contexts):
        return contexts
    golds = [(c.metadata["gold"], c.metadata.get("db_id")) for c in contexts]
    out = []
    for i, c in enumerate(contexts):
        if not needs_distractors(c):
            out.append(c)
            continue
        gold, db = c.metadata["gold"], c.metadata.get("db_id")
        same_db = [g for j, (g, d) in enumerate(golds) if j != i and d == db and _norm(g) != _norm(gold)]
        other = [g for j, (g, d) in enumerate(golds) if j != i and d != db and _norm(g) != _norm(gold)]
        rng.shuffle(same_db)
        rng.shuffle(other)
        chosen: list[str] = []
        seen = {_norm(gold)}
        for g in same_db + other:
            if _norm(g) not in seen:
                chosen.append(g)
                seen.add(_norm(g))
            if len(chosen) == 3:
                break
        if not validate_choices(gold, chosen):
            raise ValueError(f"{c.id}: cannot draw three distinct distractors from its partition")
        choices, correct = place_gold(gold, chosen, rng)
        out.append(Context(c.id, c.dataset, c.question, choices, correct,
                           {**c.metadata, "n_same_db_distractors": sum(g in same_db for g in chosen)}))
    return out


# --------------------------------------------------------------------------- #
# De-duplication and splitting                                                 #
# --------------------------------------------------------------------------- #

def question_key(ctx: Context) -> str:
    return _norm(ctx.metadata.get("nl_question") or ctx.question)


def dedupe(contexts: list[Context]) -> list[Context]:
    seen: set[str] = set()
    out = []
    for c in contexts:
        k = question_key(c)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def sample_and_split(
    contexts: list[Context],
    n_train: int = 400,
    n_test: int = 100,
    seed: int = 42,
) -> tuple[list[Context], list[Context], dict[str, list[str]]]:
    """Return (train, test, splits) with ``splits`` mapping train / test -> context ids."""
    contexts = dedupe(contexts)
    n_total = n_train + n_test
    if len(contexts) < n_total:
        raise ValueError(f"only {len(contexts)} unique contexts, need {n_total}")
    rng = random.Random(seed)
    sampled = rng.sample(contexts, n_total)
    train, test = sampled[:n_train], sampled[n_train:]

    train = assign_pool_distractors(train, rng)
    test = assign_pool_distractors(test, rng)

    splits = {
        "train": [c.id for c in train],
        "test": [c.id for c in test],
    }
    assert not {question_key(c) for c in train} & {question_key(c) for c in test}
    return train, test, splits


# --------------------------------------------------------------------------- #
# On-disk format: <contexts_dir>/<dataset>/{train.json,test.json,splits.json}  #
# --------------------------------------------------------------------------- #

def save_contexts(contexts_dir: str | Path, dataset: str, train: list[Context],
                  test: list[Context], splits: dict[str, list[str]]) -> None:
    d = Path(contexts_dir) / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.json").write_text(json.dumps([c.to_dict() for c in train], indent=2))
    (d / "test.json").write_text(json.dumps([c.to_dict() for c in test], indent=2))
    (d / "splits.json").write_text(json.dumps(splits, indent=2))


def load_split(contexts_dir: str | Path, dataset: str, split: str) -> list[Context]:
    """Load the contexts of one split: train | test."""
    d = Path(contexts_dir) / dataset
    splits = json.loads((d / "splits.json").read_text())
    if split not in splits:
        raise ValueError(f"unknown split {split!r}; have {list(splits)}")
    src = "test.json" if split == "test" else "train.json"
    by_id = {c["id"]: Context.from_dict(c) for c in json.loads((d / src).read_text())}
    return [by_id[i] for i in splits[split]]
