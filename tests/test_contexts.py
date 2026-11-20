import random
import tempfile
import unittest

from exprft.contexts import (Context, assign_pool_distractors, dedupe, load_split, sample_and_split,
                             save_contexts, validate_choices)


def _ctx(i, q):
    return Context(id=f"d_{i}", dataset="d", question=q, choices={"A": "a", "B": "b"}, correct_answer="A")


def _pool_ctx(i, q, gold, db=None):
    md = {"gold": gold}
    if db is not None:
        md["db_id"] = db
    return Context(id=f"p_{i}", dataset="mathdial", question=q, choices={}, correct_answer="", metadata=md)


class TestSplits(unittest.TestCase):
    def test_dedupe_normalises_whitespace_and_case(self):
        cs = [_ctx(0, "What is  X?"), _ctx(1, "what is x?"), _ctx(2, "Other")]
        self.assertEqual([c.id for c in dedupe(cs)], ["d_0", "d_2"])

    def test_partitions(self):
        cs = [_ctx(i, f"q{i % 60}") for i in range(200)]
        train, test, splits = sample_and_split(cs, n_train=40, n_test=10, seed=1)
        self.assertEqual((len(train), len(test)), (40, 10))
        self.assertFalse({c.question for c in train} & {c.question for c in test})
        self.assertEqual(set(splits["train"]), {c.id for c in train})
        self.assertEqual(set(splits["test"]), {c.id for c in test})
        self.assertFalse(set(splits["train"]) & set(splits["test"]))

    def test_deterministic(self):
        cs = [_ctx(i, f"q{i}") for i in range(100)]
        self.assertEqual(sample_and_split(cs, 20, 5, seed=3)[2], sample_and_split(cs, 20, 5, seed=3)[2])

    def test_roundtrip(self):
        cs = [_ctx(i, f"q{i}") for i in range(30)]
        train, test, splits = sample_and_split(cs, 20, 5, seed=0)
        with tempfile.TemporaryDirectory() as d:
            save_contexts(d, "d", train, test, splits)
            self.assertEqual([c.id for c in load_split(d, "d", "test")], splits["test"])
            self.assertEqual([c.id for c in load_split(d, "d", "train")], splits["train"])
            self.assertEqual(len(load_split(d, "d", "train")), 20)


class TestPoolDistractors(unittest.TestCase):
    def test_validate_choices(self):
        self.assertTrue(validate_choices("g", ["a", "b", "c"]))
        self.assertFalse(validate_choices("g", ["a", "b"]))
        self.assertFalse(validate_choices("g", ["a", "a ", "c"]))
        self.assertFalse(validate_choices("g", ["a", "G", "c"]))
        self.assertFalse(validate_choices("g", ["a", "", "c"]))

    def test_distractors_come_only_from_the_same_list(self):
        cs = [_pool_ctx(i, f"q{i}", f"sol{i}") for i in range(10)]
        out = assign_pool_distractors(cs, random.Random(0))
        golds = {c.metadata["gold"] for c in cs}
        for c in out:
            self.assertEqual(c.choices[c.correct_answer], c.metadata["gold"])
            self.assertEqual(len(set(c.choices.values())), 4)
            self.assertTrue(set(c.choices.values()) <= golds)

    def test_same_db_preferred(self):
        cs = [_pool_ctx(i, f"q{i}", f"sql{i}", db="x" if i < 5 else "y") for i in range(10)]
        out = assign_pool_distractors(cs, random.Random(0))
        x_golds = {f"sql{i}" for i in range(5)}
        for c in out[:5]:
            self.assertTrue(set(c.choices.values()) <= x_golds)
            self.assertEqual(c.metadata["n_same_db_distractors"], 3)

    def test_too_small_pool_raises(self):
        with self.assertRaises(ValueError):
            assign_pool_distractors([_pool_ctx(i, f"q{i}", f"s{i}") for i in range(3)], random.Random(0))

    def test_partitions_do_not_share_gold_texts(self):
        cs = [_pool_ctx(i, f"q{i}", f"sol{i}") for i in range(300)]
        train, test, splits = sample_and_split(cs, n_train=100, n_test=30, seed=0)
        by_id = {c.id: c for c in train + test}
        parts = {name: [by_id[i] for i in ids] for name, ids in splits.items()}
        for a, b in (("train", "test"),):
            golds_b = {c.metadata["gold"] for c in parts[b]}
            golds_a = {c.metadata["gold"] for c in parts[a]}
            for c in parts[a]:
                self.assertFalse(set(c.choices.values()) & golds_b, c.id)
            for c in parts[b]:
                self.assertFalse(set(c.choices.values()) & golds_a, c.id)


if __name__ == "__main__":
    unittest.main()
