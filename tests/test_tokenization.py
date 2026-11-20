import os
import unittest

from exprft.conversation import Trajectory
from exprft.tokenization import encode_example, expand_trajectory

MODEL = os.environ.get("EXPRFT_TEST_TOKENIZER", "meta-llama/Llama-3.1-8B-Instruct")


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MODEL)
    except Exception:
        return None


class TestResponseOnlyLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = _load_tokenizer()
        if cls.tok is None:
            raise unittest.SkipTest(f"tokenizer {MODEL} not available")
        cls.traj = Trajectory(
            context_id="c0", context={}, temperature=0.7,
            turns=[("Question: 1+1?\nChoices:\nA. 1\nB. 2", "Let me think. It is two."),
                   ("Why are the others wrong?", "Because 1 is not 1+1."),
                   ("Final answer?", "The answer is B. Because 1+1=2.")],
        )

    def test_three_examples_one_per_assistant_turn(self):
        pairs = expand_trajectory(self.traj.messages())
        self.assertEqual(len(pairs), 3)
        self.assertEqual([r for _, r in pairs], [a for _, a in self.traj.turns])
        self.assertEqual(len(pairs[2][0]), 1 + 2 * 2 + 1)  # system + 2 full turns + teacher

    def test_labels_cover_exactly_the_response_plus_eot(self):
        tok = self.tok
        for prefix, response in expand_trajectory(self.traj.messages()):
            enc = encode_example(tok, prefix, response, max_seq_length=4096)
            self.assertIsNotNone(enc)
            ids, labels = enc["input_ids"], enc["labels"]
            self.assertEqual(len(ids), len(labels))
            labelled = [t for t in labels if t != -100]
            self.assertEqual(labelled, ids[-len(labelled):])          # a suffix of the input
            self.assertEqual(enc["n_response_tokens"], len(labelled))
            text = tok.decode(labelled, skip_special_tokens=False)
            self.assertTrue(text.startswith(response), text)
            self.assertIn(tok.decode([labelled[-1]]), (tok.eos_token, "<|eot_id|>", "<|im_end|>", "\n"))
            # nothing from the prefix is labelled: the prefix text decodes from the -100 region
            prefix_text = tok.decode(ids[:len(ids) - len(labelled)], skip_special_tokens=False)
            for role_msg in prefix:
                self.assertIn(role_msg["content"].strip(), prefix_text)
            self.assertNotIn(response, prefix_text)

    def test_earlier_assistant_turns_are_not_labelled(self):
        tok = self.tok
        prefix, response = expand_trajectory(self.traj.messages())[2]
        enc = encode_example(tok, prefix, response, max_seq_length=4096)
        labelled_text = tok.decode([t for t in enc["labels"] if t != -100], skip_special_tokens=True)
        self.assertNotIn("Let me think", labelled_text)
        self.assertNotIn("Because 1 is not", labelled_text)
        self.assertIn("The answer is B", labelled_text)

    def test_too_long_is_dropped_not_truncated(self):
        prefix, response = expand_trajectory(self.traj.messages())[0]
        self.assertIsNone(encode_example(self.tok, prefix, response, max_seq_length=8))


if __name__ == "__main__":
    unittest.main()
