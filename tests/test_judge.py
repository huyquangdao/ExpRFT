import unittest

from exprft.judge import parse_scores

GOOD = """Accuracy:  8
The assistant reaches the right answer.

Reasoning ability:  7.5
Mostly logical.

Comprehensiveness:  6
Some gaps.

Pedagogical value:  7
Helpful.

Confidence calibration:  9
Appropriate.

Overall:  8
Good conversation overall."""


class TestParseScores(unittest.TestCase):
    def test_good(self):
        s = parse_scores(GOOD)
        self.assertEqual(s["overall"], 8.0)
        self.assertEqual(s["reasoning"], 7.5)
        self.assertEqual(len(s), 6)

    def test_markdown_bold_labels(self):
        self.assertEqual(parse_scores(GOOD.replace("Overall:  8", "**Overall:** 8"))["overall"], 8.0)

    def test_slash_ten_accepted(self):
        self.assertEqual(parse_scores(GOOD.replace("Overall:  8", "Overall: 8/10"))["overall"], 8.0)

    def test_out_of_range_rejected(self):
        self.assertIsNone(parse_scores(GOOD.replace("Overall:  8", "Overall: 100")))

    def test_non_decimal_rejected(self):
        self.assertIsNone(parse_scores(GOOD.replace("Overall:  8", "Overall: 1e1")))
        self.assertIsNone(parse_scores(GOOD.replace("Overall:  8", "Overall: eight")))

    def test_missing_label_rejected(self):
        self.assertIsNone(parse_scores(GOOD.replace("Overall:  8", "Final: 8")))

    def test_duplicate_label_rejected(self):
        self.assertIsNone(parse_scores(GOOD + "\nOverall: 3"))
        self.assertIsNone(parse_scores(GOOD + "\nOverall: bad"))


if __name__ == "__main__":
    unittest.main()
