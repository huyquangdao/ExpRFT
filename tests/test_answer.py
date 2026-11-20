import unittest

from exprft.answer import parse_answer


class TestParseAnswer(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_answer("The answer is B. Because 1+1=2."), "B")
        self.assertEqual(parse_answer("The answer is B because 1+1=2."), "B")
        self.assertEqual(parse_answer("The answer is B"), "B")

    def test_formatting_variants(self):
        self.assertEqual(parse_answer("The answer is **B**. Because it is correct."), "B")
        self.assertEqual(parse_answer("THE ANSWER IS B."), "B")
        self.assertEqual(parse_answer("the answer is: (C)"), "C")
        self.assertEqual(parse_answer("The answer is 'D'"), "D")

    def test_first_statement_wins(self):
        txt = "The answer is B. Some people say 'The answer is A' but that is not right."
        self.assertEqual(parse_answer(txt), "B")

    def test_invalid_letter_skipped(self):
        self.assertIsNone(parse_answer("The answer is E."))
        self.assertEqual(parse_answer("The answer is E. Actually, the answer is C."), "C")

    def test_prose_and_malformed_labels_rejected(self):
        self.assertIsNone(parse_answer("The answer is a difficult question."))
        self.assertIsNone(parse_answer("The answer is B2."))
        self.assertIsNone(parse_answer("The answer is B_C."))
        self.assertIsNone(parse_answer("The answer is Boston."))
        self.assertIsNone(parse_answer("I cannot determine the answer; vitamin A"))
        self.assertIsNone(parse_answer("Probably B"))

    def test_empty(self):
        self.assertIsNone(parse_answer(""))
        self.assertIsNone(parse_answer(None))

    def test_custom_valid_letters(self):
        self.assertEqual(parse_answer("The answer is E.", valid_letters="ABCDE"), "E")


if __name__ == "__main__":
    unittest.main()
