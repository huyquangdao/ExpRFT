import os
import unittest

import torch

MODEL = os.environ.get("EXPRFT_TEST_TOKENIZER", "meta-llama/Llama-3.1-8B-Instruct")


def _tiny_lora_params(seed: int) -> list[torch.Tensor]:
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM, set_seed

    set_seed(seed)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1)
    model = get_peft_model(LlamaForCausalLM(cfg), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    return [p.detach().clone() for n, p in model.named_parameters() if "lora_" in n]


class TestSeededInit(unittest.TestCase):
    def test_same_seed_same_lora_init(self):
        a, b = _tiny_lora_params(1), _tiny_lora_params(1)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))

    def test_different_seed_different_lora_init(self):
        a, b = _tiny_lora_params(1), _tiny_lora_params(2)
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(a, b)))


class TestPrefixEncoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoTokenizer
            cls.tok = AutoTokenizer.from_pretrained(MODEL)
        except Exception:
            raise unittest.SkipTest(f"tokenizer {MODEL} not available")

    def test_single_bos_and_matches_training_prefix(self):
        from exprft.tokenization import encode_example, encode_prefix

        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        ids = encode_prefix(self.tok, msgs)
        bos = self.tok.bos_token_id
        self.assertEqual(ids[0], bos)
        self.assertNotEqual(ids[1], bos)
        enc = encode_example(self.tok, msgs, "hello", max_seq_length=512)
        n_prefix = len(enc["input_ids"]) - enc["n_response_tokens"]
        self.assertEqual(enc["input_ids"][:n_prefix], ids)

    def test_request_seed_is_distinct_per_request(self):
        from exprft.generation import request_seed

        seeds = {request_seed(0, f"c{i}", t, k) for i in range(50) for t in (0.3, 0.7, 1.0) for k in (1, 2, 3)}
        self.assertEqual(len(seeds), 450)
        self.assertEqual(request_seed(7, "c3", 0.7, 2), request_seed(7, "c3", 0.7, 2))


if __name__ == "__main__":
    unittest.main()
