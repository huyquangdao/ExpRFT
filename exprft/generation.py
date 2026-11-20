from __future__ import annotations

import hashlib
from pathlib import Path

from exprft.contexts import Context
from exprft.conversation import Trajectory
from exprft.prompts import ASSISTANT_SYSTEM_PROMPT, teacher_prompt
from exprft.tokenization import encode_prefix


def request_seed(seed: int, context_id: str, temperature: float, turn: int) -> int:
    h = hashlib.sha256(f"{seed}|{context_id}|{temperature:.4f}|{turn}".encode()).digest()
    return int.from_bytes(h[:4], "big") % (2 ** 31 - 1)


class VLLMGenerator:
    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        max_new_tokens: int = 1024,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        seed: int | None = None,
        max_lora_rank: int = 64,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.max_new_tokens = max_new_tokens
        self.seed = seed
        engine_kwargs = {} if seed is None else {"seed": seed}
        self.lora_request = None
        if adapter_path is not None:
            if not (Path(adapter_path) / "adapter_config.json").exists():
                raise FileNotFoundError(f"no adapter_config.json in {adapter_path}")
            self.llm = LLM(model=base_model, enable_lora=True, max_lora_rank=max_lora_rank,
                           tensor_parallel_size=tensor_parallel_size,
                           gpu_memory_utilization=gpu_memory_utilization,
                           max_model_len=max_model_len, **engine_kwargs)
            self.lora_request = LoRARequest("adapter", 1, str(adapter_path))
        else:
            self.llm = LLM(model=base_model, tensor_parallel_size=tensor_parallel_size,
                           gpu_memory_utilization=gpu_memory_utilization,
                           max_model_len=max_model_len, **engine_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

    def generate(self, contexts: list[Context], temperature: float, num_turns: int = 3) -> list[Trajectory]:
        from vllm import SamplingParams

        convs = [[{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}] for _ in contexts]
        turns: list[list[tuple[str, str]]] = [[] for _ in contexts]
        truncated: list[list[bool]] = [[] for _ in contexts]

        for turn in range(1, num_turns + 1):
            teacher_msgs = [teacher_prompt(turn, c.to_dict(), num_turns) for c in contexts]
            prompts = []
            for conv, msg in zip(convs, teacher_msgs):
                conv.append({"role": "user", "content": msg})
                prompts.append({"prompt_token_ids": encode_prefix(self.tokenizer, conv)})
            params = [
                SamplingParams(
                    temperature=temperature, max_tokens=self.max_new_tokens,
                    seed=None if self.seed is None else request_seed(self.seed, c.id, temperature, turn),
                )
                for c in contexts
            ]
            outputs = self.llm.generate(prompts, params, lora_request=self.lora_request)
            for i, out in enumerate(outputs):
                comp = out.outputs[0]
                text = comp.text.strip()
                convs[i].append({"role": "assistant", "content": text})
                turns[i].append((teacher_msgs[i], text))
                truncated[i].append(comp.finish_reason == "length")

        return [
            Trajectory(context_id=c.id, context=c.to_dict(), turns=turns[i],
                       temperature=temperature, truncated=truncated[i],
                       metadata={"generation_seed": self.seed, "max_new_tokens": self.max_new_tokens})
            for i, c in enumerate(contexts)
        ]
