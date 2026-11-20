import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="HF dataset dir from build_dataset.py")
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--num_train_epochs", type=float, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=-1, help="cap the number of optimizer steps")
    args = p.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

    from exprft.trainer import ExpRFTTrainer, WeightedCollator, dataset_token_normalizer

    cfg = yaml.safe_load(open(args.config))
    tcfg = dict(cfg["training"])
    for k in ("learning_rate", "num_train_epochs", "per_device_train_batch_size",
              "gradient_accumulation_steps"):
        if getattr(args, k) is not None:
            tcfg[k] = getattr(args, k)
    model_name = args.model or cfg["model"]

    set_seed(args.seed)  # before the model and LoRA adapters are built

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model

    lcfg = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], bias="none", task_type="CAUSAL_LM"))
    if tcfg.get("gradient_checkpointing", False):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = load_from_disk(args.data)
    per_device_batch = int(tcfg["per_device_train_batch_size"])
    normalizer = dataset_token_normalizer(dataset["weight"], dataset["n_response_tokens"], per_device_batch)

    training_args = TrainingArguments(
        output_dir=args.out,
        seed=args.seed, data_seed=args.seed,
        learning_rate=float(tcfg["learning_rate"]),
        num_train_epochs=float(tcfg["num_train_epochs"]),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", False)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        bf16=bool(tcfg.get("bf16", True)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.0)),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        max_steps=args.max_steps,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    if training_args.n_gpu > 1:
        raise SystemExit("one process per GPU is required (use `accelerate launch`), or set CUDA_VISIBLE_DEVICES to one GPU")
    trainer = ExpRFTTrainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=WeightedCollator(tokenizer.pad_token_id), tokenizer=tokenizer,
        normalizer=normalizer,
    )
    trainer.train()
    trainer.save_model(args.out)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.out)
        build_stats = Path(args.data) / "build_stats.json"
        provenance = {
            "data": str(Path(args.data).resolve()), "model": model_name, "seed": args.seed,
            "loss_normalizer": normalizer, "lora": cfg["lora"], "training": tcfg, "max_steps": args.max_steps,
            "build_stats": json.loads(build_stats.read_text()) if build_stats.exists() else None,
        }
        (Path(args.out) / "train_provenance.json").write_text(json.dumps(provenance, indent=2))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
