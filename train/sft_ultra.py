"""FlashMini Ultra — phase-0 SFT (Track B, frontier-plan.md).

QLoRA SFT of unsloth/Qwen3.5-4B on verified agent trajectories
(data/sft/flashmini_traces.jsonl), targeting Kaggle's free T4 x2
(30 h/week). Runs on Kaggle, not locally (no CUDA on the dev box).

T4 notes (Turing, no bf16/flash-attn):
- dtype float16 (NOT bfloat16), fp16=True in the config
- packing/padding_free stay OFF (padding_free requires FlashAttention)
- 4-bit load via bitsandbytes; LoRA r=16 keeps the adapter tiny

Message format matches the runtime loop exactly (system → user task →
assistant tool-call JSON → user tool-result envelope → ... → final),
so assistant_only_loss trains only on the model's own decisions.

Usage (Kaggle notebook cell):
    !pip install -q trl peft bitsandbytes datasets
    !python sft_ultra.py --dataset flashmini_traces.jsonl --epochs 3

Honest limitations (phase 0): ~72 verified samples is a pipeline
validation set, not a frontier delta. Scale comes from (a) many more
bench rounds, (b) the released agent corpora once disk allows, (c)
NIM-teacher distillation on failed tasks. The first measured gate for
any trained artifact is THIS repo's AgentBench, run against the merged
weights re-quantized to GGUF.
"""

from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "unsloth/Qwen3.5-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/sft/flashmini_traces.jsonl")
    parser.add_argument("--output", default="flashmini-ultra-sft-epoch")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset("json", data_files=args.dataset, split="train")

    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=SFTConfig(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=args.learning_rate,
            fp16=True,  # T4: no bf16
            logging_steps=1,
            save_strategy="epoch",
            report_to=[],
            max_length=args.max_length,
            packing=False,  # padding_free needs FlashAttention (absent on T4)
            assistant_only_loss=True,
            model_init_kwargs={
                "dtype": "float16",
                "quantization_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": "float16",
                    "bnb_4bit_use_double_quant": True,
                },
            },
        ),
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
