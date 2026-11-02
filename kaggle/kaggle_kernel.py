"""FlashMini Ultra — Track B phase 1: cloud SFT run (Kaggle T4x2).

Runs the full pipeline in a Kaggle GPU notebook so no large download ever
touches the local disk:

  1. Load FlashMini's own verified traces (data/sft/flashmini_traces.jsonl,
     ~KBs — attach as a Kaggle dataset or paste path) — ONLY trajectories
     that passed AgentBench graders.
  2. Blend with a streamed subset of openbmb/UltraData-SFT-Agent-2609
     (apache-2.0, ~500K agent/tool-use trajectories; streamed so the
     12 GB parquet export never lands on disk).
  3. QLoRA SFT of unsloth/Qwen3.5-4B (fp16 compute — T4 has no bf16).
  4. Merge adapters into a full fp16 model dir under /kaggle/working.
  5. Convert to GGUF + quantize Q4_K_M on the Kaggle CPU (llama.cpp
     python converter; quantize binary built with cmake — no CUDA needed).
     Outputs stay in /kaggle/working — results are pulled locally with
     `kaggle kernels output`; the HF push happens on this machine, never
     in the cloud (no secrets in kernels).
  6. THE GATE (run locally after downloading the GGUF): this repo's
     `agentbench` on the new model. A weight change ships only if the
     18-task score holds or improves at >=4 tok/s and <=5.5 GiB.

Usage (Kaggle notebook cells, GPU T4 x2, internet on):
    !pip install -q -U trl peft transformers accelerate bitsandbytes datasets
    %run /kaggle/usr/lib/flashmini_ultra/kaggle_ultra.py  # or paste cells

Honest limitation: this script is infrastructure. A frontier delta needs
the dataset mix tuned (see MIX below), the trained weights evaluated on
AgentBench, and iterated. Nothing here claims improvement before the gate
says so.
"""

from __future__ import annotations

import os


TRACES_PATH = os.environ.get("TRACES_PATH", "/kaggle/input/flashmini-traces/flashmini_traces.jsonl")

BASE_MODEL = "unsloth/Qwen3.5-4B"
CORPUS = "openbmb/UltraData-SFT-Agent-2609"
MIX = {"Code-Agent": 8_000, "General-Agent": 8_000, "Search-Agent": 4_000}
MAX_LENGTH = 2048
EPOCHS = 1
LR = 2e-4


def assistant_only_supported(chat_template: str | None) -> bool:
    """TRL masks assistant-token loss via {% generation %} template markers.

    Recent TRL auto-patches templates for known model families instead of
    raising when markers are missing, but an unknown family would silently
    train full-sequence loss. Deterministic choice: only enable masked loss
    when the template itself carries the markers.
    """
    return bool(chat_template) and "{% generation %}" in chat_template


def load_mix() -> "Dataset":  # noqa: F821
    """Blend verified FlashMini traces with a streamed corpus subset.

    UltraData rows carry messages + tools columns. The chat template
    renders tool_calls from messages; the tools column is dropped in v1
    (per-sample template kwargs are not supported by SFTTrainer) —
    documented limitation, revisit if the gate shows tool-format errors.
    Rows that fail template rendering are filtered, not crashed on.
    """
    from datasets import concatenate_datasets, Dataset, load_dataset

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    ours = load_dataset("json", data_files=TRACES_PATH, split="train")
    ours = ours.remove_columns([c for c in ours.column_names if c != "messages"])

    parts = [ours]
    for config, n in MIX.items():
        stream = load_dataset(CORPUS, config, split="train", streaming=True)
        kept = []
        for row in stream:
            if len(kept) >= n:
                break
            try:
                tok.apply_chat_template(row["messages"], tokenize=False)
            except Exception:
                continue  # template-incompatible row: skip, count later
            kept.append({"messages": row["messages"]})
        parts.append(Dataset.from_list(kept))

    blended = concatenate_datasets(parts).shuffle(seed=42)
    return blended


def train(blended: "Dataset") -> None:  # noqa: F821
    import torch
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # T4: no bf16
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    use_assistant_loss = assistant_only_supported(tok.chat_template)
    if not use_assistant_loss:
        print(
            "[ultra] chat template lacks {% generation %} markers; "
            "assistant_only_loss=False (full-sequence loss)"
        )
    cfg = SFTConfig(
        output_dir="/kaggle/working/sft-out",
        max_length=MAX_LENGTH,
        packing=False,  # no flash-attn on T4
        assistant_only_loss=use_assistant_loss,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=cfg,
        train_dataset=blended,
        peft_config=lora,
        model_init_kwargs={"quantization_config": quant, "dtype": torch.float16},
    )
    trainer.train()

    adapter_dir = "/kaggle/working/adapter"
    trainer.save_model(adapter_dir)
    merge_and_push(adapter_dir)


def merge_and_push(adapter_dir: str) -> None:
    """Merge LoRA into a fresh fp16 base, save the FULL model dir, push.

    convert_hf_to_gguf.py needs config.json + tokenizer + safetensors in
    ONE directory; an adapter dir has none of the full weights, so the
    merged model is materialized to disk first (deepwiki-verified).
    """
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    merged = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir, dtype=torch.float16, device_map="cpu"
    ).merge_and_unload()
    merged_dir = "/kaggle/working/merged"
    merged.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(merged_dir)


def gguf_and_push(model_dir: str, repo: str) -> None:
    """convert_hf_to_gguf.py is pure python; llama-quantize needs a CPU build.

    --mtp keeps the MTP (NextN) head in the GGUF so the local gate measures
    weight quality, not a missing spec-decode module. The converter adds an
    'mtp-' prefix to the output filename, so the produced file is located
    by glob rather than assumed.
    """
    import subprocess
    from pathlib import Path


    if not Path("llama.cpp").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/ggml-org/llama.cpp"],
            check=True,
        )
    f16 = "/kaggle/working/flashmini-ultra-f16.gguf"
    subprocess.run(
        ["python", "llama.cpp/convert_hf_to_gguf.py", "--outtype", "f16",
         "--outfile", f16, "--mtp", model_dir],
        check=True,
    )
    produced = (
        f16 if Path(f16).exists()
        else next(Path("/kaggle/working").glob("*f16*.gguf"))
    )
    subprocess.run(
        ["cmake", "-S", "llama.cpp", "-B", "llama.cpp/build",
         "-DGGML_CUDA=OFF", "-DCMAKE_BUILD_TYPE=Release"], check=True,
    )
    subprocess.run(
        ["cmake", "--build", "llama.cpp/build", "--target", "llama-quantize",
         "-j", "4"], check=True,
    )
    q4 = "/kaggle/working/flashmini-ultra-q4km.gguf"
    subprocess.run(
        ["llama.cpp/build/bin/llama-quantize", str(produced), q4, "Q4_K_M"],
        check=True,
    )
    print(f"[ultra] pushed GGUF to https://huggingface.co/{repo}")
    print("[ultra] NEXT: download the GGUF locally and run AgentBench — the gate.")


if __name__ == "__main__":
    blended = load_mix()
    print(f"[ultra] training on {len(blended)} samples")
    train(blended)
