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
import time

# v3: pin to ONE GPU. On T4 x2, plain `python script.py` gives Trainer
# n_gpu=2 -> TRL raises ValueError on nn.DataParallel (no distributed
# launcher). Single-GPU T4 is sufficient for 4B nf4 QLoRA + grad ckpt.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # unconditional: setdefault fails open (I4)
# v2: Kaggle egress flaked once (HF connect OSError at ~9.5 min). Widen the
# per-request timeout and retry every hub download below.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


def retry_hf(desc: str, fn, attempts: int = 4, sleep_s: float = 30.0):
    """Run a hub-touching callable with linear backoff; last attempt raises."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - any hub/conn error is retryable
            print(f"[ultra] {desc} attempt {i}/{attempts} failed: {exc}")
            if i == attempts:
                raise
            time.sleep(sleep_s)
    raise AssertionError("unreachable")


def ensure_model_cached(repo_id: str) -> None:
    """Pre-download model weights+config so later from_pretrained calls hit cache."""
    from huggingface_hub import snapshot_download

    retry_hf(f"snapshot_download({repo_id})", lambda: snapshot_download(repo_id))


TRACES_PATH = os.environ.get("TRACES_PATH", "/kaggle/input/flashmini-traces/flashmini_traces.jsonl")

BASE_MODEL = "unsloth/Qwen3.5-4B"
GATE_REPO = "balintayisaac/flashmini-ultra"  # pull happens via `kaggle kernels output`
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


def load_mix() -> Dataset:  # noqa: F821
    """Blend verified FlashMini traces with a streamed corpus subset.

    UltraData rows carry messages + tools columns. The chat template
    renders tool_calls from messages; the tools column is dropped in v1
    (per-sample template kwargs are not supported by SFTTrainer) —
    documented limitation, revisit if the gate shows tool-format errors.
    Rows that fail template rendering are filtered, not crashed on.
    """
    from datasets import Dataset, concatenate_datasets, load_dataset
    from transformers import AutoTokenizer

    ensure_model_cached(BASE_MODEL)
    tok = retry_hf("tokenizer(load_mix)", lambda: AutoTokenizer.from_pretrained(BASE_MODEL))

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


MIX_CACHE = "/kaggle/working/mix.jsonl"


def get_blended() -> Dataset:  # noqa: F821
    """Cached blend: the streamed collection is the flakiest, slowest stage,
    so persist it once and reuse on resume (audit C2/I13)."""
    from datasets import load_dataset

    if os.path.exists(MIX_CACHE):
        print(f"[ultra] resuming from cached mix: {MIX_CACHE}")
        return load_dataset("json", data_files=MIX_CACHE, split="train")
    blended = load_mix()
    blended.to_json(MIX_CACHE)
    return blended


def report_lengths(blended: Dataset) -> None:  # noqa: F821
    """Token-length distribution of the training mix (audit I12).

    Right-truncation at MAX_LENGTH cuts the FINAL assistant turn off
    overflowing samples — teaching mid-tool-call stops. Report the overflow
    count loudly so the limit is a measured decision, not a default.
    """
    import numpy as np
    from transformers import AutoTokenizer

    tok = retry_hf("tokenizer(lengths)", lambda: AutoTokenizer.from_pretrained(BASE_MODEL))
    sample = blended.select(range(min(500, len(blended))))
    lengths = [
        len(tok.encode(" ".join(m["content"] for m in row["messages"] if m.get("content"))))
        for row in sample
    ]
    arr = np.array(lengths)
    over = int((arr > MAX_LENGTH).sum())
    print(
        f"[ultra] token lengths n={len(arr)} p50={int(np.percentile(arr, 50))} "
        f"p90={int(np.percentile(arr, 90))} p99={int(np.percentile(arr, 99))} "
        f"max={int(arr.max())} over_{MAX_LENGTH}={over}/{len(arr)}"
    )


def train(blended: Dataset) -> None:  # noqa: F821
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
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    from transformers import AutoTokenizer

    ensure_model_cached(BASE_MODEL)
    tok = retry_hf("tokenizer(train)", lambda: AutoTokenizer.from_pretrained(BASE_MODEL))
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
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,  # Kaggle 9 h GPU cap: never lose >250 steps of work
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
    )
    ensure_model_cached(BASE_MODEL)  # SFTTrainer(string) loads from hub; cache it first
    assert torch.cuda.device_count() == 1, (
        f"expected exactly 1 CUDA device, got {torch.cuda.device_count()} "
        "(TRL raises ValueError on nn.DataParallel — audit I4)"
    )
    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=cfg,
        train_dataset=blended,
        peft_config=lora,
        model_init_kwargs={"quantization_config": quant, "dtype": torch.float16},
    )
    import glob as _glob

    checkpoints = sorted(_glob.glob("/kaggle/working/sft-out/checkpoint-*"))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"[ultra] resuming training from {latest}")
        trainer.train(resume_from_checkpoint=latest)
    else:
        trainer.train()

    adapter_dir = "/kaggle/working/adapter"
    trainer.save_model(adapter_dir)
    merged_dir = merge_and_push(adapter_dir)
    gguf_and_push(merged_dir, GATE_REPO)


def merge_and_push(adapter_dir: str) -> str:
    """Merge LoRA into a fresh fp16 base, save the FULL model dir, push.

    convert_hf_to_gguf.py needs config.json + tokenizer + safetensors in
    ONE directory; an adapter dir has none of the full weights, so the
    merged model is materialized to disk first (deepwiki-verified).
    Returns the merged dir for the GGUF stage.
    """
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    merged = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir, dtype=torch.float16, device_map="cpu"
    ).merge_and_unload()
    merged_dir = "/kaggle/working/merged"
    merged.save_pretrained(merged_dir)
    del merged
    import gc

    gc.collect()  # merged (~8 GB fp16) must not overlap reinject's shard load (I5)
    tok = retry_hf("tokenizer(merge)", lambda: AutoTokenizer.from_pretrained(BASE_MODEL))
    tok.save_pretrained(merged_dir)
    reinject_mtp(merged_dir)
    return merged_dir


def reinject_mtp(merged_dir: str) -> None:
    """Restore the frozen NextN/MTP tensors that the LoRA merge drops.

    transformers ignores `mtp.*` keys when loading the base model
    (keys_to_ignore_on_load_unexpected), so merge_and_unload + save_pretrained
    silently loses the MTP head. The local runtime runs `--spec-type
    draft-mtp` against NextN tensors EMBEDDED in the main GGUF (verified on
    the shipped Qwen3.5-4B GGUF: blk.32.nextn.* + nextn.predict_layers), so
    the converted model must carry them. The MTP head is frozen during QLoRA
    (LoRA targets only q/k/v/o/gate/up/down in the main stack), so the base
    checkpoint's mtp.* tensors ARE the trained model's MTP head. Rewrites the
    merged shards in place, updating the index.
    """
    import glob
    import json

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    base = retry_hf("snapshot_download(mtp)", lambda: snapshot_download(BASE_MODEL))
    mtp: dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(f"{base}/*.safetensors")):
        for key, tensor in load_file(shard).items():
            if key.startswith("mtp."):
                mtp[key] = tensor
    if not mtp:
        # Audit I8: silent continuation produced a NextN-less GGUF with green
        # logs. The draft-mtp gate semantics depend on these tensors.
        raise RuntimeError(
            "no mtp.* tensors in base checkpoint — refusing to build a "
            "NextN-less GGUF (check BASE_MODEL revision and *.safetensors)"
        )

    index_path = f"{merged_dir}/model.safetensors.index.json"
    shards = sorted(glob.glob(f"{merged_dir}/*.safetensors"))
    target = shards[-1]  # append MTP tensors to the last shard
    merged_tensors = load_file(target)
    merged_tensors.update(mtp)
    save_file(merged_tensors, target, metadata={"format": "pt"})
    if os.path.exists(index_path):
        with open(index_path) as fh:
            index = json.load(fh)
        last_shard = os.path.basename(target)
        for key in mtp:
            index["weight_map"][key] = last_shard
            index["metadata"]["total_size"] += mtp[key].numel() * mtp[key].element_size()
        with open(index_path, "w") as fh:
            json.dump(index, fh)
    print(f"[ultra] reinjected {len(mtp)} mtp.* tensors into {last_shard}")


def gguf_and_push(model_dir: str, repo: str) -> None:
    """Default (no --mtp) conversion embeds NextN tensors when the model dir
    carries them (verified locally: shipped qwen35 GGUF has blk.32.nextn.* +
    nextn.predict_layers and serves draft-mtp with no separate -md file).
    --mtp would instead emit an MTP-ONLY draft file (deepwiki-verified
    mtp_only=True path) — NOT what the gate needs.

    Working dir is ~20 GB: merged dir is consumed by conversion, f16 GGUF is
    dropped after quantize; only the Q4_K_M gate artifact remains.

    convert_hf_to_gguf.py is pure python; llama-quantize needs a CPU build.
    """
    import shutil
    import subprocess
    from pathlib import Path

    # Audit I7: a floating main makes the toolchain non-reproducible — pin
    # the same build the local runtime is verified against (b12001).
    LLAMA_TAG = "b12001"
    if not Path("llama.cpp").exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                LLAMA_TAG,
                "https://github.com/ggml-org/llama.cpp",
            ],
            check=True,
        )
    # Audit I6: build the quantize binary BEFORE conversion so a cmake
    # failure can't strike while ~16 GB of artifacts are resident, and keep
    # the f16 intermediate on /tmp (root fs) instead of the working quota.
    subprocess.run(
        [
            "cmake",
            "-S",
            "llama.cpp",
            "-B",
            "llama.cpp/build",
            "-DGGML_CUDA=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", "llama.cpp/build", "--target", "llama-quantize", "-j", "4"],
        check=True,
    )
    f16 = "/tmp/flashmini-ultra-f16.gguf"
    subprocess.run(
        [
            "python",
            "llama.cpp/convert_hf_to_gguf.py",
            "--outtype",
            "f16",
            "--outfile",
            f16,
            model_dir,
        ],
        check=True,
    )
    _assert_nextn_present(f16)
    shutil.rmtree(model_dir)  # conversion done; free ~8 GB before quantize
    q4 = "/kaggle/working/flashmini-ultra-q4km.gguf"
    subprocess.run(
        ["llama.cpp/build/bin/llama-quantize", f16, q4, "Q4_K_M"],
        check=True,
    )
    os.remove(f16)  # /tmp hygiene; the quota artifact is the Q4_K_M only
    print(f"[ultra] GGUF ready in /kaggle/working (HF push happens locally: {repo})")
    print("[ultra] NEXT: download the GGUF locally and run AgentBench — the gate.")


def _assert_nextn_present(gguf_path: str) -> None:
    """Audit I8: reinject+conversion can silently drop NextN tensors; the
    gate then either fails at llama-server startup or measures a different
    serving config. Fail loudly, here."""
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    names = [t.name for t in reader.tensors]
    if not any("nextn" in name for name in names):
        raise RuntimeError(
            "converted GGUF lacks nextn tensors — MTP head missing; "
            f"got {len(names)} tensors, e.g. {names[:5]}"
        )
    print(f"[ultra] nextn tensors present in GGUF ({len(names)} total)")


if __name__ == "__main__":
    blended = get_blended()
    print(f"[ultra] training on {len(blended)} samples")
    report_lengths(blended)
    train(blended)
