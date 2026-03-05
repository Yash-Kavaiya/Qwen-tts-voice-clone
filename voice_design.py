#!/usr/bin/env python3
"""
voice_design.py — Design a voice from a natural language description
====================================================================
Describe the voice you want (age, gender, tone, emotion, accent) and
Qwen3-TTS will synthesize speech matching that description.

Usage:
  # Design a young female voice
  python voice_design.py \
      --text "Welcome to our podcast! Today we're discussing AI." \
      --instruct "Young woman, 25 years old, warm and friendly alto voice, slightly breathy" \
      --language English

  # Batch: design multiple voices at once
  python voice_design.py \
      --text "Hello there!" "Bonjour!" \
      --instruct "Deep male voice, 40s, authoritative news anchor" "Young French woman, cheerful" \
      --language English French

Model: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
Hardware: NVIDIA GPU with >=16 GB VRAM recommended
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Design a voice using Qwen3-TTS VoiceDesign model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--text",
        nargs="+",
        required=True,
        help="One or more sentences to speak.",
    )
    parser.add_argument(
        "--instruct",
        nargs="+",
        required=True,
        help=(
            "Natural-language description of the desired voice for each sentence. "
            "e.g. 'Young woman, warm tone, slightly breathy'"
        ),
    )
    parser.add_argument(
        "--language",
        nargs="+",
        default=None,
        help=(
            "Language(s) for each sentence. Supported: Chinese, English, Japanese, "
            "Korean, German, French, Russian, Portuguese, Spanish, Italian. "
            "Defaults to 'Auto'."
        ),
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="Model name or local path.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: auto-detect cuda:0 or cpu).",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["bfloat16", "float16", "float32"],
        help="Model precision (default: bfloat16 on GPU, float32 on CPU).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output/).",
    )
    parser.add_argument(
        "--output-prefix",
        default="design",
        help="Filename prefix (default: design).",
    )
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def main() -> None:
    args = parse_args()

    texts = args.text
    instructs = args.instruct

    # ── Validate ─────────────────────────────────────────────
    if len(instructs) == 1 and len(texts) > 1:
        instructs = instructs * len(texts)
    elif len(instructs) != len(texts):
        print(f"❌ Number of --instruct values ({len(instructs)}) must match --text ({len(texts)}).")
        sys.exit(1)

    if args.language:
        if len(args.language) == 1 and len(texts) > 1:
            languages = args.language * len(texts)
        elif len(args.language) != len(texts):
            print(f"❌ Number of --language values ({len(args.language)}) must match --text ({len(texts)}).")
            sys.exit(1)
        else:
            languages = args.language
    else:
        languages = ["Auto"] * len(texts)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Auto-detect device and dtype ─────────────────────────
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.dtype is None:
        args.dtype = "bfloat16" if "cuda" in args.device else "float32"

    # ── Load model ───────────────────────────────────────────
    print(f"🔄 Loading model: {args.model}")
    print(f"   Device: {args.device}  |  Dtype: {args.dtype}")
    if args.device == "cpu":
        print("⚠  Running on CPU — this will be slow but functional.")

    from qwen_tts import Qwen3TTSModel

    attn_impl = "eager"
    if "cuda" in args.device:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
            print("⚠  Flash Attention 2 not found — using SDPA.")
    else:
        print("   Using eager attention (CPU mode).")

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=get_dtype(args.dtype),
        attn_implementation=attn_impl,
    )
    print("✅ Model loaded.\n")

    # ── Generate ─────────────────────────────────────────────
    if len(texts) == 1:
        print(f'🎨 Designing voice: "{instructs[0]}"')
        print(f'🗣️  Generating: "{texts[0]}"')
        wavs, sr = model.generate_voice_design(
            text=texts[0],
            language=languages[0],
            instruct=instructs[0],
        )
    else:
        print(f"🎨 Designing {len(texts)} voices in batch...")
        wavs, sr = model.generate_voice_design(
            text=texts,
            language=languages,
            instruct=instructs,
        )

    # ── Save ─────────────────────────────────────────────────
    for i, wav in enumerate(wavs):
        filename = f"{args.output_prefix}_{i + 1}.wav"
        filepath = Path(args.output_dir) / filename
        sf.write(str(filepath), wav, sr)
        print(f"   💾 Saved: {filepath}")

    print(f"\n✅ Done! {len(wavs)} file(s) saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
