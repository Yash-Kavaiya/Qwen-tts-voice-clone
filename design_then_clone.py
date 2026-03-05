#!/usr/bin/env python3
"""
design_then_clone.py — Design a voice, then clone it for consistent reuse
=========================================================================
Two-step workflow:
  1. Use the VoiceDesign model to synthesize a short reference clip
     matching your target persona description.
  2. Feed that clip into the Base (clone) model to build a reusable
     voice prompt, then generate multiple sentences with it.

This is ideal when you want a **consistent character voice** across
many lines without having a real voice recording.

Usage:
  python design_then_clone.py \
      --persona "Male, 30s, warm baritone, calm and reassuring, like a therapist" \
      --ref-sentence "Hello, welcome. How are you feeling today?" \
      --text "Let's start with some breathing exercises." \
           "Take a deep breath in... and slowly out." \
           "Very good. You're doing great." \
      --language English
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Design a voice persona then clone it for multi-sentence generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--persona",
        required=True,
        help=(
            "Natural-language description of the voice persona. "
            "e.g. 'Female, early 20s, energetic, high-pitched, cheerful pop singer vibe'"
        ),
    )
    parser.add_argument(
        "--ref-sentence",
        required=True,
        help="A short sentence the designed voice will say to create the reference clip.",
    )
    parser.add_argument(
        "--ref-language",
        default="English",
        help="Language for the reference sentence (default: English).",
    )
    parser.add_argument(
        "--text",
        nargs="+",
        required=True,
        help="Sentences to generate in the designed & cloned voice.",
    )
    parser.add_argument(
        "--language",
        nargs="+",
        default=None,
        help="Language(s) for each target sentence (default: Auto).",
    )
    parser.add_argument(
        "--design-model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="VoiceDesign model name or path.",
    )
    parser.add_argument(
        "--clone-model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Base (clone) model name or path.",
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
        "--save-reference",
        action="store_true",
        default=True,
        help="Also save the designed reference clip (default: True).",
    )
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def get_attn_impl(device: str) -> str:
    if "cuda" in device:
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            print("⚠  Flash Attention 2 not found — using SDPA.")
            return "sdpa"
    else:
        print("   Using eager attention (CPU mode).")
        return "eager"


def main() -> None:
    args = parse_args()
    texts = args.text

    if args.language:
        if len(args.language) == 1 and len(texts) > 1:
            languages = args.language * len(texts)
        elif len(args.language) != len(texts):
            print(f"❌ --language count ({len(args.language)}) must match --text count ({len(texts)}).")
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

    dtype = get_dtype(args.dtype)
    attn_impl = get_attn_impl(args.device)

    if args.device == "cpu":
        print("⚠  Running on CPU — this will be slow but functional.")

    # ════════════════════════════════════════════════════════
    # STEP 1: Design the voice
    # ════════════════════════════════════════════════════════
    print("=" * 56)
    print("  STEP 1 — Design the Voice")
    print("=" * 56)
    print(f"🎨 Persona: {args.persona}")
    print(f"📝 Reference: \"{args.ref_sentence}\"\n")

    from qwen_tts import Qwen3TTSModel

    design_model = Qwen3TTSModel.from_pretrained(
        args.design_model,
        device_map=args.device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    ref_wavs, sr = design_model.generate_voice_design(
        text=args.ref_sentence,
        language=args.ref_language,
        instruct=args.persona,
    )

    if args.save_reference:
        ref_path = Path(args.output_dir) / "reference_designed_voice.wav"
        sf.write(str(ref_path), ref_wavs[0], sr)
        print(f"   💾 Reference clip saved: {ref_path}")

    # Free VoiceDesign model memory
    del design_model
    torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════
    # STEP 2: Build clone prompt from the designed voice
    # ════════════════════════════════════════════════════════
    print(f"\n{'=' * 56}")
    print("  STEP 2 — Clone the Designed Voice")
    print("=" * 56)

    clone_model = Qwen3TTSModel.from_pretrained(
        args.clone_model,
        device_map=args.device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    print("🔗 Building reusable voice-clone prompt...")
    voice_clone_prompt = clone_model.create_voice_clone_prompt(
        ref_audio=(ref_wavs[0], sr),
        ref_text=args.ref_sentence,
    )
    print("✅ Clone prompt ready.\n")

    # ════════════════════════════════════════════════════════
    # STEP 3: Generate all sentences
    # ════════════════════════════════════════════════════════
    print(f"{'=' * 56}")
    print(f"  STEP 3 — Generate {len(texts)} Sentence(s)")
    print("=" * 56)

    wavs, sr = clone_model.generate_voice_clone(
        text=texts if len(texts) > 1 else texts[0],
        language=languages if len(languages) > 1 else languages[0],
        voice_clone_prompt=voice_clone_prompt,
    )

    for i, wav in enumerate(wavs):
        filepath = Path(args.output_dir) / f"clone_{i + 1}.wav"
        sf.write(str(filepath), wav, sr)
        print(f"   💾 Saved: {filepath}")

    print(f"\n✅ Done! {len(wavs)} file(s) + reference clip saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
