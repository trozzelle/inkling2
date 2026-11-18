#!/usr/bin/env bash
# scripts/build_calib.sh: rebuild Baekpica's Inkling 2 calibration and held-out sets and check them by sha256.
set -euo pipefail
cd "$(dirname "$0")/.."
REV=f69fa26848416b8f69a0576ef29278b0d3e92d2d
M=data/model-files
W=data/calib
mkdir -p "$M" "$W"
uv run hf download thinkingmachines/Inkling-2-Small-MXFP8 tokenizer.json tokenizer_config.json \
  special_tokens_map.json chat_template.jinja processor_config.json config.json \
  --revision "$REV" --local-dir "$M"
B=vendor/ds4-dfm-rs/tools/vq_calib/build_calib_inkling2.py
uv run --group ref python -I "$B" fetch        --model "$M" --out "$W"
uv run --group ref python -I "$B" build        --model "$M" --out "$W"
uv run --group ref python -I "$B" build-images --model "$M" --out "$W"
uv run --group ref python -I "$B" check        --model "$M" --out "$W"
curl -fsSL https://huggingface.co/Baekpica/Inkling-2-Small-VQ-GGUF/raw/main/calibration/SHA256SUMS > "$W/SHA256SUMS.baekpica"
# Data files only: the manifests embed the transformers version string (5.23.0 here, 5.23.0.dev0 @ 109a7d19 in
# the published build) and differ in that string alone.
(cd "$W" && grep -E 'safetensors|samples_decoded' SHA256SUMS.baekpica | shasum -a 256 -c)
