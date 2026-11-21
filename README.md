# ink2q

Tooling for serving `thinkingmachines/Inkling-2-Small` (594B MoE: 48 layers, 384 routed experts, top-8) on one 256 GB
Apple Silicon Mac with EXL3 routed experts, and for checking that port against an MLX reference.

- **Quantization and fidelity (MLX):** stream the MXFP8 source one layer at a time, collect per-expert calibration
  statistics, splice variant experts into a teacher-forced forward, and score them (KL, top-1, NLL) with paired
  bootstrap comparisons.
- **TensorFold port checks:** build a TensorFold Inkling 2 model directory from the source and an EXL3 pack, compare
  the engine's per-layer hidden states, expert outputs and drafter proposals with the MLX reference, and measure
  served decode rates.
- **Inference-Mac jobs:** shell jobs that run on the inference Mac, most of them inside a maintenance window that
  stops the serving router while a full model is loaded.

## Setup

Python 3.12, managed by [uv](https://docs.astral.sh/uv/):

    uv sync                       # MLX, mlx-lm (pinned commit), numpy, safetensors
    uv sync --group ref           # adds torch and transformers for the PyTorch reference checks
    uv run pytest -q

Tests that need real weights or data are marked `real` and skip when the files are absent.

`vendor/ds4-dfm-rs` is a git submodule (`git submodule update --init`).

### Paths

`ink2q.config.Paths` reads:

| Variable | Meaning | Default |
|---|---|---|
| `INK2Q_ROOT` | Bulk data root: source checkpoint, models, reports | `work/` in this repository |
| `INK2Q_SOURCE` | The MXFP8 source checkpoint | `$INK2Q_ROOT/source` |
| `INK2Q_DATA` | Token sets and derived data | `$INK2Q_ROOT/data` |

`work/`, `data/` and `reports/` are ignored by git.

### Inference-Mac jobs

The scripts in `scripts/studio/` read their machine settings from `scripts/studio/local.env`, which is ignored by
git. Copy the example and fill it in:

    cp scripts/studio/local.env.example scripts/studio/local.env

| Variable | Meaning |
|---|---|
| `STUDIO_HOST` | ssh host of the inference Mac |
| `STUDIO_REPO` | this repository's checkout on the inference Mac |
| `INK2Q_WORK` | fast working directory on the inference Mac (model directory, engine build, gate data) |
| `INK2Q_ROOT` | bulk data root on the inference Mac |
| `STUDIO_HOME` | the [studio-inference](https://github.com/trozzelle/mac-studio-inference) workspace (default `$HOME/studio-inference`) |
| `UV` | the uv binary (default: `uv` on `PATH`) |
| `OMLX_BIN` | bin directory of an oMLX install, for the oMLX baseline jobs (only those jobs need it) |
| `STUDIO_ALLOWED_HOSTS` | space-separated host names the router admits, for the router reproduction job (only that job needs it) |

`TF_FORK` (`$INK2Q_WORK/tf-fork`) and `TF_PACK` (`$INK2Q_WORK/tf-pack`) are derived from `INK2Q_WORK`.
`scripts/studio/env.sh` loads the file and stops with a message naming any setting a script needs but is missing.

- `scripts/studio/sync.sh` copies this repository (including `local.env`) to `$STUDIO_REPO` and runs the tests there.
- `scripts/studio/tf_sync.sh FORK [TARGETS]` copies a TensorFold checkout to `$TF_FORK` and builds the given
  programs with the Zig bundled with the installed TensorFold runtime. It reads the settings on the inference Mac, so
  run `sync.sh` first.
- `scripts/studio/window.sh OUTDIR none -- CMD...` runs `CMD` in a maintenance window: it stops the router's engines,
  freezes the router, runs the job, and restores the router afterwards (`restore: done` in its log). Jobs that load a
  full model (about 196 GiB) run only inside a window.

## Layout

| Path | Contents |
|---|---|
| `ink2q/source.py`, `model.py`, `stream.py`, `teacher.py` | Per-tensor source access and the layer-streamed mlx-lm forward |
| `ink2q/collect.py`, `experts.py`, `harness.py`, `assemble.py` | Calibration statistics, variant experts and the multi-stream splice harness |
| `ink2q/score.py`, `evaluate.py`, `compare.py`, `stats.py` | Teacher-forced scoring and paired bootstrap comparisons |
| `ink2q/exl3.py` | EXL3 trellis decoding |
| `ink2q/tfpack.py` | A TensorFold model directory from the source and an EXL3 pack |
| `ink2q/tflayers.py`, `tfexperts.py`, `tftaps.py`, `tfio.py` | Per-layer, per-expert and drafter-tap references for the engine's outputs |
| `ink2q/tfdspark.py`, `tfaccept.py`, `tfdepth.py` | Drafter agreement, draft acceptance and draft-depth simulation |
| `ink2q/tfchat.py`, `tfprompts.py`, `tfgate.py` | Gate prompts and the served decode-rate client |
| `scripts/` | One-off measurement and diagnostic scripts |
| `scripts/studio/` | Inference-Mac jobs (see above) |
| `tests/` | pytest suite |
