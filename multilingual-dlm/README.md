# multilingual-dlm

Minimal shared utilities for parallel LLaDA / Dream experiments.

## Quick start (one command)

```bash
bash run.sh
```

First run creates `.venv`, installs CUDA torch + dependencies, downloads models/datasets from Hugging Face, then runs `smoke_test.py`.

Manual setup (optional):

```bash
bash setup.sh
source .venv/bin/activate
python smoke_test.py
```

Requirements: Linux, NVIDIA GPU (~16 GB+ VRAM), network access to Hugging Face.

## Setup

```bash
bash setup.sh
```

Installs into `.venv`:
- `torch` (CUDA 12.4 wheel by default; override with `TORCH_INDEX=...`)
- `transformers`, `accelerate`, `safetensors`, `datasets`, etc.

## API

### `load_model(preset, device=None, dtype=None)`

Load LLaDA or Dream with compatibility patches.

```python
from common import load_model, model_logits

bundle = load_model("llada")   # or "dream"
# bundle.model, bundle.tokenizer, bundle.mask_token_id, bundle.device, bundle.dtype, bundle.ar_shift
```

### `load_flores(langs=..., split="validation", max_examples=None)`

Main shared parallel corpus (FLORES-200). Each row has `example_id` (int) and `translations`.

### `load_flores_plus(langs=..., split="dev", max_examples=None)`

FLORES+ for extra parallel eval. Uses official Hub dataset when available, else consolidated parquet fallback. Same row shape as `load_flores`.

### `load_belebele(langs=..., split="test", max_examples=None)`

Belebele MCQ eval. `example_id` = `"<link>#<question_number>"`; each lang has `passage`, `question`, `choices`, `correct_answer_num`.

### `load_massive(langs=..., split="test", max_examples=None)`

MASSIVE intent utterances. `example_id` is the dataset `id`; rows include `label`, `label_text`, `utterances`.

```python
from data import load_flores, load_flores_plus, load_belebele, load_massive

rows = load_flores()          # main shared corpus
plus = load_flores_plus()     # optional parallel eval
mcq = load_belebele(max_examples=10)
intent = load_massive(max_examples=10)
```

### `get_word_units(text, tokenizer, lang="en")`

Word → subtoken mapping (per-word tokenization, not whole-sentence BPE):

```python
from data import get_word_units, build_input_ids, debug_word_units

units = get_word_units(text, bundle.tokenizer, lang="en")
input_ids = build_input_ids(units)
debug_word_units(text, bundle.tokenizer, lang="en", n=5)
```

Each unit: `word`, `char_span`, `token_positions`, `token_ids`, `k`.

### `run_oracle_trajectory(bundle, input_ids, policy="confidence", steps=32)`

Oracle denoising trajectory. Returns one `TokenReveal` per token position:

- `reveal_step` — denoising step index (same for tokens revealed together)
- `reveal_confidence` — softmax prob of argmax at reveal (confidence policy only)

```python
from trajectories import run_oracle_trajectory

traj = run_oracle_trajectory(bundle, input_ids, policy="confidence", steps=32)
traj_rand = run_oracle_trajectory(bundle, input_ids, policy="random", steps=32)
```

## Layout

```
setup.sh         — create venv + install deps
run.sh           — one-command smoke test
common.py        — model loading
data.py          — FLORES + word units
trajectories.py  — oracle trajectories
smoke_test.py    — quick sanity check
experiments/     — per-researcher entry points (templates)
```
