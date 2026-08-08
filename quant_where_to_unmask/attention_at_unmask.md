# Attention при unmask: куда смотрим и кто смотрит на нас

Traces: **32**, событий: **4096**, checkpoint: `checkpoints/results_wikitext_fp16_g64_n256`, layers: **[16, 31]**

## Методология

LLaDA не поддерживает `output_attentions` — веса считаем через hook на SDPA (bidirectional MDM).

На step `s` (before) и `s+1` (after unmask pos `p`):
- **out_X** = Σ attn[p→·] на регион X (prompt / comp_masked / comp_real)
- **in_X** = Σ attn[·→p] из региона X
- **d_*** = after − before; control = still-masked позиция на том же step

---

## Layer 16

### 1. Outgoing (куда смотрит unmask-позиция)

| mass | before | after | Δ |
|------|--------|-------|---|
| prompt | 0.398 | 0.400 | +0.003 |
| comp_masked | 0.241 | 0.201 | -0.040 |
| comp_real | 0.362 | 0.398 | +0.037 |
| self | 0.041 | 0.055 | +0.014 |

### 2. Incoming (кто смотрит на unmask-позицию)

| mass | before | after | Δ |
|------|--------|-------|---|
| prompt | 0.156 | 0.246 | +0.090 |
| comp_masked | 0.273 | 0.384 | +0.112 |
| comp_real | 0.236 | 0.350 | +0.115 |
| self | 0.041 | 0.055 | +0.014 |

### 3. vs control (still masked)

- mean Δ out→comp_masked (unmask): **-0.040**
- mean Δ out→comp_masked (control): **-0.013**
- mean Δ in←comp_real (unmask): **+0.115**
- mean Δ in←comp_real (control): **+0.031**

- mean out→sibling masks: before **0.001**, after **0.002**

---

## Layer 31

### 1. Outgoing (куда смотрит unmask-позиция)

| mass | before | after | Δ |
|------|--------|-------|---|
| prompt | 0.380 | 0.366 | -0.015 |
| comp_masked | 0.257 | 0.165 | -0.092 |
| comp_real | 0.363 | 0.470 | +0.107 |
| self | 0.095 | 0.117 | +0.022 |

### 2. Incoming (кто смотрит на unmask-позицию)

| mass | before | after | Δ |
|------|--------|-------|---|
| prompt | 0.174 | 0.263 | +0.090 |
| comp_masked | 0.360 | 0.396 | +0.036 |
| comp_real | 0.234 | 0.405 | +0.170 |
| self | 0.095 | 0.117 | +0.022 |

### 3. vs control (still masked)

- mean Δ out→comp_masked (unmask): **-0.092**
- mean Δ out→comp_masked (control): **-0.022**
- mean Δ in←comp_real (unmask): **+0.170**
- mean Δ in←comp_real (control): **+0.034**

- mean out→sibling masks: before **0.003**, after **0.002**

---

## Выводы

- После unmask позиция **меньше** смотрит на другие маски.
- После unmask **больше** attention от уже открытых (real) completion токенов.
