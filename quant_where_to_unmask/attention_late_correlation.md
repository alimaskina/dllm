# Attention позже unmask vs история открытия

Traces: **32**, событий: **2048** × layers, checkpoint: `checkpoints/results_wikitext_fp16_g64_n256`, layers: **[16, 31]**

## Методология

Для каждого unmask на step `s`, pos `p` фиксируем **историю открытия**: confidence, step, flip предсказания пока masked, steps до конца блока.

Attention **потом**:
- `final_in_*` — incoming на `p` на **финальном** forward (все 64 tok real)
- `post_mean_in_*` — средний incoming на `p` на steps `s+1..63`
- `post_mean_attn_from_later_unmaskers` — attn от query-позиций, которые unmask'ятся **позже** `p`, на key `p`

Корреляции: Pearson / Spearman.

---

## Layer 16

### Корреляции

| meta | metric | pearson | spearman |
|------|--------|---------|----------|
| confidence | final_in_completion | -0.005 | -0.153 |
| confidence | final_in_comp_real | -0.005 | -0.153 |
| confidence | final_in_prompt | -0.000 | -0.194 |
| confidence | post_mean_in_completion | -0.030 | -0.159 |
| confidence | post_mean_in_comp_real | -0.018 | -0.096 |
| confidence | post_mean_attn_from_later_unmaskers | -0.049 | -0.133 |
| confidence | after_in_comp_real | -0.011 | +0.021 |
| step unmask | final_in_completion | -0.017 | +0.250 |
| step unmask | final_in_comp_real | -0.017 | +0.250 |
| step unmask | final_in_prompt | -0.071 | -0.352 |
| step unmask | post_mean_in_completion | -0.040 | +0.164 |
| step unmask | post_mean_in_comp_real | +0.017 | +0.350 |
| step unmask | post_mean_attn_from_later_unmaskers | +0.074 | +0.411 |
| step unmask | after_in_comp_real | +0.265 | +0.621 |
| steps until block end | final_in_completion | +0.017 | -0.250 |
| steps until block end | final_in_comp_real | +0.017 | -0.250 |
| steps until block end | final_in_prompt | +0.071 | +0.352 |
| steps until block end | post_mean_in_completion | +0.040 | -0.164 |
| steps until block end | post_mean_in_comp_real | -0.017 | -0.350 |
| steps until block end | post_mean_attn_from_later_unmaskers | -0.074 | -0.411 |
| steps until block end | after_in_comp_real | -0.265 | -0.621 |
| prediction flips while masked | final_in_completion | -0.015 | +0.194 |
| prediction flips while masked | final_in_comp_real | -0.015 | +0.194 |
| prediction flips while masked | final_in_prompt | -0.051 | -0.202 |
| prediction flips while masked | post_mean_in_completion | -0.007 | +0.179 |
| prediction flips while masked | post_mean_in_comp_real | +0.021 | +0.272 |
| prediction flips while masked | post_mean_attn_from_later_unmaskers | +0.054 | +0.225 |
| prediction flips while masked | after_in_comp_real | +0.195 | +0.430 |
| mask_ratio at unmask | final_in_completion | +0.017 | -0.250 |
| mask_ratio at unmask | final_in_comp_real | +0.017 | -0.250 |
| mask_ratio at unmask | final_in_prompt | +0.071 | +0.352 |
| mask_ratio at unmask | post_mean_in_completion | +0.040 | -0.164 |
| mask_ratio at unmask | post_mean_in_comp_real | -0.017 | -0.350 |
| mask_ratio at unmask | post_mean_attn_from_later_unmaskers | -0.074 | -0.411 |
| mask_ratio at unmask | after_in_comp_real | -0.265 | -0.621 |

### Бuckets

| subset | n | final_in_real | post_in_real | attn from later unmaskers |
|--------|---|---------------|--------------|---------------------------|
| conf < 0.3 | 474 | 0.831 | 0.589 | 0.016 |
| conf 0.3–0.6 | 628 | 0.677 | 0.483 | 0.012 |
| conf ≥ 0.6 | 946 | 0.709 | 0.500 | 0.012 |
| early step < 8 | 256 | 0.814 | 0.499 | 0.012 |
| mid 8–31 | 768 | 0.745 | 0.507 | 0.011 |
| late step ≥ 32 | 1024 | 0.692 | 0.527 | 0.014 |
| no flip | 186 | 0.948 | 0.512 | 0.012 |
| had flip | 1862 | 0.705 | 0.516 | 0.013 |
| flips ≥ 3 | 1294 | 0.692 | 0.534 | 0.013 |

## Layer 31

### Корреляции

| meta | metric | pearson | spearman |
|------|--------|---------|----------|
| confidence | final_in_completion | -0.013 | -0.117 |
| confidence | final_in_comp_real | -0.013 | -0.117 |
| confidence | final_in_prompt | -0.013 | -0.072 |
| confidence | post_mean_in_completion | -0.021 | -0.081 |
| confidence | post_mean_in_comp_real | -0.007 | -0.045 |
| confidence | post_mean_attn_from_later_unmaskers | -0.047 | -0.067 |
| confidence | after_in_comp_real | +0.129 | +0.179 |
| step unmask | final_in_completion | -0.053 | -0.001 |
| step unmask | final_in_comp_real | -0.053 | -0.001 |
| step unmask | final_in_prompt | -0.140 | -0.381 |
| step unmask | post_mean_in_completion | -0.113 | -0.122 |
| step unmask | post_mean_in_comp_real | +0.027 | +0.216 |
| step unmask | post_mean_attn_from_later_unmaskers | +0.180 | +0.396 |
| step unmask | after_in_comp_real | +0.477 | +0.670 |
| steps until block end | final_in_completion | +0.053 | +0.001 |
| steps until block end | final_in_comp_real | +0.053 | +0.001 |
| steps until block end | final_in_prompt | +0.140 | +0.381 |
| steps until block end | post_mean_in_completion | +0.113 | +0.122 |
| steps until block end | post_mean_in_comp_real | -0.027 | -0.216 |
| steps until block end | post_mean_attn_from_later_unmaskers | -0.180 | -0.396 |
| steps until block end | after_in_comp_real | -0.477 | -0.670 |
| prediction flips while masked | final_in_completion | -0.024 | +0.064 |
| prediction flips while masked | final_in_comp_real | -0.024 | +0.064 |
| prediction flips while masked | final_in_prompt | -0.092 | -0.302 |
| prediction flips while masked | post_mean_in_completion | -0.049 | -0.022 |
| prediction flips while masked | post_mean_in_comp_real | +0.034 | +0.185 |
| prediction flips while masked | post_mean_attn_from_later_unmaskers | +0.064 | +0.091 |
| prediction flips while masked | after_in_comp_real | +0.293 | +0.413 |
| mask_ratio at unmask | final_in_completion | +0.053 | +0.001 |
| mask_ratio at unmask | final_in_comp_real | +0.053 | +0.001 |
| mask_ratio at unmask | final_in_prompt | +0.140 | +0.381 |
| mask_ratio at unmask | post_mean_in_completion | +0.113 | +0.122 |
| mask_ratio at unmask | post_mean_in_comp_real | -0.027 | -0.216 |
| mask_ratio at unmask | post_mean_attn_from_later_unmaskers | -0.180 | -0.396 |
| mask_ratio at unmask | after_in_comp_real | -0.477 | -0.670 |

### Бuckets

| subset | n | final_in_real | post_in_real | attn from later unmaskers |
|--------|---|---------------|--------------|---------------------------|
| conf < 0.3 | 474 | 0.751 | 0.583 | 0.016 |
| conf 0.3–0.6 | 628 | 0.730 | 0.568 | 0.012 |
| conf ≥ 0.6 | 946 | 0.713 | 0.569 | 0.013 |
| early step < 8 | 256 | 0.768 | 0.540 | 0.011 |
| mid 8–31 | 768 | 0.772 | 0.575 | 0.011 |
| late step ≥ 32 | 1024 | 0.682 | 0.578 | 0.016 |
| no flip | 186 | 0.791 | 0.542 | 0.014 |
| had flip | 1862 | 0.720 | 0.575 | 0.013 |
| flips ≥ 3 | 1294 | 0.709 | 0.582 | 0.014 |

## Выводы

- **Layer 16**: conf↔final_in_real r=-0.005; step↔final_in_real ρ=+0.250.
  attn от later unmaskers: no_flip=0.012, had_flip=0.013 (Δ=+0.001).
- **Layer 31**: conf↔final_in_real r=-0.013; step↔final_in_real ρ=-0.001.
  attn от later unmaskers: no_flip=0.014, had_flip=0.013 (Δ=-0.001).
