# Dream-v0-Base-7B × WikiText: multitoken suite

Model: **Dream-org/Dream-v0-Base-7B** (AR→diffusion, Qwen-based, bidirectional attn, AR-shifted logits)
Checkpoint: `checkpoints/results_dream_wikitext_fp16_g64_n64` — n=64, g=64, steps=64, k≈1 maskgit_plus
Same prompts as LLaDA WikiText run. MASK id=151666

## 1. Unmask order (lexical multi-token)

- multi-token lexical words: **145**
- 2-tok: **127**
- consecutive positions: **100.0%**
- same-step co-unmask: **0.0%**
- 2-tok gap=1: **92.1%**
- 2-tok order LR: **48.8%**, RL: **51.2%**

- sibling pred==final at first unmask: **101/165 (61.2%)**

## 2. Other tokens → left vs right (by layer)

| layer | phase | mean left_share | prefer left% | prefer right% | n |
|------:|-------|----------------:|-------------:|--------------:|--:|
| 0 | before_first | 0.423 | 2.4 | 97.6 | 127 |
| 0 | between | 0.445 | 48.8 | 51.2 | 127 |
| 0 | all_open | 0.451 | 22.0 | 78.0 | 127 |
| 7 | before_first | 0.526 | 70.1 | 29.9 | 127 |
| 7 | between | 0.484 | 47.2 | 52.8 | 127 |
| 7 | all_open | 0.390 | 18.1 | 81.9 | 127 |
| 14 | before_first | 0.528 | 59.1 | 40.9 | 127 |
| 14 | between | 0.475 | 41.7 | 58.3 | 127 |
| 14 | all_open | 0.436 | 26.0 | 74.0 | 127 |
| 21 | before_first | 0.480 | 37.8 | 62.2 | 127 |
| 21 | between | 0.383 | 19.7 | 80.3 | 127 |
| 21 | all_open | 0.357 | 9.4 | 90.6 | 127 |
| 27 | before_first | 0.492 | 59.1 | 40.9 | 127 |
| 27 | between | 0.451 | 37.0 | 63.0 | 127 |
| 27 | all_open | 0.446 | 30.7 | 69.3 | 127 |

## 3. Other tokens → first vs second unmasked

| layer | phase | mean first_share | prefer 1st% | prefer 2nd% | n |
|------:|-------|-----------------:|------------:|------------:|--:|
| 0 | before_first | 0.504 | 50.4 | 49.6 | 127 |
| 0 | between | 0.761 | 100.0 | 0.0 | 127 |
| 0 | all_open | 0.496 | 46.5 | 53.5 | 127 |
| 7 | before_first | 0.503 | 52.0 | 48.0 | 127 |
| 7 | between | 0.598 | 81.1 | 18.9 | 127 |
| 7 | all_open | 0.510 | 56.7 | 43.3 | 127 |
| 14 | before_first | 0.490 | 42.5 | 57.5 | 127 |
| 14 | between | 0.525 | 55.1 | 44.9 | 127 |
| 14 | all_open | 0.528 | 58.3 | 41.7 | 127 |
| 21 | before_first | 0.498 | 48.0 | 52.0 | 127 |
| 21 | between | 0.552 | 59.8 | 40.2 | 127 |
| 21 | all_open | 0.511 | 52.8 | 47.2 | 127 |
| 27 | before_first | 0.489 | 47.2 | 52.8 | 127 |
| 27 | between | 0.532 | 59.8 | 40.2 | 127 |
| 27 | all_open | 0.509 | 52.0 | 48.0 | 127 |

## 4. Within-word attn at `all_open` (self vs sibling)

| layer | L self | L→R | L sib% | R self | R→L | R sib% |
|------:|-------:|----:|-------:|-------:|----:|-------:|
| 0 | 0.1712 | 0.0397 | 0.188 | 0.1519 | 0.1670 | 0.524 |
| 7 | 0.0889 | 0.0869 | 0.494 | 0.1054 | 0.0392 | 0.271 |
| 14 | 0.0485 | 0.0722 | 0.598 | 0.0632 | 0.0180 | 0.222 |
| 21 | 0.0657 | 0.0656 | 0.500 | 0.0851 | 0.0536 | 0.387 |
| 27 | 0.1282 | 0.0640 | 0.333 | 0.1266 | 0.0182 | 0.126 |

## 5. Logit lens: where is sibling readable? (`all_open`)

Mean rank of target token (0 = top-1). Lower = more readable from that position.

| layer | L reads self | R reads self | **L reads R (sib)** | **R reads L (sib)** |
|------:|-------------:|-------------:|--------------------:|--------------------:|
| 0 | 102067.2 | 91968.2 | **81586.7** | **104515.3** |
| 7 | 108104.5 | 114288.9 | **102230.5** | **102996.6** |
| 14 | 114630.8 | 123979.0 | **109270.5** | **106907.3** |
| 21 | 116311.3 | 123188.5 | **121121.7** | **110553.0** |
| 27 | 955.5 | 1910.1 | **84.6** | **1203.7** |

If AR last-token storage holds: R should read L (and word) better than L reads R at late layers.
