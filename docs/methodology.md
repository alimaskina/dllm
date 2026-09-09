# BitSieve: methodology and evaluation

## Methodology

BitSieve combines blockwise sparse attention with a packed low-bit KV cache for Fast-dLLM v2. It selects relevant prefix entries once per output block and reuses them during denoising, following MAGE's selection-and-reuse approach [1]. Persistent cache compression follows KIVI-style asymmetric quantization [3].

### Blockwise generation

Generation proceeds in blocks of $M=32$ positions. The prompt and previously finalized blocks form a prefix of length $N$ with reusable keys and values. The current block is iteratively denoised, retaining Fast-dLLM's token shift and hierarchical block cache. After denoising, a separate commit forward adds the finalized block's KV states to the prefix.

At the first denoising step, attention uses the full prefix and current block. This forward produces the first-step logits and supplies queries for selection. Subsequent steps use only to the selected prefix entries and the current block. Selection is recomputed for each new block.

### Packed KV storage

BitSieve stores keys and values as packed 2-bit or 4-bit integer codes in byte tensors. Queries and current-block KV remain floating point. For a quantization group with minimum $z$, maximum $x_{\max}$, and bit width $p$:

$$
s = \max\left(\frac{x_{\max}-z}{2^p-1},\varepsilon\right),
\qquad
q = \mathrm{clip}\left(\mathrm{round}\left(\frac{x-z}{s}\right),0,2^p-1\right),
\qquad
\hat{x}=sq+z.
$$

Keys are quantized across groups of 32 tokens independently for each channel. Values are quantized across groups of 32 channels independently for each token. Scale and offset parameters use FP16. The recent residual remains floating point, with a target length of 32 tokens; older complete groups are packed as blocks are committed. Keys and values share a quantized-prefix boundary in this implementation, rather than using the original KIVI algorithm's distinct residual-update rules.

### Prefix selection

Selection is performed independently for each request, layer, and KV head. Let $D$ be the head dimension and $G$ the number of query heads sharing a KV head. BitSieve uses up to five uniformly spaced available masked positions, denoted by $\mathcal{Q}$. For a fully masked 32-position block, these are $[0,8,16,23,31]$.

Each representative query is normalized over the prefix only:

$$
a_{g,t,j} =
\frac{\exp\left(Q_{g,t}^{\top}\hat{K}_j/\sqrt{D}\right)}
{\sum_{u=1}^{N}\exp\left(Q_{g,t}^{\top}\hat{K}_u/\sqrt{D}\right)}.
$$

The selector averages over representative positions and grouped query heads, then keeps the highest-scoring $k=512$ prefix entries:

$$
\alpha_j = \frac{1}{G|\mathcal{Q}|}
\sum_{g=1}^{G}\sum_{t\in\mathcal{Q}} a_{g,t,j},
\qquad
\mathcal{S} = \mathrm{TopK}_{j\in\{1,\ldots,N\}}(\alpha_j,k).
$$

The selected entries are gathered and dequantized into compact BF16 buffers, shared across the remaining denoising steps. Unselected entries remain in the persistent packed cache for future blocks. The first two layers retain full-prefix attention, and selection is bypassed when the prefix does not exceed the budget.

## Implementation

Selection and gathering run on a separate CUDA stream, overlapping with the first-step model computation. Later attention operates on the compact sequence.

Triton kernels fuse tile-level dequantization with floating-point attention. The dense packed path reuses KV tiles across grouped query heads and partitions long prefixes with split-K and online-softmax reduction. Compact attention uses SDPA.

The model adapter delegates prefill to the original attention implementation and uses the BitSieve cache during decoding.

## Evaluation

### Compared methods

| Method | Persistent KV | Selector |
|---|---|---|
| Official dense BF16 | Full precision | None |
| KIVI-4 | K4/V4, residual 32 | None |
| KIVI-2 | K2/V2, residual 32 | None |
| MAGE BF16, k512 | Full precision | All available masked positions |
| HERALD-center BF16, k512 | Full precision | Masked position nearest the block center |
| BitSieve K4/V4, k512 | K4/V4, residual 32 | Uniform-5 |
| BitSieve K2/V2, k512 | K2/V2, residual 32 | Uniform-5 |

The MAGE baseline is a local implementation of first-step selection and reuse. The HERALD-center baseline is a GPU selector proxy inspired by HERALD [2], without CPU offloading or draft lookahead.

### Quality

Quality evaluation uses batch size one, temperature zero, 8-token small blocks, and adaptive unmasking at confidence threshold $0.95$. The output budget is 512 tokens, or 1,024 for MATH-500.

| Tasks | Implemented metric |
|---|---|
| GSM8K, MATH-500 | Extracted-answer normalized exact match |
| HotpotQA, Qasper, TriviaQA, 2WikiMQA, MuSiQue | Best-reference normalized token F1 |
| NarrativeQA, QMSum | Best-reference ROUGE-L |
| LCC, RepoBench-P | Best-reference `SequenceMatcher` similarity |

### Performance

Controlled systems experiments use synthetic prompts of 512, 2,048, 8,192, 16,384, and 28,672 tokens across request batch sizes. Each request generates 128 tokens with 20 denoising forwards per 32-token block and no early stopping. Reported results use medians.

For request batch size $b$, output length $L$, decode time $t_{\mathrm{decode}}$ in seconds, and $J$ output blocks:

$$
\mathrm{Throughput}=\frac{bL}{t_{\mathrm{decode}}},
\qquad
\mathrm{TPOB}=\frac{1}{J}\sum_{j=1}^{J}t_j,
$$

Here $t_j$ is the block-round latency across the batch, including selection, attention, and commit work. TPOB is reported in milliseconds and is not divided by batch size. Prefill and packing are timed separately. Memory outputs record logical cache size, decode-phase CUDA allocation peaks, and externally sampled process memory.

## References

1. MAGE: All-[MASK] Block Already Knows Where to Look in Block Diffusion LLM. arXiv:2602.14209.
2. HERALD: High-Throughput Block Diffusion LLM Serving via CPU-GPU Cooperative KV Cache Retrieval. arXiv:2606.21633.
3. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. arXiv:2402.02750.
