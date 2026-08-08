import math
import time
import torch
import torch.nn as nn
import transformers
from utils.abmp import allocate_block_orders
from utils.structure_arb import structural_gaussian_distribution_multip_alternating_group_x
import logging
logger = logging.getLogger()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

'''
BRAGPTQ is the meaning of GPTQ used Binary Residual Approximation in paper to realize 1-bit quantization
BRAGPTQ uses structural mask to distinguish outliers and other data, and takes advantage of part of GPTQ to lower error
'''
class BRAGPTQ:
    def __init__(
        self, layer, braq_quantizer,salient_metric, disable_gptq=False, method='arb', order2_group=False, gptaq=False
    ):
        self.method = method
        self.order2_group = order2_group
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.braq_quantizer = braq_quantizer
        self.salient_metric = salient_metric  # "magnitude" or "hessian"
        self.disable_gptq = disable_gptq
        self.gptaq = gptaq
        self.inp = []
        
        # SliM 相关属性
        self.block_salience = []
        
        if self.gptaq:
            self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
            self.fp_inp = []

    def add_batch(self, inp, _out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            
        if self.method == 'arb-x':
            self.inp.append(inp)
            
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        if self.gptaq:
            self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        if self.gptaq:
            dX = self.fp_inp[0].float() * math.sqrt(2 / self.nsamples) - inp
            self.dXXT += dX.matmul(inp.t()) 
            del self.fp_inp[0]

    def get_salience(self, blocksize=128):
        """
        基于 H^{-1} 的对角和权重能量计算每个分块的重要性（salience）
        仅在 slim 模式下使用
        """
        # Large MLP projections (e.g. 12288 cols) need multiple Cholesky
        # workspaces; keep them on CPU when the GPU is shared.
        # Always avoid 12288-d Cholesky — it SIGKILLs on this host.
        use_diag = self.columns > 8192
        h = self.H.detach().to(self.dev if not use_diag else "cpu").clone()
        w = self.layer.weight.data.detach().to(h.device).clone()
        
        if isinstance(self.layer, nn.Conv2d):
            w = w.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            w = w.t()
        
        dead = torch.diag(h) == 0
        h[dead, dead] = 1
        
        diag = torch.arange(self.columns, device=h.device)
        damp = 0.01 * torch.mean(torch.diag(h))
        h[diag, diag] += damp
        if use_diag:
            Hinv_diag = 1.0 / torch.clamp(torch.diag(h), min=1e-8)
            for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
                col_ed = min(col_st + blocksize, self.columns)
                block_value = w[:, col_st:col_ed] ** 2 * (Hinv_diag[col_st:col_ed].reshape((1, -1)) ** 2)
                self.block_salience.append(torch.sum(block_value).item())
            del h, w
            return

        h = torch.linalg.cholesky(h)
        h = torch.cholesky_inverse(h)
        h = torch.linalg.cholesky(h, upper=True)
        Hinv = h
        
        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            st = col_st
            ed = col_ed
            block_value = w[:, st:ed] ** 2 / (torch.diag(Hinv[st:ed, st:ed]).reshape((1, -1))) ** 2
            self.block_salience.append(torch.sum(block_value).item())
        del h, Hinv, w
        torch.cuda.empty_cache()

    def _determine_block_orders(
        self,
        blocksize=128,
        no_mask_order=2,
        saved_block_orders=None,
        abmp_ratio=0.05,
    ):
        """Allocate ABMP orders from per-block Hessian salience."""
        if saved_block_orders is not None:
            return saved_block_orders

        num_blocks = (self.columns + blocksize - 1) // blocksize
        if not self.block_salience:
            return [no_mask_order] * num_blocks
        if len(self.block_salience) != num_blocks:
            raise ValueError(
                f"Expected {num_blocks} block salience values, "
                f"got {len(self.block_salience)}"
            )
        return allocate_block_orders(
            self.block_salience,
            base_order=no_mask_order,
            ratio=abmp_ratio,
        )

    def fasterquant(self,
                    blocksize=128, 
                    percdamp=0.01,
                    orders=(1,1,2),
                    num_p=1,
                    disable_mask=False,
                    no_mask_order=2,
                    alpha=0.25,
                    slim=False,
                    saved_block_orders=None,
                    abmp_ratio=0.05,
                    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        if self.gptaq:
            self.dXXT[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)
        W = W.to(self.dev)

        # 12288-d Cholesky (CPU/GPU) SIGKILLs on this host. Full 4096-d GPU
        # Cholesky also intermittently dies mid-run under shared-node pressure.
        # RTN (disable_gptq) only needs Hinv's diagonal — skip Cholesky entirely.
        # Note: self.nsamples counts batches, not tokens — not a rank proxy.
        use_diag_hinv = self.columns > 8192 or self.disable_gptq
        free_bytes, _total_bytes = torch.cuda.mem_get_info(self.dev)
        h_bytes = self.columns * self.columns * 4 * 4  # rough workspace estimate
        use_gpu_chol = (not use_diag_hinv) and free_bytes > h_bytes + (2 << 30)
        chol_device = self.dev if use_gpu_chol else "cpu"
        damp = (percdamp * torch.mean(torch.diag(H.detach()))).item()
        if use_diag_hinv:
            logger.info("Using diagonal Hinv (cols=%s, rtn=%s)", self.columns, self.disable_gptq)
            H = H.to(self.dev)
            diag = torch.arange(self.columns, device=self.dev)
            # Strong ridge: diagonal-only GPTQ is unstable with tiny Hessians.
            H[diag, diag] += max(damp, 1e-2) + 1e-2 * torch.mean(torch.diag(H))
            Hinv_diag_full = 1.0 / torch.clamp(torch.diag(H), min=1e-3)
            Hinv_diag_full = torch.clamp(Hinv_diag_full, max=1e2)
            # Keep only the diagonal vector until block loop needs a matrix slice;
            # materialize a thin diag matrix to preserve existing indexing.
            Hinv = torch.diag(Hinv_diag_full)
            del H
        else:
            logger.info(
                "Hessian Cholesky on %s (cols=%s, free_gb=%.1f)",
                chol_device,
                self.columns,
                free_bytes / 1e9,
            )
            H = H.to(chol_device)
            diag = torch.arange(self.columns, device=H.device)
            H[diag, diag] += damp
            # Extra ridge for stability with few calibration batches.
            H[diag, diag] += 1e-5 * torch.mean(torch.diag(H))
            try:
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H
                logger.info("Cholesky done (cols=%s)", self.columns)
            except Exception as exc:
                logger.warning("Cholesky failed (%s); falling back to diagonal Hinv", exc)
                Hinv = torch.diag(1.0 / torch.clamp(torch.diag(H), min=1e-8)).to(self.dev)
                use_diag_hinv = True
            del H
        torch.cuda.empty_cache()
        
        if self.gptaq:
            P = alpha * ((self.dXXT @ Hinv.to(self.dev).T).triu_(diagonal=1)) @ Hinv.to(self.dev)
            del self.dXXT

        if self.method == 'arb-x':
            self.inp = torch.concat(self.inp)
        # print(self.inp.shape)
        
        if slim:
            block_orders = self._determine_block_orders(
                blocksize,
                no_mask_order,
                saved_block_orders,
                abmp_ratio,
            )
            logger.info(f'SliM Block no_mask_orders: {block_orders}')

        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st

            st = col_st
            ed = col_ed
            
            # SliM 分支：获取当前block的order
            if slim:
                current_block_order = block_orders[blocki] if blocki < len(block_orders) else no_mask_order
            else:
                current_block_order = no_mask_order

            if self.method == 'arb-x':
                # S = torch.einsum('bki,bkj->ij', self.inp[:, :, st:ed], self.inp[:, :, st:ed])
                S = torch.matmul(self.inp[:, :, st:ed].to(torch.float32).transpose(1, 2), self.inp[:, :, st:ed].to(torch.float32)).mean(dim=0)   # avoid overflow
            else:
                S = None
            # S = self.inp2[st:ed, st:ed]
            # print(S==S2)
            
            # mask
            if not disable_mask:
                if self.order2_group:
                    num_mask = 2 * (num_p+1)
                    orders = [2 for _ in range(num_p+1)] + [1 for _ in range(num_p+1)]
                else:
                    num_mask = 1 + num_p + 1
                    orders = [2] + [1 for _ in range(num_p+1)]
                mask = torch.zeros_like(W[:, st:ed], dtype=torch.bool).unsqueeze(0).repeat_interleave(num_mask, dim=0)
                mask_list = structural_gaussian_distribution_multip_alternating_group_x(
                    W[:, st:ed],
                    Hinv[st:ed, st:ed].to(self.dev),
                    self.salient_metric,
                    50,
                    num_p,
                    S,
                    self.method,
                    self.order2_group,
                )
                for i in range(num_mask):
                    mask[i] = mask_list[i]

            assert self.braq_quantizer.groupsize % blocksize == 0
            Hinv1 = Hinv[col_st:col_ed, col_st:col_ed].to(self.dev)
            Hinv_diag = torch.diag(Hinv1)

            if self.disable_gptq:
                # RTN
                # print("RTN")
                w = W[:, col_st:col_ed]
                
                # mask
                if not disable_mask:
                    # from low to high group
                    q_part_groups = []
                    for i in range(mask.shape[0]):
                        q_part_groups.append(
                            self.braq_quantizer.quantize(
                                w,
                                mask[i],
                                Hinv_diag=Hinv_diag,
                                order=orders[i],
                                S=S,
                            )
                        )
                else:
                    q = self.braq_quantizer.quantize(
                        w,
                        None,
                        Hinv_diag=Hinv_diag,
                        order=current_block_order,
                        S=S,
                    )

                # mask
                if not disable_mask:
                    for j in range(mask.shape[0]):
                        q += q_part_groups[j][:] * mask[j, :]
                        
                W[:, col_st:col_ed] = q
            else:
                # shape of W1: [oc, n_cols]
                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                if self.gptaq:
                    P1 = P[col_st:col_ed, col_st:col_ed]

                # mask
                if not disable_mask:
    
                    q_part_groups = []

                    for i in range(mask.shape[0]):
                        q_part_groups.append(
                            self.braq_quantizer.quantize(
                                W1,
                                mask[i],
                                Hinv_diag=Hinv_diag,
                                order=orders[i],
                                S=S,
                            )
                        )
                    q = torch.zeros_like(W1)
                    for j in range(mask.shape[0]):
                        q += q_part_groups[j] * mask[j]

                else:
                    q = self.braq_quantizer.quantize(W1, None, Hinv_diag=Hinv_diag, order=current_block_order, S=S)

                # 并行优化后的版本
                diff = W1 - q        # [oc, n_cols]
                d_vec = Hinv_diag.view(1, -1)  # [1, n_cols]
                Q1 = q
                Losses1 = (diff ** 2) / (d_vec ** 2)
                Err1 = diff / d_vec


                W[:, col_st:col_ed] = Q1
                Losses += torch.sum(Losses1, 1) / 2

                # Diagonal Hinv has no off-block coupling; skip feedback that
                # otherwise amplifies noise when the Hessian is ill-conditioned.
                if not use_diag_hinv:
                    hinv_tail = Hinv[col_st:col_ed, col_ed:].to(self.dev)
                    if self.gptaq:
                        W[:, col_ed:] -= Err1.matmul(hinv_tail) - W1.matmul(P[col_st:col_ed, col_ed:])
                    else:
                        W[:, col_ed:] -= Err1.matmul(hinv_tail)
                    del hinv_tail
                del Hinv1

        torch.cuda.synchronize()
        times = time.time() - tick
        logger.info(f'time {times:.2f}')
        logger.info(f'error {torch.sum(Losses).item()}')

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if not disable_mask:
            del mask
            # del mask1, mask2, mask3
            del mask_list
            del q_part_groups
        if not self.disable_gptq:
            del W1, Q1, W, Err1, Losses1
        del Hinv, S
        if hasattr(self, "inp"):
            del self.inp
        torch.cuda.empty_cache()
        result = {"error": torch.sum(Losses).item()}
        if slim:
            result["block_orders"] = block_orders
        return result

    def free(self):
        self.H = None
        if self.gptaq:
            self.dXXT = None
        torch.cuda.empty_cache()
