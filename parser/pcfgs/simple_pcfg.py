"""Triton-backed simple-PCFG inside/decode (SimplePCFG_Triton, SimplePCFG_Triton_Batch).
Backs the HN/SN (single-sentence) and HC/SC (batched) models."""
from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import  stripe, diagonal_copy_, checkpoint, diagonal, stripe_add_
import torch
from parser.triton.fn import _merge, _log_then_diagonal_copy_


class SimplePCFG_Triton(PCFG_base):
    def __init__(self):
        super(SimplePCFG_Triton, self).__init__()

    def loss(self, rules, lens):
        return self._inside(rules, lens)


    @torch.enable_grad()
    def _inside(self, rules, lens, mbr=False, viterbi=False, marginal=False, s_span=None, entropy = False, span_dist=False):
        assert viterbi is not True
        # B, L, r_p
        unary = rules['unary'].clone()
        # B, L, r_m
        root = rules['root'].exp()

        # r_m, r_m
        L = rules['left_m']
        R = rules['right_m']
        # r_p, r_p
        L_p = rules['left_p']
        R_p = rules['right_p']
        LR = torch.cat([L, R], dim=-1)
        r_p = unary.shape[-1]
        r_m = L.shape[1]

        batch, N, *_ = unary.shape
        N += 1
        # for estimating marginals.
        if s_span is None:
            span_indicator = unary.new_zeros(batch, N, N).requires_grad_(mbr or span_dist)
        else:
            span_indicator = s_span
            if mbr or viterbi:
                span_indicator = span_indicator.detach().clone().requires_grad_(True)
            unary += diagonal(span_indicator, w=1).unsqueeze(-1)

        # normalizer = unary.new_zeros(batch, N, N).fill_(-1e9)

        with torch.no_grad():
            unary_max = unary.max(-1)[0]

        unary = (unary - unary_max.unsqueeze(-1)).exp()
        unary = torch.einsum('bnp, pq -> bnq',  unary ,torch.cat([L_p, R_p], dim=-1))

        alpha_c = unary.new_zeros(batch, N, N,  2, r_m)
        alpha_c = _log_then_diagonal_copy_(unary, unary_max, alpha_c)

        tocopy_array = []
        # w: span width
        for w in range(2, N):
            n = N - w
            normalizer = alpha_c.new_zeros(batch, n)
            out, normalizer = _merge(normalizer, diagonal(span_indicator, w), alpha_c)
            tocopy_array.append(out)
            if w < N-1:
                out = torch.einsum('blr, rq -> blq', out, LR)
                alpha_c = _log_then_diagonal_copy_(out, normalizer, alpha_c)

        logZ = (torch.einsum('bnr, br -> b', out, root) + 1e-9).log() + normalizer.squeeze(1)

        if span_dist:
            marginals = self._get_span_distribution(logZ, tocopy_array, span_indicator)
            return {
                "partition": logZ,
                "span_marginals": marginals,
            }

        if not mbr and not viterbi:
            return {'partition': logZ}

        elif marginal:
            logZ.sum().backward()
            return {'marginal': span_indicator.grad}

        else:
            return {
                "prediction": self._get_prediction(logZ, span_indicator, lens, mbr=True),
                "partition": logZ
            }

    def _get_span_distribution(self, logZ, tocopy_array, span_indicator):
        """Per-span unnormalised NT marginals via autograd on the inside chart.

        For each span width w >= 2, the `out` tensor produced just after
        `_merge` has shape (B, N-w, NT) and corresponds to the inside score
        contributions of each NT at that span. The gradient of `logZ` w.r.t.
        `out` (clamped to non-negative) recovers the expected NT-rule usage
        per span, i.e. the unnormalised posterior over NT symbols.
        """
        NT = tocopy_array[0].shape[-1]
        B, N = span_indicator.shape[:2]
        assert logZ.requires_grad
        grads = torch.autograd.grad(
            logZ.sum(), tocopy_array, create_graph=False, retain_graph=False
        )
        corrected = [torch.clamp(g, min=0.0) for g in grads]
        marginals = span_indicator.new_zeros(B, N, N, NT)
        for w in range(2, marginals.shape[1]):
            diagonal_copy_(marginals, corrected[w - 2], w)
        return marginals



class SimplePCFG_Triton_Batch(PCFG_base):
    def __init__(self):
        super(SimplePCFG_Triton_Batch, self).__init__()

    def loss(self, rules, lens):
        return self._inside(rules, lens)


    @torch.enable_grad()
    def _inside(self, rules, lens, mbr=False, viterbi=False, marginal=False, s_span=None, entropy = False):
        assert viterbi is not True
        # B, L, r_p
        unary = rules['unary'].clone()
        # B, L, r_m
        root = rules['root'].exp()

        # r_m, r_m 
        L = rules['left_m']
        R = rules['right_m']
        # r_p, r_p
        L_p = rules['left_p']
        R_p = rules['right_p']
        LR = torch.cat([L, R], dim=-1)
        r_p = unary.shape[-1]
        r_m = L.shape[-2]        
        batch, N, *_ = unary.shape
        N += 1
        # for estimating marginals.
        if s_span is None:
            span_indicator = unary.new_zeros(batch, N, N).requires_grad_(mbr)
        else:
            span_indicator = s_span
            if mbr or viterbi:
                span_indicator = span_indicator.detach().clone().requires_grad_(True)
            unary += diagonal(span_indicator, w=1).unsqueeze(-1)

        # normalizer = unary.new_zeros(batch, N, N).fill_(-1e9)

        with torch.no_grad():
            unary_max = unary.max(-1)[0]

        unary = (unary - unary_max.unsqueeze(-1)).exp()        

        unary = torch.einsum('bnp, bpq -> bnq',  unary ,torch.cat([L_p, R_p], dim=-1))

        alpha_c = unary.new_zeros(batch, N, N,  2, r_m)

        alpha_c = _log_then_diagonal_copy_(unary, unary_max, alpha_c)
        
        # w: span width
        for w in range(2, N):
            n = N - w      
            normalizer = alpha_c.new_zeros(batch, n)
            
            out, normalizer = _merge(normalizer, diagonal(span_indicator, w), alpha_c)

            if w < N-1:                                
                out = torch.einsum('blr, brq -> blq', out, LR)                
                alpha_c = _log_then_diagonal_copy_(out, normalizer, alpha_c)
        
        logZ = (torch.einsum('bnr, br -> b', out, root) + 1e-9).log() + normalizer.squeeze(1)

        if not mbr and not viterbi:
            return {'partition': logZ}

        elif marginal:
            logZ.sum().backward()
            
            return {'marginal': span_indicator.grad}

        else:
            return {
                
                "prediction": self._get_prediction(logZ, span_indicator, lens, mbr=True),
                "partition": logZ
            }



