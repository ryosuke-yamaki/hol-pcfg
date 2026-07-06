"""HC-PCFG -- Holographic Compound PCFG (HolE scoring + compound VAE).

ARCHIVED: previously a fork contribution dispatched via model_name='HCPCFG' in
parser/helper/util.py:get_model. The class is no longer registered in
parser/model/__init__.py or get_model; this file is kept for provenance only.
To revive it, re-add the import/export and the 'HCPCFG' get_model branch (the
module's `from parser.*` imports still resolve from the repo root)."""
import math
import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.checkpoint import checkpoint as ckp
from parser.pcfgs.simple_pcfg import SimplePCFG_Triton_Batch


class HC_PCFG(nn.Module):
    """Holographic Compound PCFG.

    Combines HolE scoring (from HN-PCFG) with the VAE framework (from SC-PCFG).
    Rules are parameterized via circular convolution instead of MLP bilinear scoring.
    Two z-injection modes:
        - 'relation': batch-dependent relation vectors
        - 'parent':   batch-dependent parent embeddings

    Two z-injection methods:
        - 'additive':       v^z = v + W(z) (original, norm-breaking)
        - 'phase_rotation': v^z via freq-domain phase rotation (norm-preserving)
    """

    def __init__(self, args, dataset):
        super(HC_PCFG, self).__init__()
        self.pcfg = SimplePCFG_Triton_Batch()

        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)
        self.s_dim = args.s_dim
        self.z_dim = args.z_dim
        self.enc_dim = args.h_dim
        self.word_emb_size = args.w_dim

        ## root
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        input_dim = self.s_dim + self.z_dim

        # terms (z-dependent, same as SC-PCFG)
        self.term_mlp = nn.Sequential(nn.Linear(input_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V)
        )

        # Entity embeddings: N(0, 1/d) following HolE (Plate 1995)
        self.rule_state_emb = nn.Parameter(
            torch.randn(self.NT + self.T, self.s_dim) / math.sqrt(self.s_dim)
        )

        # --- HolE config flags (from HN-PCFG) ---
        self.projection_mode = getattr(args, 'projection_mode', 'none')
        self.use_cnorm = getattr(args, 'use_cnorm', False)
        self.use_entity_cnorm = getattr(args, 'use_entity_cnorm', False)
        self.use_multi_tau = getattr(args, 'use_multi_tau', False)
        self.use_temperature = getattr(args, 'use_temperature', False)
        self.use_relation_projection_cnorm = getattr(args, 'use_relation_projection_cnorm', False)
        self.max_norm = getattr(args, 'max_norm', 1.0)

        # --- Temperature / tau (from HN-PCFG) ---
        if self.use_multi_tau:
            # multi_tau is exclusive with single temperature
            self.use_temperature = False
            tau_init = getattr(args, 'tau_init', 1.0)
            log_tau_init = math.log(tau_init)
            self.log_tau_root = nn.Parameter(torch.tensor(log_tau_init))
            self.log_tau_term = nn.Parameter(torch.tensor(log_tau_init))
            self.log_tau_rule = nn.Parameter(torch.tensor(log_tau_init))
        elif self.use_temperature:
            self.log_tau = nn.Parameter(torch.tensor(0.0))

        # --- HolE relation vectors (from HN-PCFG) ---
        bnd = math.sqrt(6.0 / (1 + self.s_dim))
        self.v_left = nn.Parameter(torch.empty(self.s_dim).uniform_(-bnd, bnd))
        self.v_right = nn.Parameter(torch.empty(self.s_dim).uniform_(-bnd, bnd))

        # --- z injection mode and method ---
        self.z_inject_mode = getattr(args, 'z_inject_mode', 'relation')
        self.z_inject_method = getattr(args, 'z_inject_method', 'additive')
        if self.z_inject_mode == 'relation':
            self.W_left = nn.Linear(self.z_dim, self.s_dim, bias=False)
            self.W_right = nn.Linear(self.z_dim, self.s_dim, bias=False)
        elif self.z_inject_mode == 'parent':
            self.parent_mlp = nn.Sequential(
                nn.Linear(self.s_dim + self.z_dim, self.s_dim), nn.ReLU()
            )

        # --- Encoder (same as SC-PCFG) ---
        self.enc_emb = nn.Embedding(self.V, 512)
        self.enc_rnn = nn.LSTM(512, 512, bidirectional=True, num_layers=1, batch_first=True)
        self.enc_out = nn.Linear(512 * 2, self.z_dim * 2)

        # --- KL config ---
        self._current_beta = 1.0
        self.kl_beta_max = getattr(args, 'kl_beta_max', 1.0)
        self.free_bits = getattr(args, 'free_bits', 0.0)
        self.freeze_encoder = getattr(args, 'freeze_encoder', False)
        self.use_checkpoint = getattr(args, 'use_checkpoint', False)

        if self.freeze_encoder:
            for module in [self.enc_emb, self.enc_rnn, self.enc_out]:
                for p in module.parameters():
                    p.requires_grad = False

        # --- Monitoring state ---
        self._last_mean = None
        self._last_lvar = None
        self._last_kl = None

        self._initialize()

    def _initialize(self):
        skip = {'v_left', 'v_right', 'log_tau', 'log_tau_root', 'log_tau_term', 'log_tau_rule'}
        for name, p in self.named_parameters():
            if name in skip:
                continue
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        if self.z_inject_mode == 'relation':
            if self.z_inject_method == 'phase_rotation':
                # Small-scale init: phases are random at start but W learns quickly
                nn.init.normal_(self.W_left.weight, std=0.01)
                nn.init.normal_(self.W_right.weight, std=0.01)
            else:
                # Zero init so v^z = v + 0 = v at start
                nn.init.zeros_(self.W_left.weight)
                nn.init.zeros_(self.W_right.weight)

    @torch.no_grad()
    def project_embeddings(self):
        """Project entity embeddings according to the configured mode."""
        if self.projection_mode == 'normless1':
            nrm_sq = self.rule_state_emb.data.pow(2).sum(dim=-1, keepdim=True)
            nrm_sq = nrm_sq.clamp(min=1.0)
            self.rule_state_emb.data.div_(nrm_sq)

        elif self.projection_mode == 'normless1_paper':
            nrm = self.rule_state_emb.data.norm(dim=-1, keepdim=True)
            nrm = nrm.clamp(min=1.0)
            self.rule_state_emb.data.div_(nrm)

        elif self.projection_mode == 'unit_sphere':
            nrm = self.rule_state_emb.data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.rule_state_emb.data.div_(nrm)

        elif self.projection_mode == 'max_norm':
            nrm = self.rule_state_emb.data.norm(dim=-1, keepdim=True)
            scale = (self.max_norm / nrm).clamp(max=1.0)
            self.rule_state_emb.data.mul_(scale)

        elif self.projection_mode == 'freq_cnorm':
            e_f = torch.fft.rfft(self.rule_state_emb.data, dim=-1)
            e_f = e_f / e_f.abs().clamp(min=1e-8)
            self.rule_state_emb.data = torch.fft.irfft(e_f, n=self.s_dim, dim=-1)

        # Relation projection cnorm (independent of entity projection mode)
        if self.use_relation_projection_cnorm:
            for v in (self.v_left, self.v_right):
                v_f = torch.fft.rfft(v.data)
                v_f = v_f / v_f.abs().clamp(min=1e-8)
                v.data = torch.fft.irfft(v_f, n=self.s_dim)

    def _phase_rotate(self, base: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Apply norm-preserving phase rotation in the frequency domain.

        Computes irfft(rfft(base) * unit_complex(rfft(shift))).
        The result has the same spectral magnitudes as base, with phases
        rotated by the phases of shift. Preserves ||base|| by construction.

        Args:
            base: Base signal (projection-cnormed, norm≈1).
                R-mode: (s_dim,) single relation vector
                P-mode: (NT, s_dim) entity embeddings
            shift: Phase shift signal (z-dependent).
                R-mode: (B, s_dim) from W(z)
                P-mode: (B, NT, s_dim) from parent_mlp(cat(e_A, z))

        Returns:
            Phase-rotated signal with same shape as shift. norm≈||base||.
        """
        base_f = torch.fft.rfft(base, dim=-1)
        shift_f = torch.fft.rfft(shift, dim=-1)
        unit_shift = shift_f / (shift_f.abs() + 1e-6)
        # Broadcast base over batch dimension(s)
        rotated_f = base_f.unsqueeze(0) * unit_shift
        return torch.fft.irfft(rotated_f, n=self.s_dim, dim=-1)

    def _hole_scores_batch(self, v: torch.Tensor, parent_emb: torch.Tensor,
                           child_emb: torch.Tensor, batch_mode: str) -> torch.Tensor:
        """Compute batch-dependent HolE scores.

        Args:
            v: Relation vector.
                batch_mode='v':      (B, s_dim) [R-mode]
                batch_mode='parent': (s_dim,)   [P-mode]
            parent_emb: Parent entity embeddings.
                batch_mode='v':      (NT, s_dim)
                batch_mode='parent': (B, NT, s_dim)
            child_emb: Child entity embeddings, (NT+T, s_dim), always z-independent.
            batch_mode: 'v' for R-mode, 'parent' for P-mode.

        Returns:
            scores: (B, NT+T, NT)
        """
        v_f = torch.fft.rfft(v, dim=-1)
        if self.use_cnorm:
            v_f = v_f / (v_f.abs() + 1e-12)
        parent_f = torch.fft.rfft(parent_emb, dim=-1)
        if self.use_entity_cnorm:
            parent_f = parent_f / (parent_f.abs() + 1e-12)

        if batch_mode == 'v':
            # v_f (B, freq), parent_f (NT, freq) → (B, NT, freq)
            template = torch.fft.irfft(v_f[:, None, :] * parent_f[None, :, :],
                                       n=self.s_dim, dim=-1)
        else:  # batch_mode == 'parent'
            # v_f (freq,), parent_f (B, NT, freq) → (B, NT, freq)
            template = torch.fft.irfft(v_f[None, None, :] * parent_f,
                                       n=self.s_dim, dim=-1)

        # child_emb already ecnorm'd by caller if needed
        scores = torch.einsum('md, bnd -> bmn', child_emb, template)

        if self.use_multi_tau:
            scores = scores * self.log_tau_rule.exp()
        elif self.use_temperature:
            scores = scores * self.log_tau.exp()
        return scores

    def forward(self, input, evaluating=False, **kwargs):
        x = input['word']
        b, n = x.shape[:2]
        seq_len = input['seq_len']

        # --- Encoder (same as SC-PCFG) ---
        def enc(x):
            x_embbed = self.enc_emb(x)
            x_packed = pack_padded_sequence(
                x_embbed, seq_len.cpu(), batch_first=True, enforce_sorted=False
            )
            h_packed, _ = self.enc_rnn(x_packed)
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
            out = self.enc_out(h)
            mean = out[:, : self.z_dim]
            lvar = out[:, self.z_dim :]
            return mean, lvar

        def kl(mean, logvar):
            result = -0.5 * (logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1)
            return result

        mean, lvar = enc(x)
        z = mean

        if self.freeze_encoder:
            z = torch.zeros_like(mean)
        elif not evaluating:
            z = mean.new(b, mean.size(1)).normal_(0, 1)
            z = (0.5 * lvar).exp() * z + mean

        # Save for monitoring (posterior collapse detection)
        self._last_mean = mean.detach()
        self._last_lvar = lvar.detach()
        self._last_kl = kl(mean, lvar).sum(1).detach()

        # --- Roots (z-independent, from HN-PCFG) ---
        def roots():
            logits = self.root_emb @ self.rule_state_emb[:self.NT].t()
            if self.use_multi_tau:
                logits = logits * self.log_tau_root.exp()
            roots = logits.log_softmax(-1)
            return roots.expand(b, roots.shape[-1])

        # --- Terms (z-dependent, same as SC-PCFG + multi_tau) ---
        def terms():
            term_emb = self.rule_state_emb[self.NT:].unsqueeze(0).expand(
                b, self.T, self.s_dim
            )
            z_expand = z.unsqueeze(1).expand(b, self.T, self.z_dim)
            term_emb = torch.cat([term_emb, z_expand], -1)
            logits = self.term_mlp(term_emb)
            if self.use_multi_tau:
                logits = logits * self.log_tau_term.exp()
            term_prob = logits.log_softmax(-1)
            return term_prob.gather(-1, x.unsqueeze(1).expand(b, self.T, x.shape[-1])).transpose(-1, -2)

        # --- Rules (z-dependent via HolE + z injection) ---
        def rules(z):
            nonterm_emb = self.rule_state_emb[:self.NT]   # (NT, s_dim)
            child_emb = self.rule_state_emb               # (NT+T, s_dim)

            # Pre-compute child ecnorm (shared by left/right, avoids redundant FFT)
            if self.use_entity_cnorm:
                child_f = torch.fft.rfft(child_emb, dim=-1)
                child_f = child_f / (child_f.abs() + 1e-12)
                child_emb = torch.fft.irfft(child_f, n=self.s_dim, dim=-1)

            if self.z_inject_mode == 'relation':
                if self.z_inject_method == 'phase_rotation':
                    v_left_z = self._phase_rotate(self.v_left, self.W_left(z))
                    v_right_z = self._phase_rotate(self.v_right, self.W_right(z))
                else:
                    v_left_z = self.v_left.unsqueeze(0) + self.W_left(z)
                    v_right_z = self.v_right.unsqueeze(0) + self.W_right(z)
                left = self._hole_scores_batch(v_left_z, nonterm_emb, child_emb, batch_mode='v')
                right = self._hole_scores_batch(v_right_z, nonterm_emb, child_emb, batch_mode='v')
            elif self.z_inject_mode == 'parent':
                nonterm_expand = nonterm_emb.unsqueeze(0).expand(b, self.NT, self.s_dim)
                z_expand = z.unsqueeze(1).expand(b, self.NT, self.z_dim)
                if self.z_inject_method == 'phase_rotation':
                    shift = self.parent_mlp(torch.cat([nonterm_expand, z_expand], -1))
                    parent_z = self._phase_rotate(nonterm_emb, shift)
                else:
                    parent_z = self.parent_mlp(torch.cat([nonterm_expand, z_expand], -1)) + nonterm_expand
                left = self._hole_scores_batch(self.v_left, parent_z, child_emb, batch_mode='parent')
                right = self._hole_scores_batch(self.v_right, parent_z, child_emb, batch_mode='parent')
            else:
                raise NotImplementedError(f"Unknown z_inject_mode: {self.z_inject_mode}")

            left = left.softmax(-2)
            right = right.softmax(-2)

            left_m = left[:, :self.NT, :].contiguous()
            left_p = left[:, self.NT:, :].contiguous()
            right_m = right[:, :self.NT, :].contiguous()
            right_p = right[:, self.NT:, :].contiguous()

            return (left_m, left_p, right_m, right_p)

        root, unary = roots(), terms()
        if self.use_checkpoint and self.training:
            (left_m, left_p, right_m, right_p) = ckp(rules, z, use_reentrant=False)
        else:
            (left_m, left_p, right_m, right_p) = rules(z)

        kl_per_dim = kl(mean, lvar)  # (B, z_dim)
        if self.free_bits > 0:
            kl_per_dim = kl_per_dim.clamp(min=self.free_bits)
        kl_total = kl_per_dim.sum(1)  # (B,)

        return {'unary': unary,
                'root': root,
                'left_m': left_m,
                'right_m': right_m,
                'left_p': left_p,
                'right_p': right_p,
                'kl': kl_total}

    @torch.no_grad()
    def get_monitoring_metrics(self) -> dict:
        """Return embedding/relation norm statistics for W&B monitoring."""
        metrics = {}
        emb = self.rule_state_emb.data
        metrics['monitor/emb_norm_mean'] = emb.norm(dim=-1).mean().item()
        metrics['monitor/emb_norm_std'] = emb.norm(dim=-1).std().item()
        metrics['monitor/v_left_norm'] = self.v_left.data.norm().item()
        metrics['monitor/v_right_norm'] = self.v_right.data.norm().item()
        if self.use_cnorm:
            v_f = torch.fft.rfft(self.v_left.data)
            metrics['monitor/v_f_magnitude_mean'] = v_f.abs().mean().item()
        if self.projection_mode == 'freq_cnorm' or self.use_entity_cnorm:
            emb_f = torch.fft.rfft(emb, dim=-1)
            emb_f_mag = emb_f.abs()
            metrics['monitor/emb_spectral_cv'] = (
                emb_f_mag.std(dim=-1).mean() / emb_f_mag.mean(dim=-1).mean()
            ).item()
        if self.use_multi_tau:
            metrics['monitor/tau_root'] = self.log_tau_root.data.exp().item()
            metrics['monitor/tau_term'] = self.log_tau_term.data.exp().item()
            metrics['monitor/tau_rule'] = self.log_tau_rule.data.exp().item()
        elif self.use_temperature:
            metrics['monitor/tau'] = self.log_tau.data.exp().item()
        # VAE monitoring
        metrics['monitor/kl_beta'] = min(self._current_beta, self.kl_beta_max)
        if self.z_inject_mode == 'relation':
            metrics['monitor/W_left_norm'] = self.W_left.weight.data.norm().item()
            metrics['monitor/W_right_norm'] = self.W_right.weight.data.norm().item()
        # Norm-preservation verification for phase rotation
        if self._last_mean is not None and self.z_inject_mode == 'relation':
            z_sample = self._last_mean  # (B, z_dim)
            if self.z_inject_method == 'phase_rotation':
                v_z = self._phase_rotate(self.v_left, self.W_left(z_sample))
            else:
                v_z = self.v_left.unsqueeze(0) + self.W_left(z_sample)
            metrics['monitor/v_left_z_norm_mean'] = v_z.norm(dim=-1).mean().item()
        # Posterior collapse detection
        if self._last_mean is not None:
            metrics['monitor/enc_mean_norm'] = self._last_mean.norm(dim=-1).mean().item()
            metrics['monitor/enc_lvar_norm'] = self._last_lvar.norm(dim=-1).mean().item()
            metrics['monitor/enc_kl'] = self._last_kl.mean().item()
        return metrics

    def loss(self, input):
        rules = self.forward(input)
        result = self.pcfg._inside(rules=rules, lens=input['seq_len'])
        beta = min(self._current_beta, self.kl_beta_max)
        loss = (-result['partition'] + beta * rules['kl']).mean()
        return loss

    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input, evaluating=True)
        if decode_type == 'viterbi':
            raise NotImplementedError
        elif decode_type == 'mbr':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'],
                                    viterbi=False, mbr=True)
        else:
            raise NotImplementedError
