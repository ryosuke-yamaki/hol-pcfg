"""HN-PCFG -- HolE-based neural PCFG with independent left/right productions.
Fork contribution. Dispatched via model_name='HNPCFG' (parser/helper/util.py:get_model)."""
import math
import torch
import torch.nn as nn
from parser.pcfgs.simple_pcfg import SimplePCFG_Triton


class HN_PCFG(nn.Module):
    """HolE-based Neural PCFG with independent left and right productions.

    Scores grammar rules via circular convolution (HolE):
        score(child | parent; v) = child^T @ circonv(v, parent)

    All embeddings are projected onto the phase-only manifold (|FFT[k]| = 1)
    after every optimizer step. Only FFT phases carry learned information.
    """

    def __init__(self, args, dataset):
        super().__init__()
        self.pcfg = SimplePCFG_Triton()
        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab) if hasattr(dataset, 'word_vocab') else len(dataset.V)
        self.s_dim = args.s_dim

        self.scoring_fn = getattr(args, 'scoring_fn', 'hole')
        self.complex_normalization = getattr(args, 'complex_normalization', True)
        self.learnable_temperature = getattr(args, 'learnable_temperature', True)
        if self.scoring_fn not in ('hole', 'hadamard', 'conv'):
            raise ValueError(f"Unknown scoring_fn: {self.scoring_fn}")

        if self.complex_normalization:
            init_vec = self._uniform_phase_1d
            init_mat = self._uniform_phase
        else:
            init_vec = self._gaussian_1d
            init_mat = self._gaussian

        # Root
        self.root_emb = nn.Parameter(init_mat(1, self.s_dim))
        self._init_log_tau('log_tau_root', getattr(args, 'tau_root_init', 1.0))

        # Terms (HolE)
        self.v_term = nn.Parameter(init_vec(self.s_dim))
        self._init_log_tau('log_tau_term', getattr(args, 'tau_term_init', 1.0))
        self.vocab_emb = nn.Parameter(init_mat(self.V, self.s_dim).t())

        # Entity embeddings
        self.rule_state_emb = nn.Parameter(
            init_mat(self.NT + self.T, self.s_dim))

        # Relation vectors (HolE)
        self.v_left = nn.Parameter(init_vec(self.s_dim))
        self.v_right = nn.Parameter(init_vec(self.s_dim))
        self._init_log_tau('log_tau_rule', getattr(args, 'tau_rule_init', 1.0))

        if not self.complex_normalization:
            for p in (self.root_emb, self.vocab_emb, self.rule_state_emb):
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _uniform_phase(rows: int, s_dim: int) -> torch.Tensor:
        """Initialize on the freq_cnorm manifold with uniform random phases."""
        num_freq = s_dim // 2 + 1
        phase = torch.empty(rows, num_freq).uniform_(-math.pi, math.pi)
        return torch.fft.irfft(torch.exp(1j * phase), n=s_dim, dim=-1)

    @staticmethod
    def _uniform_phase_1d(s_dim: int) -> torch.Tensor:
        """1-D version of _uniform_phase."""
        num_freq = s_dim // 2 + 1
        phase = torch.empty(num_freq).uniform_(-math.pi, math.pi)
        return torch.fft.irfft(torch.exp(1j * phase), n=s_dim)

    @staticmethod
    def _gaussian(rows: int, s_dim: int) -> torch.Tensor:
        return torch.randn(rows, s_dim)

    @staticmethod
    def _gaussian_1d(s_dim: int) -> torch.Tensor:
        return torch.randn(s_dim)

    def _init_log_tau(self, name: str, tau_init: float) -> None:
        log_val = torch.tensor(math.log(tau_init))
        if self.learnable_temperature:
            setattr(self, name, nn.Parameter(log_val))
        else:
            self.register_buffer(name, torch.zeros_like(log_val))

    # ------------------------------------------------------------------
    # Projection (called after each optimizer step)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def project_embeddings(self):
        """Project all embeddings onto the phase-only manifold (|FFT[k]| = 1)."""
        if not self.complex_normalization:
            return
        for param in (self.rule_state_emb, self.root_emb,
                      self.v_left, self.v_right, self.v_term):
            self._complex_normalization(param)
        # vocab_emb is stored as (s_dim, V); apply per-word
        ve = self.vocab_emb.data.T
        ve_f = torch.fft.rfft(ve, dim=-1)
        ve_f = ve_f / ve_f.abs().clamp(min=1e-12)
        self.vocab_emb.data = torch.fft.irfft(ve_f, n=self.s_dim, dim=-1).T

    @staticmethod
    def _complex_normalization(param: nn.Parameter) -> None:
        """Normalize FFT magnitudes to 1 in-place (phase-only projection)."""
        p_f = torch.fft.rfft(param.data, dim=-1)
        p_f = p_f / p_f.abs().clamp(min=1e-12)
        param.data = torch.fft.irfft(p_f, n=param.data.shape[-1], dim=-1)

    # ------------------------------------------------------------------
    # HolE scoring
    # ------------------------------------------------------------------

    def _hol_scores(self, v: torch.Tensor, source_emb: torch.Tensor,
                    target_emb: torch.Tensor,
                    log_tau: torch.Tensor) -> torch.Tensor:
        """Compute scoring(v, source, target) * exp(log_tau).

        Default 'hole': target^T @ circonv(v, source) which equals
        <v, source ⋆ target> via the corr-conv identity.

        'hadamard': target^T @ (v ⊙ source) = <v, source ⊙ target>.

        'conv': target^T @ (source ⋆ v) which equals <v, source * target>
        via <v, a*b> = <b, a⋆v>.

        Args:
            v: (s_dim,) relation vector
            source_emb: (S, s_dim) source entity embeddings
            target_emb: (T, s_dim) target entity embeddings
            log_tau: scalar log-temperature (parameter or buffer)

        Returns:
            scores: (T, S) tau-scaled scores
        """
        if self.scoring_fn == 'hadamard':
            template = v.unsqueeze(0) * source_emb
            return (target_emb @ template.t()) * log_tau.exp()
        v_f = torch.fft.rfft(v)
        source_f = torch.fft.rfft(source_emb, dim=-1)
        if self.scoring_fn == 'hole':
            template_f = v_f.unsqueeze(0) * source_f
        else:  # 'conv'
            template_f = source_f.conj() * v_f.unsqueeze(0)
        template = torch.fft.irfft(template_f, n=self.s_dim, dim=-1)
        return (target_emb @ template.t()) * log_tau.exp()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input, **kwargs):
        x = input['word']
        batch_size = x.shape[0]
        nonterm_emb = self.rule_state_emb[:self.NT]
        term_emb = self.rule_state_emb[self.NT:]

        # Root
        root_logits = self.root_emb @ nonterm_emb.t()
        root = (root_logits * self.log_tau_root.exp()).log_softmax(-1)
        root = root.expand(batch_size, -1)

        # Terms
        term_logits = self._hol_scores(
            self.v_term, term_emb, self.vocab_emb.T, self.log_tau_term
        ).t()
        unary = term_logits.log_softmax(-1)[
            torch.arange(self.T)[None, None], x[:, :, None]
        ]

        # Rules
        left = self._hol_scores(
            self.v_left, nonterm_emb, self.rule_state_emb, self.log_tau_rule
        ).softmax(dim=-2)
        right = self._hol_scores(
            self.v_right, nonterm_emb, self.rule_state_emb, self.log_tau_rule
        ).softmax(dim=-2)

        return {'unary': unary,
                'root': root,
                'left_m': left[:self.NT],
                'right_m': right[:self.NT],
                'left_p': left[self.NT:],
                'right_p': right[self.NT:],
                'kl': 0}

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_monitoring_metrics(self) -> dict:
        """Return embedding/relation statistics for W&B monitoring."""
        metrics = {}

        # --- Rule state embeddings ---
        rse = self.rule_state_emb.data
        metrics['monitor/rule_state_emb_norm_mean'] = rse.norm(dim=-1).mean().item()
        metrics['monitor/rule_state_emb_norm_std'] = rse.norm(dim=-1).std().item()
        rse_f_mag = torch.fft.rfft(rse, dim=-1).abs()
        metrics['monitor/rule_state_emb_spectral_cv'] = (
            rse_f_mag.std(dim=-1).mean() / rse_f_mag.mean(dim=-1).mean()
        ).item()

        # --- Root embedding ---
        root = self.root_emb.data
        metrics['monitor/root_emb_norm'] = root.norm().item()
        root_f_mag = torch.fft.rfft(root, dim=-1).abs()
        metrics['monitor/root_emb_spectral_cv'] = (
            root_f_mag.std(dim=-1).mean() / root_f_mag.mean(dim=-1).mean()
        ).item()

        # --- Relation vectors ---
        metrics['monitor/v_left_norm'] = self.v_left.data.norm().item()
        metrics['monitor/v_right_norm'] = self.v_right.data.norm().item()
        metrics['monitor/v_term_norm'] = self.v_term.data.norm().item()

        # --- Vocabulary embeddings ---
        ve_norms = self.vocab_emb.data.T.norm(dim=-1)
        metrics['monitor/vocab_emb_norm_mean'] = ve_norms.mean().item()
        metrics['monitor/vocab_emb_norm_std'] = ve_norms.std().item()

        # --- Temperature ---
        metrics['monitor/tau_root'] = self.log_tau_root.data.exp().item()
        metrics['monitor/tau_term'] = self.log_tau_term.data.exp().item()
        metrics['monitor/tau_rule'] = self.log_tau_rule.data.exp().item()

        return metrics

    # ------------------------------------------------------------------
    # Loss / evaluation
    # ------------------------------------------------------------------

    def loss(self, input):
        rules = self.forward(input)
        result = self.pcfg._inside(rules=rules, lens=input['seq_len'])
        return -result['partition'].mean()

    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input)
        if decode_type == 'mbr':
            # First pass: MBR span prediction.
            result = self.pcfg.decode(rules=rules, lens=input['seq_len'],
                                      viterbi=False, mbr=True)
            # Optional second pass: per-span NT marginals + per-leaf PT argmax for
            # label visualisation (scripts/render_parse_pdf.py). Gated so normal
            # eval/training is not slowed by the extra inside pass.
            if kwargs.get('return_labels'):
                rules2 = self.forward(input)
                with torch.enable_grad():
                    dist_out = self.pcfg._inside(rules=rules2, lens=input['seq_len'],
                                                 span_dist=True)
                span_marginals = dist_out['span_marginals']  # (B, N, N, NT)
                pt_argmax = rules2['unary'].detach().argmax(-1).cpu().tolist()
                nt_argmax_full = span_marginals.detach().argmax(-1).cpu().tolist()
                lens_list = input['seq_len'].cpu().tolist()
                result['pt_labels'] = [pt_argmax[b][: lens_list[b]]
                                       for b in range(len(lens_list))]
                nt_labels = []
                for b, spans in enumerate(result['prediction']):
                    span_to_nt = {}
                    for (i, j) in spans:
                        if j - i >= 2 and 0 <= i < len(nt_argmax_full[b]) and 0 <= j < len(nt_argmax_full[b][i]):
                            span_to_nt[(int(i), int(j))] = int(nt_argmax_full[b][i][j])
                    nt_labels.append(span_to_nt)
                result['nt_labels'] = nt_labels
            return result
        raise NotImplementedError(f"decode_type={decode_type}")
