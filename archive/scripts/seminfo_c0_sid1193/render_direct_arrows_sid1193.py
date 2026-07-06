import argparse
import sys, json, pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

REPO = Path('/workspace/hol-pcfg-seminfo')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'parsing_by_maxseminfo'))
sys.path.insert(0, str(REPO / 'scripts/c0_phase_landscape'))
from easydict import EasyDict as edict
from parsing_by_maxseminfo.parser.model.HN_PCFG import HNPCFGFixedCostReward
from cossin_pca_viz import _NT_VISUAL, _OTHER_N_COLOR, nt_visual
from fdr_torus_viz import (
    compute_phases, _draw_donut_surface, _torus_xyz,
    trim_png_whitespace, make_groups, wrap_to_pi,
)


class Arrow3DFront(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return 1e9


def force_scatter_front(scatter):
    """Override do_3d_projection on a 3D scatter so it sorts to the front.
    The original projection still runs so the 2D positions update normally;
    we just return a huge depth value (matching Arrow3DFront's convention)
    so matplotlib's painter sorting places this artist on top.
    """
    import types
    original = type(scatter).do_3d_projection

    def _front(self, renderer=None):
        try:
            original(self, renderer)
        except TypeError:
            # Newer matplotlib drops the renderer arg on collection projections.
            original(self)
        return 1e9

    scatter.do_3d_projection = types.MethodType(_front, scatter)
    return scatter


def arrow_front(ax, pks, pls, dks, dls, R, r, color, linestyle, lw=2.2, alpha=0.95, head=22):
    t = np.linspace(0, 1, 49)
    pk_t = pks + t * dks
    pl_t = pls + t * dls
    xs = (R + r * np.cos(pl_t)) * np.cos(pk_t)
    ys = (R + r * np.cos(pl_t)) * np.sin(pk_t)
    zs = r * np.sin(pl_t)
    # Plot the line up to xs[-2] only, so the arrow head triangle does not
    # have a black line extending past its tip (issue #2).
    line, = ax.plot(xs[:-1], ys[:-1], zs[:-1], color=color, linewidth=lw,
                    alpha=alpha, linestyle=linestyle, zorder=1000)
    line.set_zorder(1000)
    arrow = Arrow3DFront(
        [xs[-2], xs[-1]], [ys[-2], ys[-1]], [zs[-2], zs[-1]],
        arrowstyle='-|>', mutation_scale=head,
        color=color, lw=0.0, alpha=min(1.0, alpha + 0.05), zorder=1001,
    )
    ax.add_artist(arrow)


def place_labels(endpoint_nts, phi_nt, k_idx, l_idx, R, r):
    """Place labels just slightly outside the NT point (issue #1)."""
    sorted_eps = sorted(endpoint_nts)
    positions = []
    placed = []
    for idx in sorted_eps:
        pk, pl = phi_nt[idx, k_idx], phi_nt[idx, l_idx]
        px, py, pz = _torus_xyz(pk, pl, R, r)
        radial = np.sqrt(px ** 2 + py ** 2)
        if radial > 0:
            lx = px * 1.10   # was 1.30 - much closer to the dot now
            ly = py * 1.10
        else:
            lx = ly = 0.0
        lz = pz + (0.18 if pz >= 0 else -0.16)  # was 0.35 - closer in z too
        for plx, ply, plz in placed:
            for _ in range(5):
                d2 = (lx - plx) ** 2 + (ly - ply) ** 2
                if d2 >= 0.6 ** 2:
                    break
                lz += 0.28 if lz >= 0 else -0.28
        placed.append((lx, ly, lz))
        positions.append((idx, lx, ly, lz, px, py, pz))
    return positions


# ---------------------------------------------------------------------------
# CLI overrides. Defaults reproduce the rank3_seed4 figure shipped with PR #25.
_ap = argparse.ArgumentParser()
_ap.add_argument('--ckpt', default='ckpt/optuna/hnpcfg-rank1-seminfo-v3/phase2_rank3_seed4_0417_173559/best.ckpt')
_ap.add_argument('--fdr_json', default='results/c0_phase_landscape/english/n7e2qm8t/label/fdr_scores_gold_english_phase2_rank3_seed4_0417_173559.json')
_ap.add_argument('--label_json', default='results/c0_phase_landscape/english/n7e2qm8t/label/symbol_labels_english_phase2_rank3_seed4_0417_173559.json')
_ap.add_argument('--out_dir', default='results/c0_phase_landscape/english/n7e2qm8t/label')
_ap.add_argument('--stem', default='fdr_torus_gold_DIRECT_arrows_sid1193_view_top_45_clean_labeled_english_phase2_rank3_seed4_0417_173559')
_ap.add_argument('--label_offsets', default='{"660": [-0.3, 0.4, 0.25]}',
                 help='JSON {nt_index: [dx,dy,dz]} of manual donut-label nudges')
_args = _ap.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = REPO / _args.ckpt
fdr_json = REPO / _args.fdr_json
label_json = REPO / _args.label_json
out_dir = REPO / _args.out_dir

ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
hp = ck['hyper_parameters']; mp = edict(hp['model_params']); NT_count = mp['NT']
model = HNPCFGFixedCostReward(mp, hp['vocab_size'])
sd = {k[len('model.'):]: v for k, v in ck['state_dict'].items() if k.startswith('model.')}
model.load_state_dict(sd, strict=True); model.eval(); model.to(device)
with open(REPO / 'data/english/ptb_en-full.gd_instruction.batch.gpt4omini-ew-exp-tbtok-idf/vocab.pkl', 'rb') as f:
    word_vocab = pickle.load(f)
with open(label_json) as f:
    lm = json.load(f)
nt_label = lm['nt_label']

words = ['the', 'value', 'of', 'the', 'acquisition', 'was', "n't", 'disclosed']
n = len(words)
ids = word_vocab._index(words)
x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
rules_out = model(input={'word': x})
out = model.pcfg._inside(rules_out, torch.tensor([n], device=device), span_dist=True, include_unary=False)
nt_argmax = out['span_marginals'].argmax(dim=-1).detach().cpu().numpy()[0]
phi_nt = compute_phases(sd['rule_state_emb'][:NT_count].float())

gold_rules = [(0, 5, 8, 'S', 'NP', 'VP'), (0, 2, 5, 'NP', 'NP', 'PP')]
rules = []
for (i, m, j, Pl, Ll, Rl) in gold_rules:
    A_P, A_L, A_R = int(nt_argmax[i, j]), int(nt_argmax[i, m]), int(nt_argmax[m, j])
    rules.append({'P_lab': Pl, 'L_lab': Ll, 'R_lab': Rl,
                  'A_P': A_P, 'A_L': A_L, 'A_R': A_R})
    print('  %s -> %s %s  A=(%d,%d,%d)' % (Pl, Ll, Rl, A_P, A_L, A_R))

endpoint_nts = set()
for r in rules:
    endpoint_nts.update([r['A_P'], r['A_L'], r['A_R']])

with open(fdr_json) as f: fdr = json.load(f)
k_bin, l_bin = fdr['k_star_bin'], fdr['l_star_bin']
k_idx, l_idx = k_bin - 1, l_bin - 1
R_major, r_minor = 3.0, 0.7
phrase_groups = ['NP', 'VP', 'PP', 'S', 'SBAR', 'ADJP', 'ADVP']
groups = make_groups(lm['nt_label'], lm['nt_support'], 5, phrase_groups)


def color_for_label(lab):
    if lab in phrase_groups:
        gi = phrase_groups.index(lab); col, _ = nt_visual(lab, gi); return col
    return _OTHER_N_COLOR


ARROW_COLOR = '#1a1a1a'
elev, azim = 45, 45

label_positions = place_labels(endpoint_nts, phi_nt, k_idx, l_idx, R_major, r_minor)

fig = plt.figure(figsize=(9, 5.5))
ax = fig.add_subplot(1, 1, 1, projection='3d')
_draw_donut_surface(ax, R_major, r_minor, draw_wireframe=True)
nk, nl = phi_nt[:, k_idx], phi_nt[:, l_idx]
mask_other = groups == 'other-N'
if mask_other.any():
    xo, yo, zo = _torus_xyz(nk[mask_other], nl[mask_other], R_major, r_minor)
    ax.scatter(xo, yo, zo, s=14, c=_OTHER_N_COLOR, alpha=0.6,
               edgecolors='#3a3a3a', linewidths=0.15, marker='o', zorder=4)
nt_handles = []
star_mask = np.array([i in endpoint_nts for i in range(NT_count)])
for gi, g in enumerate(phrase_groups):
    mask = (groups == g) & ~star_mask
    if not mask.any():
        continue
    col, marker = nt_visual(g, gi)
    xn, yn, zn = _torus_xyz(nk[mask], nl[mask], R_major, r_minor)
    h = ax.scatter(xn, yn, zn, s=40, color=col, alpha=0.85,
                   edgecolors='white', linewidths=0.2, marker=marker, zorder=5)
    nt_handles.append((g, h))

# Endpoint NTs: same dot size as the others (s=40) with black outline,
# and forced to render in front of everything else (donut surface, arrows,
# background NT scatter) via the Arrow3DFront-style depth override.
for idx in endpoint_nts:
    g_lab = nt_label[idx] if nt_label[idx] is not None else None
    col = color_for_label(g_lab) if g_lab in phrase_groups else _OTHER_N_COLOR
    px, py, pz = _torus_xyz(phi_nt[idx, k_idx], phi_nt[idx, l_idx], R_major, r_minor)
    sc = ax.scatter([px], [py], [pz], s=40, color=col, alpha=1.0,
                    edgecolors='black', linewidths=1.0, marker='o', zorder=2000)
    force_scatter_front(sc)

# Arrows (DIRECT, NT-to-NT)
for r in rules:
    dkL = wrap_to_pi(phi_nt[r['A_L'], k_idx] - phi_nt[r['A_P'], k_idx])
    dlL = wrap_to_pi(phi_nt[r['A_L'], l_idx] - phi_nt[r['A_P'], l_idx])
    arrow_front(ax, float(phi_nt[r['A_P'], k_idx]), float(phi_nt[r['A_P'], l_idx]),
                float(dkL), float(dlL), R_major, r_minor, color=ARROW_COLOR,
                linestyle='solid', lw=2.2, alpha=0.95, head=22)
    dkR = wrap_to_pi(phi_nt[r['A_R'], k_idx] - phi_nt[r['A_P'], k_idx])
    dlR = wrap_to_pi(phi_nt[r['A_R'], l_idx] - phi_nt[r['A_P'], l_idx])
    arrow_front(ax, float(phi_nt[r['A_P'], k_idx]), float(phi_nt[r['A_P'], l_idx]),
                float(dkR), float(dlR), R_major, r_minor, color=ARROW_COLOR,
                linestyle='dashed', lw=2.2, alpha=0.95, head=22)

# Labels — semi-transparent bbox + visible leader line (issue #1 follow-up)
# Per-NT manual offset to fine-tune positions that auto-placement misses
# (issue #3: NT-660 sits too close to NT-847 in the (k=6,l=248) projection;
# shift it down/right in donut space so the box clears the other one).
manual_label_offset = {int(k): tuple(v) for k, v in json.loads(_args.label_offsets).items()}
for idx, lx, ly, lz, px, py, pz in label_positions:
    g = nt_label[idx] if nt_label[idx] is not None else 'other'
    if idx in manual_label_offset:
        dx, dy, dz = manual_label_offset[idx]
        lx += dx; ly += dy; lz += dz
    ax.plot([px, lx], [py, ly], [pz, lz], color='#000', linewidth=0.9,
            alpha=0.85, zorder=900)
    ax.text(lx, ly, lz, 'NT-%d(%s)' % (idx, g),
            fontsize=10, fontweight='bold', ha='center', va='center', zorder=2000,
            bbox=dict(boxstyle='round,pad=0.20', facecolor='white',
                      edgecolor='#444', linewidth=0.5, alpha=0.55))

ax.set_box_aspect((1.0, 1.0, 2 * r_minor / (R_major + r_minor)))
ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
pad = 0.05
ax.set_xlim(-(R_major + r_minor) - pad, (R_major + r_minor) + pad)
ax.set_ylim(-(R_major + r_minor) - pad, (R_major + r_minor) + pad)
ax.set_zlim(-r_minor - pad, r_minor + pad)
ax.set_position([-0.10, -0.20, 0.92, 1.40])

# Legend 1: N gold categories + Other (issue #3+#4 follow-up)
labels = [g for g, _ in nt_handles]
handles = [h for _, h in nt_handles]
# Other entry: no black border, sized to match the others (scatter dots are
# s=40 so we use markersize ~ sqrt(40) ≈ 6 for visual parity).
other_handle = Line2D([0], [0], marker='o', color='none',
                      markerfacecolor=_OTHER_N_COLOR, markeredgecolor='none',
                      markersize=7, label='Other')
labels.append('Other'); handles.append(other_handle)
leg1 = fig.legend(handles, labels, loc='upper left',
                  bbox_to_anchor=(0.66, 0.92),
                  fontsize=10, framealpha=0.92)

# Legend 2: plain solid / dashed lines with parenthesised bold v notation.
arrow_l = Line2D([0], [0], color=ARROW_COLOR, linewidth=2.2, linestyle='-',
                 label=r'left child  ($\mathbf{v}^{(L)}$)')
arrow_r = Line2D([0], [0], color=ARROW_COLOR, linewidth=2.2, linestyle='--',
                 label=r'right child  ($\mathbf{v}^{(R)}$)')
leg2 = fig.legend([arrow_l, arrow_r], [arrow_l.get_label(), arrow_r.get_label()],
                  loc='upper left', bbox_to_anchor=(0.66, 0.40),
                  fontsize=11, framealpha=0.92)
fig.add_artist(leg1)
# leg3 removed (issue #5 last part)

stem = _args.stem
for ext in ['.png', '.svg', '.pdf']:
    p = out_dir / f'{stem}{ext}'
    if p.exists(): p.unlink()
    fig.savefig(p, dpi=140, bbox_inches='tight', pad_inches=0.05,
                bbox_extra_artists=(leg1, leg2))
    if ext == '.png': trim_png_whitespace(p)
plt.close(fig)
print('wrote %s' % stem)
