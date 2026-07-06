"""Render parse tree + donut in a single matplotlib figure (vertically stacked)
and save SVG / PNG / PDF.
"""
import sys, json, pickle, argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.proj3d import proj_transform

REPO = Path('/workspace/hol-pcfg-seminfo')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'parsing_by_maxseminfo'))
sys.path.insert(0, str(REPO / 'scripts/c0_phase_landscape'))
from easydict import EasyDict as edict
from parsing_by_maxseminfo.parser.model.HN_PCFG import HNPCFGFixedCostReward
from cossin_pca_viz import _NT_VISUAL, _OTHER_N_COLOR, load_state, nt_visual
from fdr_torus_viz import (
    compute_phases, _draw_donut_surface, _torus_xyz,
    trim_png_whitespace, make_groups, wrap_to_pi,
)


# ---------------------------------------------------------------------------
# Front-rendering helpers (same as the standalone donut script)
class Arrow3DFront(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return 1e9


def arrow_front(ax, pks, pls, dks, dls, R, r, color, linestyle, lw=2.2, alpha=0.95, head=22):
    t = np.linspace(0, 1, 49)
    pk_t = pks + t * dks
    pl_t = pls + t * dls
    xs = (R + r * np.cos(pl_t)) * np.cos(pk_t)
    ys = (R + r * np.cos(pl_t)) * np.sin(pk_t)
    zs = r * np.sin(pl_t)
    line, = ax.plot(xs[:-1], ys[:-1], zs[:-1], color=color, linewidth=lw,
                    alpha=alpha, linestyle=linestyle, zorder=1000)
    line.set_zorder(1000)
    arrow = Arrow3DFront(
        [xs[-2], xs[-1]], [ys[-2], ys[-1]], [zs[-2], zs[-1]],
        arrowstyle='-|>', mutation_scale=head,
        color=color, lw=0.0, alpha=min(1.0, alpha + 0.05), zorder=1001,
    )
    ax.add_artist(arrow)


def force_scatter_front(scatter):
    import types
    original = type(scatter).do_3d_projection

    def _front(self, renderer=None):
        try:
            original(self, renderer)
        except TypeError:
            original(self)
        return 1e9

    scatter.do_3d_projection = types.MethodType(_front, scatter)
    return scatter


def place_labels(endpoint_nts, phi_nt, k_idx, l_idx, R, r):
    sorted_eps = sorted(endpoint_nts)
    placed = []
    out = []
    for idx in sorted_eps:
        pk, pl = phi_nt[idx, k_idx], phi_nt[idx, l_idx]
        px, py, pz = _torus_xyz(pk, pl, R, r)
        radial = np.sqrt(px ** 2 + py ** 2)
        if radial > 0:
            lx = px * 1.10
            ly = py * 1.10
        else:
            lx = ly = 0.0
        lz = pz + (0.18 if pz >= 0 else -0.16)
        for plx, ply, plz in placed:
            for _ in range(5):
                d2 = (lx - plx) ** 2 + (ly - ply) ** 2
                if d2 >= 0.6 ** 2:
                    break
                lz += 0.28 if lz >= 0 else -0.28
        placed.append((lx, ly, lz))
        out.append((idx, lx, ly, lz, px, py, pz))
    return out


# ---------------------------------------------------------------------------
# CLI overrides. Defaults reproduce the rank2_seed5 figure; pass flags to
# render another checkpoint (paths are relative to REPO).
_ap = argparse.ArgumentParser()
_ap.add_argument('--ckpt', default='ckpt/optuna/hnpcfg-rank1-seminfo-v3/phase2_rank2_seed5_0604_041817/best.ckpt')
_ap.add_argument('--fdr_json', default='results/c0_phase_landscape/english/rank2_seed5/label/fdr_scores_gold_english_phase2_rank2_seed5_0604_041817.json')
_ap.add_argument('--label_json', default='results/c0_phase_landscape/english/rank2_seed5/label/symbol_labels_english_phase2_rank2_seed5_0604_041817.json')
_ap.add_argument('--out_dir', default='results/c0_phase_landscape/english/rank2_seed5/label')
_ap.add_argument('--stem', default='combined_sid1193_tree_top_donut_bottom_seminfo_hn_NT1024_sdim512_phase2_rank2_seed5_0604_041817')
_ap.add_argument('--label_offsets', default='{"349": [-0.15, 0.15, 0.30], "623": [0.35, 0.0, -0.45]}',
                 help='JSON {nt_index: [dx,dy,dz]} of manual donut-label nudges')
_args = _ap.parse_args()

# ---------------------------------------------------------------------------
# Load model + parse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = REPO / _args.ckpt
fdr_json = REPO / _args.fdr_json
label_json = REPO / _args.label_json

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
out = model.pcfg._inside(rules_out, torch.tensor([n], device=device),
                          span_dist=True, include_unary=False)
nt_argmax = out['span_marginals'].argmax(dim=-1).detach().cpu().numpy()[0]
phi_nt = compute_phases(sd['rule_state_emb'][:NT_count].float())
phi_L = compute_phases(sd['v_left'].float().unsqueeze(0))[0]
phi_R = compute_phases(sd['v_right'].float().unsqueeze(0))[0]

gold_rules = [(0, 5, 8, 'S', 'NP', 'VP'), (0, 2, 5, 'NP', 'NP', 'PP')]
rules = []
for (i, m, j, Pl, Ll, Rl) in gold_rules:
    A_P, A_L, A_R = int(nt_argmax[i, j]), int(nt_argmax[i, m]), int(nt_argmax[m, j])
    rules.append({'P_lab': Pl, 'L_lab': Ll, 'R_lab': Rl,
                  'A_P': A_P, 'A_L': A_L, 'A_R': A_R})

endpoint_nts = set()
for r in rules:
    endpoint_nts.update([r['A_P'], r['A_L'], r['A_R']])

with open(fdr_json) as f:
    fdr = json.load(f)
k_bin, l_bin = fdr['k_star_bin'], fdr['l_star_bin']
k_idx, l_idx = k_bin - 1, l_bin - 1

phrase_groups = ['NP', 'VP', 'PP', 'S', 'SBAR', 'ADJP', 'ADVP']
groups = make_groups(lm['nt_label'], lm['nt_support'], 5, phrase_groups)


def color_for_label(lab):
    if lab in phrase_groups:
        gi = phrase_groups.index(lab); col, _ = nt_visual(lab, gi); return col
    return _OTHER_N_COLOR


# ---------------------------------------------------------------------------
# Build combined figure: tree on top, donut on bottom. We explicitly place
# both subplots via `set_position` so the donut can use almost the entire
# figure width / lower 80 % of the height while leaving a thin strip at the
# top for the parse tree.
fig = plt.figure(figsize=(8.5, 9.2))
ax_tree = fig.add_axes([0.04, 0.80, 0.92, 0.18])
ax_donut = fig.add_axes([-0.10, -0.13, 1.10, 0.93], projection='3d')


# ----- Tree (top) ----------------------------------------------------------
class Node(dict):
    pass

def make_node(span, gold):
    s, e = span
    nt = int(nt_argmax[s, e]) if e - s >= 2 else None
    return Node(span=span, gold=gold, nt=nt,
                global_lab=(nt_label[nt] if nt is not None else None),
                children=[])

root_S = make_node((0, 8), 'S')
np_0_5 = make_node((0, 5), 'NP')
np_0_2 = make_node((0, 2), 'NP')
pp_2_5 = make_node((2, 5), 'PP')
vp_5_8 = make_node((5, 8), 'VP')
np_0_2['span_text'] = 'the value'
pp_2_5['span_text'] = 'of the acquisition'
vp_5_8['span_text'] = "was n't disclosed"
root_S['children'] = [np_0_5, vp_5_8]
np_0_5['children'] = [np_0_2, pp_2_5]

Y_SCALE = 0.55

def assign_layout(node, depth=0):
    node['depth'] = depth
    if not node['children']:
        s, e = node['span']
        node['x'] = (s + e - 1) / 2.0
        node['y'] = depth * Y_SCALE
        return
    for c in node['children']:
        assign_layout(c, depth + 1)
    xs = [c['x'] for c in node['children']]
    node['x'] = sum(xs) / len(xs)
    node['y'] = depth * Y_SCALE

assign_layout(root_S)

def collect(node, out):
    out.append(node)
    for c in node['children']:
        collect(c, out)

all_nodes = []
collect(root_S, all_nodes)

X_SCALE = 0.55
center_x = (len(words) - 1) / 2.0
for n_ in all_nodes:
    n_['x'] = center_x + (n_['x'] - center_x) * X_SCALE


def draw_edge(parent, child, child_idx):
    x1, y1 = parent['x'], parent['y']
    x2, y2 = child['x'], child['y']
    linestyle = '-' if child_idx == 0 else '--'
    ax_tree.plot([x1, x2], [-y1, -y2], color='#222', linewidth=1.2,
                 alpha=0.9, linestyle=linestyle, solid_capstyle='round', zorder=1)


def draw_nt_box(node):
    x, y = node['x'], -node['y']
    nt_idx = node['nt']
    lab = node['global_lab'] if node['global_lab'] is not None else 'other'
    col = color_for_label(lab)
    box = FancyBboxPatch((x - 0.45, y - 0.12), 0.9, 0.24,
                         boxstyle='round,pad=0.03', linewidth=1.3,
                         edgecolor='black', facecolor=col, alpha=0.85,
                         zorder=2)
    ax_tree.add_patch(box)
    rb, gb, bb = matplotlib.colors.to_rgb(col)
    lum = 0.299 * rb + 0.587 * gb + 0.114 * bb
    text_col = 'white' if lum < 0.5 else 'black'
    ax_tree.text(x, y, f'NT-{nt_idx}({lab})', ha='center', va='center',
                 fontsize=9, fontweight='bold', color=text_col, zorder=3)


def draw_leaf(node):
    draw_nt_box(node)
    x, y = node['x'], -node['y']
    ax_tree.text(x, y - 0.30, node.get('span_text', ''),
                 ha='center', va='center', fontsize=10,
                 fontstyle='italic', color='#000', zorder=3)


def draw(node):
    for ci, c in enumerate(node['children']):
        draw_edge(node, c, ci)
        draw(c)
    if node['children']:
        draw_nt_box(node)
    else:
        draw_leaf(node)


draw(root_S)
max_y = max(n_['y'] for n_ in all_nodes)
min_x = min(n_['x'] for n_ in all_nodes) - 0.8
max_x = max(n_['x'] for n_ in all_nodes) + 0.8
ax_tree.set_xlim(min_x, max_x)
ax_tree.set_ylim(-max_y - 0.55, 0.45)
ax_tree.set_aspect('auto')
ax_tree.axis('off')
ax_tree.text(center_x, 0.32,
             '"the value of the acquisition was n\'t disclosed"',
             ha='center', va='center', fontsize=12, fontstyle='italic',
             color='#000')

# ----- Donut (bottom) ------------------------------------------------------
R_major, r_minor = 3.0, 0.7
elev, azim = 45, 45

_draw_donut_surface(ax_donut, R_major, r_minor, draw_wireframe=True)
nk, nl = phi_nt[:, k_idx], phi_nt[:, l_idx]
star_mask = np.array([i in endpoint_nts for i in range(NT_count)])
mask_other = (groups == 'other-N') & ~star_mask
if mask_other.any():
    xo, yo, zo = _torus_xyz(nk[mask_other], nl[mask_other], R_major, r_minor)
    ax_donut.scatter(xo, yo, zo, s=14, c=_OTHER_N_COLOR, alpha=0.6,
                     edgecolors='#3a3a3a', linewidths=0.15, marker='o', zorder=4)
nt_handles = []
for gi, g in enumerate(phrase_groups):
    mask = (groups == g) & ~star_mask
    if not mask.any():
        continue
    col, marker = nt_visual(g, gi)
    xn, yn, zn = _torus_xyz(nk[mask], nl[mask], R_major, r_minor)
    h = ax_donut.scatter(xn, yn, zn, s=40, color=col, alpha=0.85,
                          edgecolors='white', linewidths=0.2, marker=marker, zorder=5)
    nt_handles.append((g, h))

for idx in endpoint_nts:
    g_lab = nt_label[idx] if nt_label[idx] is not None else None
    col = color_for_label(g_lab) if g_lab in phrase_groups else _OTHER_N_COLOR
    px, py, pz = _torus_xyz(phi_nt[idx, k_idx], phi_nt[idx, l_idx], R_major, r_minor)
    sc = ax_donut.scatter([px], [py], [pz], s=40, color=col, alpha=1.0,
                           edgecolors='black', linewidths=1.0, marker='o', zorder=2000)
    force_scatter_front(sc)

ARROW_COLOR = '#1a1a1a'
for r in rules:
    dkL = wrap_to_pi(phi_nt[r['A_L'], k_idx] - phi_nt[r['A_P'], k_idx])
    dlL = wrap_to_pi(phi_nt[r['A_L'], l_idx] - phi_nt[r['A_P'], l_idx])
    arrow_front(ax_donut, float(phi_nt[r['A_P'], k_idx]),
                float(phi_nt[r['A_P'], l_idx]),
                float(dkL), float(dlL), R_major, r_minor, color=ARROW_COLOR,
                linestyle='solid', lw=2.2, alpha=0.95, head=22)
    dkR = wrap_to_pi(phi_nt[r['A_R'], k_idx] - phi_nt[r['A_P'], k_idx])
    dlR = wrap_to_pi(phi_nt[r['A_R'], l_idx] - phi_nt[r['A_P'], l_idx])
    arrow_front(ax_donut, float(phi_nt[r['A_P'], k_idx]),
                float(phi_nt[r['A_P'], l_idx]),
                float(dkR), float(dlR), R_major, r_minor, color=ARROW_COLOR,
                linestyle='dashed', lw=2.2, alpha=0.95, head=22)

# Manual donut-label nudges (keyed by endpoint NT index), from --label_offsets.
# Default separates rank2_seed5's NT-349(NP)/NT-623(PP), which land ~0.8 apart
# (just above place_labels' 0.6 auto-separation threshold).
manual_label_offset = {int(k): tuple(v) for k, v in json.loads(_args.label_offsets).items()}
for idx, lx, ly, lz, px, py, pz in place_labels(endpoint_nts, phi_nt, k_idx, l_idx, R_major, r_minor):
    g_lab = nt_label[idx] if nt_label[idx] is not None else 'other'
    if idx in manual_label_offset:
        dx, dy, dz = manual_label_offset[idx]
        lx += dx; ly += dy; lz += dz
    ax_donut.plot([px, lx], [py, ly], [pz, lz], color='#000', linewidth=0.9,
                  alpha=0.85, zorder=900)
    ax_donut.text(lx, ly, lz, f'NT-{idx}({g_lab})',
                  fontsize=10, fontweight='bold', ha='center', va='center',
                  zorder=2000,
                  bbox=dict(boxstyle='round,pad=0.20', facecolor='white',
                            edgecolor='#444', linewidth=0.5, alpha=0.55))

ax_donut.set_box_aspect((1.0, 1.0, 2 * r_minor / (R_major + r_minor)))
ax_donut.view_init(elev=elev, azim=azim)
ax_donut.set_axis_off()
pad = 0.05
ax_donut.set_xlim(-(R_major + r_minor) - pad, (R_major + r_minor) + pad)
ax_donut.set_ylim(-(R_major + r_minor) - pad, (R_major + r_minor) + pad)
ax_donut.set_zlim(-r_minor - pad, r_minor + pad)
# (Donut position is set above via fig.add_axes.)

# Inset legends on the right of the donut
labels = [g for g, _ in nt_handles]
handles = [h for _, h in nt_handles]
other_handle = Line2D([0], [0], marker='o', color='none',
                      markerfacecolor=_OTHER_N_COLOR, markeredgecolor='none',
                      markersize=7, label='Other')
labels.append('Other'); handles.append(other_handle)
# Place legends inside the figure, overlaying the empty upper-right corner
# of the donut's bounding box (which is white space at this view angle).
leg1 = fig.legend(handles, labels, loc='upper right',
                  bbox_to_anchor=(0.98, 0.78),
                  fontsize=10, framealpha=0.92)
arrow_l = Line2D([0], [0], color=ARROW_COLOR, linewidth=2.2, linestyle='-',
                 label=r'left child  ($\mathbf{v}^{(L)}$)')
arrow_r = Line2D([0], [0], color=ARROW_COLOR, linewidth=2.2, linestyle='--',
                 label=r'right child  ($\mathbf{v}^{(R)}$)')
leg2 = fig.legend([arrow_l, arrow_r], [arrow_l.get_label(), arrow_r.get_label()],
                  loc='lower right', bbox_to_anchor=(0.98, 0.03),
                  fontsize=10, framealpha=0.92)
fig.add_artist(leg1)

out_dir = REPO / _args.out_dir
stem = _args.stem
for ext in ['.svg', '.png', '.pdf']:
    p = out_dir / f'{stem}{ext}'
    if p.exists():
        p.unlink()
    fig.savefig(p, dpi=140)
    if ext == '.png':
        trim_png_whitespace(p)
    print(f'wrote {p}')
plt.close(fig)
