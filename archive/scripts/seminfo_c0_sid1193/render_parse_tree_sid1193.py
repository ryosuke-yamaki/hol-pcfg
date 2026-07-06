r"""Render the SemInfo NT=1024 parse tree for sid=1193 as a constituent tree SVG.

Tree structure (from gold spans + model NT/PT argmax):
                    S (NT-598)
                  /            \
            NP (NT-847)        VP (NT-584)
            /        \         / | \
       NP (660)    PP (342)  was n't VP (?)
       /  \       /  \              |
     the value  of  NP (?)         disclosed
                    /  \
                  the  acquisition

The 8 token-level PTs are rendered as plain text under their words.
"""
import argparse
import sys, pickle, math
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

REPO = Path('/workspace/hol-pcfg-seminfo')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'parsing_by_maxseminfo'))
sys.path.insert(0, str(REPO / 'scripts/c0_phase_landscape'))
from easydict import EasyDict as edict
from parsing_by_maxseminfo.parser.model.HN_PCFG import HNPCFGFixedCostReward
from cossin_pca_viz import _NT_VISUAL, _OTHER_N_COLOR, nt_visual


# ---------------------------------------------------------------------------
# CLI overrides. Defaults reproduce the rank3_seed4 figure shipped with PR #25.
_ap = argparse.ArgumentParser()
_ap.add_argument('--ckpt', default='ckpt/optuna/hnpcfg-rank1-seminfo-v3/phase2_rank3_seed4_0417_173559/best.ckpt')
_ap.add_argument('--label_json', default='results/c0_phase_landscape/english/n7e2qm8t/label/symbol_labels_english_phase2_rank3_seed4_0417_173559.json')
_ap.add_argument('--out_dir', default='results/c0_phase_landscape/english/n7e2qm8t/label')
_ap.add_argument('--stem', default='parse_tree_sid1193_seminfo_hn_NT1024_sdim512_phase2_rank3_seed4_0417_173559')
_args = _ap.parse_args()

# === Load model + parse ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = REPO / _args.ckpt
ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
hp = ck['hyper_parameters']; mp = edict(hp['model_params']); NT_count = mp['NT']
model = HNPCFGFixedCostReward(mp, hp['vocab_size'])
sd = {k[len('model.'):]: v for k, v in ck['state_dict'].items() if k.startswith('model.')}
model.load_state_dict(sd, strict=True); model.eval(); model.to(device)
with open(REPO / 'data/english/ptb_en-full.gd_instruction.batch.gpt4omini-ew-exp-tbtok-idf/vocab.pkl', 'rb') as f:
    word_vocab = pickle.load(f)

import json
with open(REPO / _args.label_json) as f:
    lm = json.load(f)
nt_label = lm['nt_label']

words = ['the', 'value', 'of', 'the', 'acquisition', 'was', "n't", 'disclosed']
pos_tags = ['DT', 'NN', 'IN', 'DT', 'NN', 'VBD', 'RB', 'VBN']
n = len(words)
ids = word_vocab._index(words)
x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
rules_out = model(input={'word': x})
out = model.pcfg._inside(rules_out, torch.tensor([n], device=device),
                          span_dist=True, include_unary=False)
nt_argmax = out['span_marginals'].argmax(dim=-1).detach().cpu().numpy()[0]
pt_argmax = rules_out['unary'].argmax(dim=-1).detach().cpu().numpy()[0]


# === Build tree from gold spans (binarized to match model's view) ===
# Gold spans for sid=1193:
#   S(0,8), NP(0,5), NP(0,2), PP(2,5), NP(3,5), VP(5,8), VP(7,8)
# We construct the binarised tree compatible with the model's parse.
# Note: VP(5,8) is flat (3 tokens, gold has nested VP(7,8) but model doesn't
# binary-decompose it).
def label_of_span(s, e):
    if e - s < 2:
        return None  # leaf, no NT
    idx = int(nt_argmax[s, e])
    return idx


def build_node(s, e, gold_label):
    """Build a tree node for span [s, e). Returns dict with label, NT,
    children, span, words."""
    nt = int(nt_argmax[s, e]) if e - s >= 2 else None
    node = {
        'span': (s, e),
        'gold': gold_label,
        'nt': nt,
        'global': nt_label[nt] if nt is not None else None,
        'children': [],
        'words': words[s:e],
    }
    return node


# Hand-build the tree mirroring the gold structure plus a binarised VP.
# The gold tree has VP(5,8) flat over "was n't disclosed" but a binary
# decomposition is required by the model and matches gold's separately
# labeled VP(7,8) "disclosed". We use a left-leaning split:
#   VP(5,8) -> NT(5,7) "was n't"  +  VP(7,8) "disclosed"
# Donut-endpoint NTs only. Each "leaf NT" carries its span text directly;
# the text is rendered just below the NT box rather than as a separate
# tree node, so no edge crosses the text.
root_S = build_node(0, 8, 'S')
np_0_5 = build_node(0, 5, 'NP')
np_0_2 = build_node(0, 2, 'NP')
pp_2_5 = build_node(2, 5, 'PP')
vp_5_8 = build_node(5, 8, 'VP')

np_0_2['span_text'] = 'the value'
pp_2_5['span_text'] = 'of the acquisition'
vp_5_8['span_text'] = "was n't disclosed"

root_S['children'] = [np_0_5, vp_5_8]
np_0_5['children'] = [np_0_2, pp_2_5]
# np_0_2 / pp_2_5 / vp_5_8 stay as leaves (no children).


# === Layout ===
# Compress vertical spacing: each depth step = Y_SCALE units instead of 1.0,
# so the lines between NT boxes are short and the tree stays compact.
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

# Compress horizontally around the sentence center so the top-level S → NP
# and S → VP edges are short. Leaf NT positions move inward proportionally,
# the text below them slides with them.
def _collect_pre(node, out):
    out.append(node)
    for c in node['children']:
        _collect_pre(c, out)

X_SCALE = 0.55
center_x = (len(words) - 1) / 2.0
_all_for_scale = []
_collect_pre(root_S, _all_for_scale)
for _n in _all_for_scale:
    _n['x'] = center_x + (_n['x'] - center_x) * X_SCALE

# Find max depth + leaf positions
def collect(node, out):
    out.append(node)
    for c in node['children']:
        collect(c, out)

all_nodes = []
collect(root_S, all_nodes)
max_depth = max(n['depth'] for n in all_nodes)
max_y = max(n['y'] for n in all_nodes)
n_leaves = len(words)


# === Draw ===
def color_for_label(lab):
    if lab in _NT_VISUAL:
        return _NT_VISUAL[lab][0]
    return _OTHER_N_COLOR


fig, ax = plt.subplots(figsize=(7.0, 3.0))

# Coordinate system: x in [0, n_leaves-1], y in [0, max_depth + 2 (for word + PT)]
# We will draw y axis inverted (root at top).

def draw_edge(parent, child, child_idx=0):
    """Left child (index 0) -> solid line, right child (index 1) -> dashed.
    Matches the donut figure convention: solid = v^(L), dashed = v^(R)."""
    x1, y1 = parent['x'], parent['y']
    x2, y2 = child['x'], child['y']
    linestyle = '-' if child_idx == 0 else '--'
    ax.plot([x1, x2], [-y1, -y2], color='#222', linewidth=1.2, alpha=0.9,
            linestyle=linestyle, solid_capstyle='round', zorder=1)


def draw_nt_node(node):
    x, y = node['x'], -node['y']
    gold = node['gold']
    global_lab = node['global']
    nt_idx = node['nt']
    # Single-line label "NT-XX(label)"; label = model's global gold-argmax
    # for that NT (falls back to "other" when none).
    lab_for_box = global_lab if global_lab is not None else 'other'
    text = f'NT-{nt_idx}({lab_for_box})'
    # Box color from the GLOBAL label of the model NT (not gold) so the
    # palette is consistent with the donut figure.
    col = color_for_label(lab_for_box)
    box = FancyBboxPatch((x - 0.45, y - 0.12), 0.9, 0.24,
                         boxstyle='round,pad=0.03', linewidth=1.3,
                         edgecolor='black', facecolor=col, alpha=0.85,
                         zorder=2)
    ax.add_patch(box)
    r, g, b = matplotlib.colors.to_rgb(col)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    text_col = 'white' if lum < 0.5 else 'black'
    ax.text(x, y, text, ha='center', va='center', fontsize=9,
            fontweight='bold', color=text_col, zorder=3)


def draw_leaf(node):
    """Leaf NT: NT box first; span text immediately below."""
    draw_nt_node(node)
    x, y = node['x'], -node['y']
    txt = node.get('span_text', '')
    ax.text(x, y - 0.30, txt, ha='center', va='center', fontsize=10,
            fontstyle='italic', color='#000', zorder=3)


# Recursive draw
def draw(node):
    for ci, c in enumerate(node['children']):
        draw_edge(node, c, child_idx=ci)
        draw(c)
    if node['children']:
        draw_nt_node(node)
    else:
        draw_leaf(node)


draw(root_S)

# Tight y range based on scaled depths + small slack for span text below.
# Tight x range matched to the compressed leaf-NT positions.
min_x = min(n['x'] for n in all_nodes) - 0.8
max_x = max(n['x'] for n in all_nodes) + 0.8
ax.set_xlim(min_x, max_x)
ax.set_ylim(-max_y - 0.55, 0.45)
ax.set_aspect('auto')
ax.axis('off')

ax.text(center_x, 0.32,
        '"the value of the acquisition was n\'t disclosed"',
        ha='center', va='center', fontsize=12, fontstyle='italic',
        color='#000')

out_dir = REPO / _args.out_dir
stem = _args.stem
for ext in ['.svg', '.png', '.pdf']:
    p = out_dir / f'{stem}{ext}'
    if p.exists():
        p.unlink()
    fig.savefig(p, bbox_inches='tight', pad_inches=0.05, dpi=140)
    print(f'wrote {p}')
plt.close(fig)
