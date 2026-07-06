"""PCA visualization of embedding spaces for the MLP-free HN-PCFG model."""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

# ── Paths ──────────────────────────────────────────────────────────────────
CKPT_PATH = Path(
    "/workspace/hol-pcfg-seminfo/ckpt/holeterm-nt1024/ckpt-sf1_val/"
    "sentence_f1=0.67-v2.ckpt"
)
VOCAB_PATH = Path(
    "/workspace/hol-pcfg/data/seminfo/"
    "ptb_en-full.gd_instruction.batch.gpt4omini-ew-exp-tbtok-idf/vocab.pkl"
)
OUT_DIR = Path("/workspace/hol-pcfg/results/pca_visualization")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication-quality defaults ───────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "svg.fonttype": "none",  # editable text in SVG
})

# Colorblind-friendly palette (Okabe-Ito inspired)
C_NT = "#0072B2"      # blue
C_PT = "#D55E00"      # vermillion
C_VOCAB = "#009E73"   # green
C_SPECIAL = "#F0E442" # yellow
C_RULE_TPL = "#CC79A7" # pink
C_TERM_TPL = "#56B4E9" # sky blue


def circonv(v: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Circular convolution: IFFT(FFT(v) * FFT(e)), real part."""
    return np.fft.ifft(np.fft.fft(v, axis=-1) * np.fft.fft(e, axis=-1), axis=-1).real


# ── Load checkpoint ────────────────────────────────────────────────────────
print("Loading checkpoint …")
sd = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)["state_dict"]

rule_state_emb = sd["model.rule_state_emb"].numpy()       # (3072, 512)
vocab_emb = sd["model.vocab_emb"].numpy()                  # (512, 10020)
v_term = sd["model.v_term"].numpy()                        # (512,)
v_left = sd["model.v_left"].numpy().squeeze()              # (512,)
v_right = sd["model.v_right"].numpy().squeeze()            # (512,)
root_emb = sd["model.root_emb"].numpy().squeeze()          # (512,)

NT = 1024
T = 2048
nt_emb = rule_state_emb[:NT]        # (1024, 512)
pt_emb = rule_state_emb[NT:]        # (2048, 512)
vocab_emb_t = vocab_emb.T           # (10020, 512)

print(f"  rule_state_emb: {rule_state_emb.shape}")
print(f"  vocab_emb_t:    {vocab_emb_t.shape}")
print(f"  v_term: {v_term.shape}, v_left: {v_left.shape}, v_right: {v_right.shape}")

# ── Load vocabulary ────────────────────────────────────────────────────────
print("Loading vocabulary …")


class _Dummy:
    pass


class _CustomUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "parser.helper.vocab":
            return type(name, (), {})
        return super().find_class(module, name)


with open(VOCAB_PATH, "rb") as f:
    vocab_obj = _CustomUnpickler(f).load()

idx2word: list[str] = vocab_obj.idx2word  # length 10020
word2idx: dict[str, int] = vocab_obj.word2idx
print(f"  Vocabulary size: {len(idx2word)}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 1 – 2D PCA of rule_state_emb (NT vs PT)
# ══════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] PCA of rule_state_emb (NT vs PT) …")
pca1 = PCA(n_components=2)
proj1 = pca1.fit_transform(rule_state_emb)  # (3072, 2)

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(proj1[:NT, 0], proj1[:NT, 1], s=4, alpha=0.4, c=C_NT, label=f"NT (n={NT})")
ax1.scatter(proj1[NT:, 0], proj1[NT:, 1], s=4, alpha=0.4, c=C_PT, label=f"PT (n={T})")

# Project special vectors
specials = {
    "v_term": v_term,
    "v_left": v_left,
    "v_right": v_right,
    "root_emb": root_emb,
}
special_proj = pca1.transform(np.stack(list(specials.values())))
for i, (name, _) in enumerate(specials.items()):
    ax1.scatter(
        special_proj[i, 0], special_proj[i, 1],
        s=200, marker="*", c=C_SPECIAL, edgecolors="black", linewidths=0.7, zorder=5,
    )
    ax1.annotate(
        name, (special_proj[i, 0], special_proj[i, 1]),
        fontsize=9, fontweight="bold",
        xytext=(8, 8), textcoords="offset points",
    )

evr = pca1.explained_variance_ratio_
ax1.set_xlabel(f"PC1 ({evr[0]:.1%})")
ax1.set_ylabel(f"PC2 ({evr[1]:.1%})")
ax1.set_title("PCA of Entity Embeddings (NT vs PT)")
ax1.legend(loc="best", markerscale=3)
ax1.text(
    0.02, 0.02,
    f"Explained variance: PC1={evr[0]:.2%}, PC2={evr[1]:.2%}\nTotal={evr[:2].sum():.2%}",
    transform=ax1.transAxes, fontsize=9, va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
)
fig1.tight_layout()
fig1.savefig(OUT_DIR / "fig1_pca_nt_vs_pt.svg", format="svg")
plt.close(fig1)
print(f"  Explained variance PC1={evr[0]:.4f}, PC2={evr[1]:.4f}, sum={evr[:2].sum():.4f}")

# ══════════════════════════════════════════════════════════════════════════
# Fig 2 – 2D PCA of vocab_emb
# ══════════════════════════════════════════════════════════════════════════
print("\n[Fig 2] PCA of vocabulary embeddings …")
pca2 = PCA(n_components=2)
proj2 = pca2.fit_transform(vocab_emb_t)  # (10020, 2)

# Compute word frequencies from vocab (use word2idx order to map back)
# For frequency, use the training data word counts
train_data = pickle.load(open("/workspace/hol-pcfg/data/ptb-train.pickle", "rb"))
from collections import Counter

word_freq: Counter = Counter()
for sent in train_data["word"]:
    word_freq.update(sent)

# Map frequency to vocab indices
freq_by_idx = np.zeros(len(idx2word))
for word, idx in word2idx.items():
    freq_by_idx[idx] = word_freq.get(word, 0)

# Normalise for coloring (log scale)
log_freq = np.log1p(freq_by_idx)
log_freq_norm = log_freq / (log_freq.max() + 1e-8)

fig2, ax2 = plt.subplots(figsize=(8, 6))
sc = ax2.scatter(
    proj2[:, 0], proj2[:, 1],
    s=3, alpha=0.3, c=log_freq_norm, cmap="viridis",
)
cbar = fig2.colorbar(sc, ax=ax2, label="log(1 + freq) (normalised)")

# Label top-20 frequent real words (skip special tokens)
real_words = [(w, idx) for w, idx in word2idx.items() if not w.startswith("<")]
real_words.sort(key=lambda x: freq_by_idx[x[1]], reverse=True)
for w, idx in real_words[:20]:
    ax2.annotate(
        w, (proj2[idx, 0], proj2[idx, 1]),
        fontsize=7, alpha=0.9,
        xytext=(5, 5), textcoords="offset points",
        arrowprops=dict(arrowstyle="-", lw=0.3, color="gray"),
    )
    ax2.scatter(proj2[idx, 0], proj2[idx, 1], s=20, c="red", edgecolors="black", linewidths=0.3, zorder=5)

evr2 = pca2.explained_variance_ratio_
ax2.set_xlabel(f"PC1 ({evr2[0]:.1%})")
ax2.set_ylabel(f"PC2 ({evr2[1]:.1%})")
ax2.set_title("PCA of Vocabulary Embeddings")
ax2.text(
    0.02, 0.02,
    f"Explained variance: PC1={evr2[0]:.2%}, PC2={evr2[1]:.2%}",
    transform=ax2.transAxes, fontsize=9, va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "fig2_pca_vocab.svg", format="svg")
plt.close(fig2)
print(f"  Explained variance PC1={evr2[0]:.4f}, PC2={evr2[1]:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# Fig 3 – 2D PCA of template space (terminal vs rule templates)
# ══════════════════════════════════════════════════════════════════════════
print("\n[Fig 3] PCA of terminal templates vs rule templates …")
np.random.seed(42)
sample_idx = np.random.choice(T, 200, replace=False)
sample_pt = pt_emb[sample_idx]  # (200, 512)

# Terminal templates: circonv(v_term, e_A) for each preterminal A
term_templates = circonv(v_term[None, :], sample_pt)  # (200, 512)

# Rule templates: circonv(v_left, e_A) for the same preterminals
rule_templates = circonv(v_left[None, :], sample_pt)   # (200, 512)

combined_tpl = np.vstack([term_templates, rule_templates])  # (400, 512)
pca3 = PCA(n_components=2)
proj3 = pca3.fit_transform(combined_tpl)

# For each preterminal, find top-1 word
# emission_logit = circonv(v_term, e_A) @ vocab_emb  →  (200, 10020)
emission_logits = term_templates @ vocab_emb  # (200, 10020)
top1_word_idx = emission_logits.argmax(axis=1)
top1_words = [idx2word[i] for i in top1_word_idx]

# Categorise top-1 words into simple groups
def categorise(w: str) -> str:
    if w.startswith("<"):
        return "special"
    if w[0].isupper():
        return "capitalised"
    if w.isdigit() or w == "N":
        return "number"
    if len(w) <= 3:
        return "function"
    return "content"


categories = [categorise(w) for w in top1_words]
cat_set = sorted(set(categories))
cat_colors = {
    "special": "#999999",
    "capitalised": "#E69F00",
    "number": "#56B4E9",
    "function": "#009E73",
    "content": "#0072B2",
}

fig3, ax3 = plt.subplots(figsize=(8, 6))
for cat in cat_set:
    mask = np.array([c == cat for c in categories])
    ax3.scatter(
        proj3[:200][mask, 0], proj3[:200][mask, 1],
        s=15, alpha=0.6, c=cat_colors.get(cat, "gray"), label=f"term: {cat}",
    )
ax3.scatter(
    proj3[200:, 0], proj3[200:, 1],
    s=10, alpha=0.3, c=C_RULE_TPL, marker="x", label="rule (v_left)",
)

evr3 = pca3.explained_variance_ratio_
ax3.set_xlabel(f"PC1 ({evr3[0]:.1%})")
ax3.set_ylabel(f"PC2 ({evr3[1]:.1%})")
ax3.set_title("PCA of Terminal Templates vs Rule Templates")
ax3.legend(loc="best", fontsize=8, markerscale=2)
ax3.text(
    0.02, 0.02,
    f"Explained variance: PC1={evr3[0]:.2%}, PC2={evr3[1]:.2%}",
    transform=ax3.transAxes, fontsize=9, va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
)
fig3.tight_layout()
fig3.savefig(OUT_DIR / "fig3_pca_templates.svg", format="svg")
plt.close(fig3)
print(f"  Explained variance PC1={evr3[0]:.4f}, PC2={evr3[1]:.4f}")
print(f"  Top-1 word categories: {Counter(categories)}")

# ══════════════════════════════════════════════════════════════════════════
# Fig 4 – 3D PCA of combined space
# ══════════════════════════════════════════════════════════════════════════
print("\n[Fig 4] 3D PCA of combined embedding space …")
np.random.seed(123)
nt_sub = nt_emb[np.random.choice(NT, 200, replace=False)]
pt_sub = pt_emb[np.random.choice(T, 200, replace=False)]
vocab_sub = vocab_emb_t[np.random.choice(vocab_emb_t.shape[0], 500, replace=False)]

combined_4 = np.vstack([nt_sub, pt_sub, vocab_sub])  # (900, 512)
pca4 = PCA(n_components=3)
proj4 = pca4.fit_transform(combined_4)

fig4 = plt.figure(figsize=(10, 8))
ax4 = fig4.add_subplot(111, projection="3d")
ax4.scatter(proj4[:200, 0], proj4[:200, 1], proj4[:200, 2], s=8, alpha=0.5, c=C_NT, label="NT (200)")
ax4.scatter(proj4[200:400, 0], proj4[200:400, 1], proj4[200:400, 2], s=8, alpha=0.5, c=C_PT, label="PT (200)")
ax4.scatter(proj4[400:, 0], proj4[400:, 1], proj4[400:, 2], s=4, alpha=0.3, c=C_VOCAB, label="Vocab (500)")

# Project special vectors into this PCA space
sp_proj4 = pca4.transform(np.stack([v_term, v_left, v_right, root_emb]))
sp_names = ["v_term", "v_left", "v_right", "root_emb"]
for i, name in enumerate(sp_names):
    ax4.scatter(
        sp_proj4[i, 0], sp_proj4[i, 1], sp_proj4[i, 2],
        s=200, marker="*", c=C_SPECIAL, edgecolors="black", linewidths=0.7, zorder=5,
    )
    ax4.text(sp_proj4[i, 0], sp_proj4[i, 1], sp_proj4[i, 2], f" {name}", fontsize=8, fontweight="bold")

evr4 = pca4.explained_variance_ratio_
ax4.set_xlabel(f"PC1 ({evr4[0]:.1%})", labelpad=8)
ax4.set_ylabel(f"PC2 ({evr4[1]:.1%})", labelpad=8)
ax4.set_zlabel(f"PC3 ({evr4[2]:.1%})", labelpad=8)
ax4.set_title("3D PCA of Combined Embedding Space")
ax4.legend(loc="upper left", fontsize=9, markerscale=2)
fig4.tight_layout()
fig4.savefig(OUT_DIR / "fig4_pca_3d_combined.svg", format="svg")
plt.close(fig4)
print(f"  Explained variance PC1={evr4[0]:.4f}, PC2={evr4[1]:.4f}, PC3={evr4[2]:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# Fig 5 – Explained variance plot
# ══════════════════════════════════════════════════════════════════════════
print("\n[Fig 5] Explained variance of entity embeddings …")
n_components = 200
pca5 = PCA(n_components=n_components)
pca5.fit(rule_state_emb)
cum_var = np.cumsum(pca5.explained_variance_ratio_)

# Random baseline: PCA on random data of same shape
rng = np.random.RandomState(0)
random_data = rng.randn(*rule_state_emb.shape)
pca5_rand = PCA(n_components=n_components)
pca5_rand.fit(random_data)
cum_var_rand = np.cumsum(pca5_rand.explained_variance_ratio_)

fig5, ax5 = plt.subplots(figsize=(8, 6))
components = np.arange(1, n_components + 1)
ax5.plot(components, cum_var, "o-", markersize=4, color=C_NT, label="Entity embeddings")
ax5.plot(components, cum_var_rand, "s--", markersize=3, color="#999999", label="Random baseline")
ax5.axhline(y=0.9, color=C_PT, linestyle=":", linewidth=1, label="90% threshold")

# Mark where 90% is reached
idx_90 = np.searchsorted(cum_var, 0.9) + 1
if idx_90 <= n_components:
    ax5.axvline(x=idx_90, color=C_PT, linestyle=":", linewidth=0.8, alpha=0.5)
    ax5.annotate(
        f"90% at PC{idx_90}",
        (idx_90, 0.9), fontsize=9,
        xytext=(idx_90 + 3, 0.85),
        arrowprops=dict(arrowstyle="->", color=C_PT),
    )

ax5.set_xlabel("Number of principal components")
ax5.set_ylabel("Cumulative explained variance ratio")
ax5.set_title("PCA Explained Variance (Entity Embeddings)")
ax5.legend(loc="lower right")
ax5.set_xlim(1, n_components)
ax5.set_ylim(0, 1.02)
fig5.tight_layout()
fig5.savefig(OUT_DIR / "fig5_explained_variance.svg", format="svg")
plt.close(fig5)
print(f"  90% variance reached at PC{idx_90}")
print(f"  First 10 components capture {cum_var[9]:.2%}")
print(f"  First 50 components capture {cum_var[49]:.2%}")
print(f"  First 100 components capture {cum_var[99]:.2%}")
print(f"  First 200 components capture {cum_var[199]:.2%}")

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("サマリー統計")
print("=" * 60)
print(f"  rule_state_emb: shape={rule_state_emb.shape}, "
      f"norm mean={np.linalg.norm(rule_state_emb, axis=1).mean():.4f}, "
      f"std={np.linalg.norm(rule_state_emb, axis=1).std():.4f}")
print(f"  NT embeddings:  norm mean={np.linalg.norm(nt_emb, axis=1).mean():.4f}")
print(f"  PT embeddings:  norm mean={np.linalg.norm(pt_emb, axis=1).mean():.4f}")
print(f"  vocab_emb_t:    shape={vocab_emb_t.shape}, "
      f"norm mean={np.linalg.norm(vocab_emb_t, axis=1).mean():.4f}")
print(f"  v_term norm:    {np.linalg.norm(v_term):.4f}")
print(f"  v_left norm:    {np.linalg.norm(v_left):.4f}")
print(f"  v_right norm:   {np.linalg.norm(v_right):.4f}")
print(f"  root_emb norm:  {np.linalg.norm(root_emb):.4f}")

# Cosine similarities between special vectors
def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print(f"\n  コサイン類似度 (特殊ベクトル間):")
print(f"    v_term ↔ v_left:  {cos_sim(v_term, v_left):.4f}")
print(f"    v_term ↔ v_right: {cos_sim(v_term, v_right):.4f}")
print(f"    v_left ↔ v_right: {cos_sim(v_left, v_right):.4f}")
print(f"    root   ↔ v_left:  {cos_sim(root_emb, v_left):.4f}")
print(f"    root   ↔ v_right: {cos_sim(root_emb, v_right):.4f}")
print(f"    root   ↔ v_term:  {cos_sim(root_emb, v_term):.4f}")

# NT-PT overlap measure
nt_mean = nt_emb.mean(axis=0)
pt_mean = pt_emb.mean(axis=0)
print(f"\n  NT重心 ↔ PT重心 コサイン類似度: {cos_sim(nt_mean, pt_mean):.4f}")
print(f"  NT重心 ↔ PT重心 ユークリッド距離: {np.linalg.norm(nt_mean - pt_mean):.4f}")

print(f"\n  PCA分散 (rule_state_emb):")
print(f"    PC1: {pca1.explained_variance_ratio_[0]:.4f}")
print(f"    PC2: {pca1.explained_variance_ratio_[1]:.4f}")
print(f"    累積50成分: {cum_var[49]:.4f}")
print(f"    90%到達: PC{idx_90}")

print(f"\n  出力先: {OUT_DIR}")
for f in sorted(OUT_DIR.glob("*.svg")):
    print(f"    {f.name}")

print("\n完了。")
