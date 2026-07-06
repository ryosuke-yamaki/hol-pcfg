# Third-Party Notices

This repository deliberately carries **no explicit open-source license**, consistent with
the upstream research code bases from which it derives — several of which likewise ship
without a license. No license is claimed over the first-party contributions in this fork
either. Copyright of every portion remains with its respective authors; the notices below
document provenance and attribution.

## TN-PCFG (base of this fork)

- Upstream: <https://github.com/sustcsonglin/TN-PCFG>
- License: none stated upstream (no `LICENSE` file in the source repository).

This repository is a fork of the TN-PCFG / NBL-PCFG / SimplePCFG code base; the retained
upstream models and the inside/decoding machinery derive from it. That code base implements:

- Yang et al., NAACL 2021 — "PCFGs Can Do Better: Inducing Probabilistic Context-Free
  Grammars with Many Symbols"
- Yang et al., ACL 2021 — "Neural Bi-Lexicalized PCFG Induction"
- Yang et al., NAACL 2022 — "Dynamic Programming in Rank Space: Scaling Structured
  Inference with Low-Rank HMMs and PCFGs"
- Liu et al., Findings of EMNLP 2023 — "Simple Hardware-Efficient PCFGs with Independent
  Left and Right Productions"

BibTeX for all four:

```bibtex
@inproceedings{yang-etal-2021-pcfgs,
    title = "{PCFG}s Can Do Better: Inducing Probabilistic Context-Free Grammars with Many Symbols",
    author = "Yang, Songlin and Zhao, Yanpeng and Tu, Kewei",
    booktitle = "Proceedings of NAACL-HLT 2021",
    year = "2021", url = "https://www.aclweb.org/anthology/2021.naacl-main.117", pages = "1487--1498",
}

@inproceedings{yang-etal-2021-neural,
    title = "Neural Bi-Lexicalized {PCFG} Induction",
    author = "Yang, Songlin and Zhao, Yanpeng and Tu, Kewei",
    booktitle = "Proceedings of ACL-IJCNLP 2021",
    year = "2021", url = "https://aclanthology.org/2021.acl-long.209", pages = "2688--2699",
}

@inproceedings{yang-etal-2022-dynamic,
    title = "Dynamic Programming in Rank Space: Scaling Structured Inference with Low-Rank {HMM}s and {PCFG}s",
    author = "Yang, Songlin and Liu, Wei and Tu, Kewei",
    booktitle = "Proceedings of NAACL-HLT 2022",
    year = "2022", url = "https://aclanthology.org/2022.naacl-main.353", pages = "4797--4809",
}

@inproceedings{liu-etal-2023-simple,
    title = "Simple Hardware-Efficient {PCFG}s with Independent Left and Right Productions",
    author = "Liu, Wei and Yang, Songlin and Kim, Yoon and Tu, Kewei",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    year = "2023", url = "https://aclanthology.org/2023.findings-emnlp.113", pages = "1662--1669",
}
```

## SemInfo

- Upstream: <https://github.com/junjiechen-chris/Improving-Unsupervised-Constituency-Parsing-via-Maximizing-Semantic-Information>
- License: none stated upstream (no `LICENSE` file in the source repository).
- Paper: Chen, He, Miyao, and Bollegala, "Improving Unsupervised Constituency Parsing via
  Maximizing Semantic Information", ICLR 2025 (Spotlight); OpenReview forum id
  [`qyU5s4fzLg`](https://openreview.net/forum?id=qyU5s4fzLg).

The SemInfo training objective and the paraphrase preprocessing pipeline are adapted from
this repository. The upstream repository is itself derived from TN-PCFG (see above).

```bibtex
@inproceedings{chen-etal-2025-improving,
    title = "Improving Unsupervised Constituency Parsing via Maximizing Semantic Information",
    author = "Chen, Junjie and He, Xiangheng and Miyao, Yusuke and Bollegala, Danushka",
    booktitle = "The Thirteenth International Conference on Learning Representations",
    year = "2025", url = "https://openreview.net/forum?id=qyU5s4fzLg",
}
```

## fastNLP (vendored)

- Upstream: <https://github.com/fastnlp/fastNLP>, version `0.5.6`.
- License: Apache License 2.0 — full text at [`fastNLP/LICENSE`](fastNLP/LICENSE).

`fastNLP/` is a vendored copy carrying local behavioral patches. Those modifications are
documented in [`fastNLP/LOCAL_PATCHES.md`](fastNLP/LOCAL_PATCHES.md), which serves as the
change statement required by Apache-2.0 §4(b).

## Supar

- Upstream: <https://github.com/yzhangcs/parser>
- License: MIT.

The upstream TN-PCFG code base used Supar as a code template; this fork inherits that
lineage. It is also acknowledged in the README.
