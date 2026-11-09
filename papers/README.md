# Papers

Source papers used by this project, numbered as in `specs/00_MASTER_IMPLEMENTATION_INDEX.md`
§ Core research sources. Adoption levels come from `references/provenance.yaml`.
PDFs are git-ignored (publisher copyright); this README is tracked. Downloaded 2026-11-09.

| # | Paper | Used by | Adoption | Local file |
|---|---|---|---|---|
| 1 | Hosenie et al. 2021, *MeerCRAB*, Exp. Astron. 51, DOI 10.1007/s10686-021-09757-1 | M02 data/crop/split, M04 CNN | adapted | `01_Hosenie2021_MeerCRAB_arXiv2104.13950.pdf` (arXiv preprint) |
| 2 | Stoppa et al., MeerLICHT/ATLAS/Pan-STARRS release, Zenodo 14714279 | dataset | — | dataset, not a paper; data files are in the project root |
| 3 | Bonnet-Guerrini et al. 2026, *Interpretable Human-Label-Free Deep Learning for Real-Bogus Classification with UQ*, arXiv 2607.05393 | M04 architecture, M08 UQ metrics | adapted | `03_BonnetGuerrini2026_HumanLabelFree_UQ_arXiv2607.05393.pdf` |
| 4 | Taylor et al. 2018, *Adaptive Deep Learning Model Selection on Embedded Systems*, LCTES, DOI 10.1145/3211332.3211336 | M07 tier selection | inspired | `04_Taylor2018_AdaptiveModelSelection_arXiv1805.04252.pdf` (arXiv preprint, titled "Adaptive Selection of Deep Learning Models on Embedded Systems"; same authors) |
| 5 | Lee et al. 2026, *Quilt: Peak-memory-aware partitioning and scheduling for multi-tenant DNN inference*, JSA 173, DOI 10.1016/j.sysarc.2026.103696 | M07/M09 admission | inspired | **missing** — closed access (Elsevier), no preprint found |
| 6 | Jiang et al. 2025, *Convergo: Multi-SLO-Aware Scheduling for Heterogeneous AI Accelerators on Edge Devices*, IEEE EDGE, DOI 10.1109/EDGE67623.2025.00022 | M09/M12 framing | inspired | **missing** — closed access (IEEE), no preprint found |
| 7 | Sun et al. 2025, *PRAS: Adaptive scheduling of online inference pipelines at the edge*, JSA 169, DOI 10.1016/j.sysarc.2025.103597 | M06 workloads, M12 framing | inspired | **missing** — SSRN preprint 10.2139/ssrn.5260642 exists but blocks automated download |
| 8 | Nguyen et al. 2025, *OCTOPINF*, arXiv 2502.01277 | M09/M12 framing | inspired | `08_Nguyen2025_OCTOPINF_arXiv2502.01277.pdf` |
| 9 | Gal & Ghahramani 2016, *Dropout as a Bayesian Approximation*, ICML, arXiv 1506.02142 | M08 MC dropout | adapted | `09_Gal2016_MCDropout_arXiv1506.02142.pdf` |
| 10 | Guo et al. 2017, *On Calibration of Modern Neural Networks*, ICML, arXiv 1706.04599 | M05 temperature scaling | adapted | `10_Guo2017_Calibration_arXiv1706.04599.pdf` |
| 11 | El-Yaniv & Wiener 2010, *On the Foundations of Noise-free Selective Classification*, JMLR 11 | M08/M10 ambiguous zone | adapted | `11_ElYaniv2010_SelectiveClassification_JMLR.pdf` |

## Getting the three missing papers
Download through the university library (IEEE Xplore / ScienceDirect institutional access)
or SSRN in a browser, and save them here as:
`05_Lee2026_Quilt_JSA.pdf`, `06_Jiang2025_Convergo_EDGE.pdf`, `07_Sun2025_PRAS_JSA.pdf`.
Until then those three relationships rest on the earlier verification recorded in `references/provenance.yaml`;
all three are only *inspired*, so no mechanism is reproduced from them.
