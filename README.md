# CPGC-Project

This repository provides the official implementation of **CPGC**, a few-shot prototype-based segmentation framework designed for aeroengine borescope damage inspection and other industrial defect scenarios under extremely limited supervision.

CPGC addresses two key challenges commonly observed in few-shot segmentation: **evidence collapse** in prototype matching and **geometric instability** in predicted masks. To this end, the framework integrates two core components:

* **Complementary Evidence Prototype Matching (CEPM)**, which explicitly models multiple complementary prototypes to capture diverse but semantically consistent discriminative cues from limited support samples.
* **Global Geometric Evolution (GGE)**, which introduces differentiable geometric refinement to enforce boundary continuity and region closure, improving structural reliability of segmentation results.

By jointly modeling complementary evidence and global geometric priors in an end-to-end manner, CPGC enables robust and complete segmentation of unseen damage categories without requiring additional training or fine-tuning.

---

## Repository Status

⚠️ **Note:** The corresponding paper is currently under review and has not yet been formally published.
To comply with publication policies, **this repository currently releases only the source code for the two core modules:**

* `CEPM/`: Complementary Evidence Prototype Matching
* `GGE/`: Global Geometric Evolution

The remaining components, including the full training pipeline, dataset preprocessing, and evaluation scripts, will be released after the paper is officially accepted.

---

## Citation

If you find this work useful, please consider citing our paper once it becomes available.
Citation information will be updated upon publication.

---

If you want, I can also help you:

* Add a **method overview diagram caption** for the README
* Write a **minimal usage example** (pseudo-code level, safe before publication)
* Prepare a **“Coming Soon” roadmap section** that looks professional and reviewer-safe
