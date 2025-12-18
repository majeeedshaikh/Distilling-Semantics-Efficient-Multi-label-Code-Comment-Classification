[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/majeeedshaikh/Distilling-Semantics-Efficient-Multi-label-Code-Comment-Classification/blob/main/Code_Comment_FinalNotebook.ipynb)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](Paper_Finalv.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-NLBSE'26-orange.svg)](https://huggingface.co/datasets/NLBSE/nlbse26-code-comment-classification)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# Distilling Semantics: Efficient Multi-label Code Comment Classification (NLBSE'26)

This repository contains the **official, fully reproducible implementation** of our submission to the **NLBSE'26 Tool Competition** for multi-label code comment classification.

Our approach uses **semantic-aware knowledge distillation** to compress a CodeBERT teacher into a lightweight MiniLM student, achieving strong macro-F1 while substantially reducing runtime and GFLOPs under the official evaluation protocol.

📄 **Paper:** `Paper_Finalv.pdf` (included in this repository)

🏆 **Final submission score:** **0.74** (rounded to 2 decimal places)

---

## Repository Contents

```
.
├── Code_Comment_FinalNotebook.ipynb    # Main end-to-end training + evaluation notebook
├── Paper_Finalv.pdf                     # Final paper
├── Semantic-Boosters/
│   ├── java_semantic_boosters.csv       # Teacher-only semantic booster (Java)
│   ├── python_semantic_boosters.csv     # Teacher-only semantic booster (Python)
│   └── pharo_semantic_boosters.csv      # Teacher-only semantic booster (Pharo)
└── README.md
```

---

## Method Summary

- **Task:** Multi-label code comment classification
- **Dataset:** Official NLBSE'26 dataset (HuggingFace)
- **Languages:** Java, Python, Pharo
- **Teacher model:** `microsoft/codebert-base`
- **Student model:** `sentence-transformers/all-MiniLM-L6-v2`

### Key Ideas

- Teacher is fine-tuned on official data **plus small semantic booster CSVs**
- Student is trained **only on official NLBSE data**
- Knowledge is transferred via:
  - per-sample **standardized logit distillation**
  - **MSE-based KD loss**
  - mild **semantic label weighting**
- Thresholds are tuned on a validation split to handle label imbalance

⚠️ **Important:** Semantic booster CSV files are **only used during teacher fine-tuning** and are **never included in student hard-label training**.

---

## Execution Environment

This code is designed to be run **entirely on Google Colab**.

- No local environment setup is required
- No manual dependency installation is required
- All imports rely on libraries **preinstalled in Colab**

**Recommended runtime:**
- **GPU** → T4 / V100 / A100
- Mixed precision (FP16) is enabled automatically

---

## How to Run (Reproducibility Instructions)

### Step 1: Open Google Colab

Create a new Colab notebook and upload the following files:

- `Code_Comment_FinalNotebook.ipynb`
- `Semantic-Boosters/java_semantic_boosters.csv`
- `Semantic-Boosters/python_semantic_boosters.csv`
- `Semantic-Boosters/pharo_semantic_boosters.csv`

Ensure all files are uploaded to the **Colab root directory (`/content`)**.

The script expects the booster files at:

```python
/content/java_semantic_boosters.csv
/content/python_semantic_boosters.csv
/content/pharo_semantic_boosters.csv
```

**Note:** After uploading, you may need to move the CSV files from `Semantic-Boosters/` to `/content/` or update the paths in the notebook.

### Step 2: Open the Notebook

Open `Code_Comment_FinalNotebook.ipynb` in Colab.

Select:
```
Runtime → Change runtime type → GPU
```

### Step 3: Run All Cells (Top to Bottom)

Run the notebook **sequentially**.

The notebook performs the following automatically:

1. Loads the official NLBSE dataset from HuggingFace
2. Splits training data into train/validation
3. Fine-tunes a **CodeBERT teacher** (with semantic boosters)
4. Distills knowledge into a **MiniLM student**
5. Tunes decision thresholds on validation data
6. Evaluates on the official test sets
7. Measures runtime and GFLOPs using PyTorch Profiler
8. Computes macro F1 and submission score

No manual intervention is required.

---

## Reported Results (Reproducible)

Running the notebook end-to-end reproduces the following:

| Metric           | Value        |
| ---------------- | ------------ |
| Macro F1         | **0.6689**   |
| Avg runtime      | **0.7955 s** |
| Avg GFLOPs       | **769.97**   |
| Submission score | **0.74**     |

Minor variation (±0.002 F1) may occur depending on GPU type.

---

## Runtime and GFLOPs Measurement

- Runtime and FLOPs are measured using **PyTorch Profiler**
- Measurements are averaged over multiple inference runs (`N_RUNS = 10`)
- Matches the **official NLBSE baseline evaluation protocol**

---

## Determinism and Reproducibility Notes

- Random seed is fixed (`SEED = 42`)
- Validation split is deterministic
- Threshold tuning is deterministic given fixed logits
- No external APIs or services are used

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{asim2026distilling,
  title={Distilling Semantics: Efficient Multi-label Code Comment Classification},
  author={Majeed, Muhammad and Abdul Asim, Ahmed Bin},
  booktitle={Proceedings of the NLBSE'26 Tool Competition},
  year={2026}
}
```

---

## Contact

For questions or clarifications:

- **Muhammad Abdul Majeed** — [i221216@nu.edu.pk](mailto:i221216@nu.edu.pk)
- **Ahmed Bin Asim** — [i220949@nu.edu.pk](mailto:i220949@nu.edu.pk)

