# Learned Modality Gating for Multimodal Hate Speech Detection

## Research Question

When fusing image and text for multimodal classification, does a model that **learns how to combine modalities per-sample** outperform fixed fusion strategies? And **what does the model learn** about when each modality matters?

---

## Hypotheses

**H1:** Neither modality alone is sufficient for hate detection in memes — the task fundamentally requires multimodal understanding.
- Test: Text-only and Image-only baselines both underperform fusion models.

**H2:** Naive concatenation fails because it gives equal weight to both modalities regardless of the sample, adding noise from the uninformative modality.
- Test: Concat fusion only marginally improves over text-only (sometimes worse).

**H3:** A learned gate that sees cross-modal interaction (not just independent features) makes better routing decisions, because hate in memes emerges from the **relationship** between image and text, not from either alone.
- Test: Gate with `[h_t; h_i; h_t*h_i]` outperforms gate with just `[h_t; h_i]`.

**H4:** Domain-adapted (fine-tuned) CLIP representations improve fusion because CLIP was trained on natural photos + web captions, which are distributionally different from memes.
- Test: Fine-tuned CLIP + gate outperforms frozen CLIP + gate.

---

## Dataset

**Facebook Hateful Memes** (Kiela et al., NeurIPS 2020)
- ~10,000 meme images with overlaid text
- Binary classification: hateful (1) vs not hateful (0)
- Designed with benign confounders — neither modality alone is sufficient
- Source: Kaggle (parthplc/facebook-hateful-meme-dataset)
- Splits: train.jsonl, dev_seen.jsonl, dev_unseen.jsonl, test_seen.jsonl

---

## Model Variants (Ablation Study)

| # | Model | Gate Input | CLIP | Tests | Expected AUROC |
|---|-------|-----------|------|-------|---------------|
| 1 | Text-only MLP | — | Frozen | H1 baseline | ~65% |
| 2 | Image-only MLP | — | Frozen | H1 baseline | ~53% |
| 3 | Concat fusion MLP | — | Frozen | H2 | ~68% |
| 4 | Scalar gate | [h_t; h_i] | Frozen | H2 (gating helps) | ~71% |
| 5 | Scalar gate + interaction | [h_t; h_i; h_t*h_i] | Frozen | H3 | ~73% |
| 6 | Vector gate + interaction | [h_t; h_i; h_t*h_i] | Frozen | Per-dim vs per-sample | ~74% |
| 7 | Vector gate + interaction | [h_t; h_i; h_t*h_i] | Fine-tuned last 2 layers | H4 | ~76% |

Each row adds exactly ONE change from the previous. Clean ablation design.

---

## Architecture Details

### CLIP Backbone
- Model: openai/clip-vit-base-patch32
- Image encoder: Vision Transformer → 512-d vector
- Text encoder: Transformer → 512-d vector
- For Models 1-6: extract features once, freeze CLIP
- For Model 7: unfreeze last 2 layers of both encoders

### Gate Mechanism

**Scalar gate (Models 4-5):**
```
gate_input = [h_text; h_image] or [h_text; h_image; h_text * h_image]
g = sigmoid(MLP(gate_input))     # g is scalar in (0,1)
h_fused = g * h_text + (1-g) * h_image
```

**Vector gate (Models 6-7):**
```
gate_input = [h_text; h_image; h_text * h_image]
g = sigmoid(MLP(gate_input))     # g is 512-d vector in (0,1)^512
h_fused = g * h_text + (1-g) * h_image
```

### Classifier Head
```
h_fused (512-d) → Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 1)
```

### Loss
- BCEWithLogitsLoss (no sigmoid in model, numerically stable)
- pos_weight to handle class imbalance (~35% hateful)

### Training

Models 1-6 (frozen CLIP):
- Optimizer: Adam, lr=1e-3
- Batch size: 128
- Early stopping: patience 5 epochs
- Scheduler: ReduceLROnPlateau, patience 3

Model 7 (fine-tuned CLIP):
- Differential learning rates:
  - CLIP last 2 layers: lr=1e-6
  - Gate + Classifier: lr=1e-3
- Optimizer: AdamW, weight_decay=0.01
- Batch size: 32 (larger model, more memory)
- Early stopping: patience 5 epochs

### Regularization for Gate Collapse
If gate converges to all-0 or all-1, add entropy regularization:
```
gate_entropy = -(g * log(g) + (1-g) * log(1-g))
loss = bce_loss - lambda * gate_entropy.mean()
```

---

## Analysis Plan

### Analysis 1: Gate Value Distribution
- Histogram of gate values across test set
- Check: bimodal (good, model learned preferences) vs unimodal at 0.5 (bad, gate not useful)

### Analysis 2: Sample-Level Examples
- 2 memes with high gate (g > 0.8) — text-dominant, explain why
- 2 memes with low gate (g < 0.2) — image-dominant, explain why
- 2 memes with middle gate + wrong prediction — explain the difficulty

### Analysis 3: Gate vs Correctness
- Average gate value for correct vs incorrect predictions
- Hypothesis: wrong predictions cluster around g ~ 0.5 (model uncertain about which modality to trust)

### Analysis 4: Effect of Interaction Term
- Compare gate distributions between Model 4 (no interaction) and Model 5 (with interaction)
- Does the interaction term make the gate more decisive (more extreme values)?

---

## Evaluation Metrics

- **Primary:** AUROC (threshold-free, standard for this benchmark)
- **Secondary:** Accuracy (at optimal threshold)
- **Analysis:** Gate value statistics, per-sample gate visualization

### Published Baselines for Reference
- Human performance: 82.65% AUROC
- Text-only (BERT): 65.08% AUROC
- Image-only: ~52% AUROC
- VisualBERT: ~71% AUROC
- Competition SOTA: ~80-84% AUROC

---

## Paper Review

**Paper:** I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts (ICML 2025)

**Connection to project:** I2MoE uses multiple experts with learned routing for different modality interaction types. Our gated fusion is a simpler single-expert alternative where the gate serves as the router. The paper review will compare our lightweight approach against their MoE framework.

**Review structure:**
- Liked: Interpretable routing reveals interaction types per sample (validates that different samples need different fusion)
- Disliked: MoE adds complexity (multiple experts, load balancing) — is it worth it for binary classification?
- Improve: Test whether a simple learned gate captures comparable interaction patterns with less overhead

---

## Tech Stack

- Python 3.10+
- PyTorch (MPS backend for Apple Silicon)
- transformers (CLIP model)
- scikit-learn (metrics)
- matplotlib (visualizations)
- Jupyter notebooks (exploration + analysis)
- Python files (models + training)

---

## Project Structure

```
hateful-memes-gated-fusion/
├── data/
│   ├── img/                      # meme images from Kaggle
│   ├── train.jsonl
│   ├── dev_seen.jsonl
│   ├── dev_unseen.jsonl
│   └── test_seen.jsonl
├── features/                     # cached CLIP features
│   └── clip_features.pt
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_extract_features.ipynb
│   └── 03_analysis.ipynb
├── models.py                     # all model classes
├── train.py                      # training loop
├── evaluate.py                   # metrics + gate analysis
├── RESEARCH_PLAN.md              # this file
└── README.md
```

---

## Build Schedule

### Day 1: Foundations + Baselines

| Step | Task | Output | Time |
|------|------|--------|------|
| 1 | Explore dataset — load jsonl, display memes, check class balance | Data intuition | 1.5h |
| 2 | Extract CLIP features for all images and texts, save to disk | features.pt | 1.5h |
| 3 | Build PyTorch Dataset class for cached features | Reusable dataloader | 1h |
| 4 | Implement Models 1-3 (text-only, image-only, concat) | 3 model classes | 1.5h |
| 5 | Write training loop + evaluation function | train() and evaluate() | 2h |
| 6 | Train Models 1-3, record AUROC | Rows 1-3 of results table | 1.5h |
| 7 | Implement + train Model 4 (scalar gate, frozen) | Row 4 result | 1h |
| 8 | Verify gating improves over concat | Day 1 checkpoint | 1h |

### Day 2: Core Contribution + Analysis

| Step | Task | Output | Time |
|------|------|--------|------|
| 9 | Add interaction term, train Model 5 | Row 5 result | 1.5h |
| 10 | Upgrade to vector gate, train Model 6 | Row 6 result | 1.5h |
| 11 | Fine-tune CLIP last 2 layers, train Model 7 | Row 7 result | 2.5h |
| 12 | Gate value distribution plots | Histogram visualization | 1h |
| 13 | Sample-level analysis with meme examples | 6 annotated examples | 1.5h |
| 14 | Gate vs correctness + interaction term comparison | Key findings | 1h |
| 15 | Clean code, README, push to GitHub | Public repo | 1.5h |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Gate collapse (all-0 or all-1) | Entropy regularization on gate values |
| Fine-tuning overfits (small dataset) | Weight decay, early stopping, freeze all but last 2 layers |
| Interaction term doesn't help | Valid finding — report honestly |
| MPS backend issues | Fall back to CPU |
| Numbers don't match estimates | Trend matters, not exact numbers |

---

## Application Connection

### Technical Challenge Description (300 words)
Frame around: tested 4 hypotheses about multimodal fusion, built 7 model variants, discovered that learned gating with cross-modal interaction outperforms naive fusion, analyzed what the gate learned.

### Training Bug
Likely: differential learning rate issue during fine-tuning, or gate collapse, or class imbalance causing all-negative predictions.

### Validation Loss vs Metric Question
1. Class imbalance — loss decreases by confidently predicting majority class
2. Threshold mismatch — AUROC is threshold-free but accuracy depends on threshold choice
3. Overfitting to easy samples while failing on hard cross-modal confounders

### What Would You Redo
Explore attention-based fusion (cross-attention between image and text tokens) instead of feature-level gating. Or test with a larger VLM backbone instead of CLIP.

### Paper Review
I2MoE (ICML 2025) — connected to gated fusion as described above.

---

## Research Grounding

- Gated Multimodal Units: Arevalo et al., ICLR 2017 Workshop
- Hateful Memes Challenge: Kiela et al., NeurIPS 2020
- CLIP: Radford et al., ICML 2021
- I2MoE: ICML 2025
- Deep Multimodal Data Fusion Survey: ACM Computing Surveys 2024
- Multimodal Representation Collapse: ICML 2025 Spotlight

## QUEST Lab Connection
Learned adaptive fusion principle — same philosophy as DNOD's multi-scale adaptive fusion (TMLR → ICLR 2026 J2C), applied to cross-modal instead of cross-scale setting.
