# Learned Modality Gating for Multimodal Hate Speech Detection

Investigating whether a learned per-sample gating mechanism that dynamically weights visual and textual features can outperform static fusion strategies for multimodal classification.

## Research Question

When fusing image and text for hate detection in memes, does a model that learns **how** to combine modalities per-sample outperform fixed fusion? And what does the model learn about when each modality matters?

## Dataset

[Facebook Hateful Memes](https://arxiv.org/abs/2005.04790) (Kiela et al., NeurIPS 2020) -- 10,000 memes with binary labels. Designed with benign confounders: neither modality alone is sufficient.

## Approach

CLIP (ViT-B/32) encodes images and text into 512-d vectors. We test 7 fusion strategies:

| # | Model | AUROC |
|---|-------|-------|
| 1 | Text Only | 0.635 |
| 2 | Image Only | 0.655 |
| 3 | Concat Fusion | 0.706 |
| 4 | Scalar Gate | 0.709 |
| 5 | Scalar Gate + Interaction | 0.709 |
| 6 | Vector Gate + Interaction | 0.712 |
| 7 | **Fine-tuned CLIP + Vector Gate** | **0.727** |

## Key Findings

**Gate collapse in shared embedding spaces.** CLIP maps both modalities into the same vector space. A gate interpolating between vectors in the same space has no reason to differentiate -- it collapses to 0.5 for all samples. We fix this with modality-specific projection layers.

**Vector gating outperforms scalar gating.** Per-dimension gates (256 independent decisions) outperform a single per-sample scalar, suggesting modality importance varies across feature dimensions.

**Fine-tuning adapts representations to meme domain.** CLIP's generic web-trained features improve with domain adaptation, yielding the best AUROC (0.727), outperforming VisualBERT (~0.71).

## Architecture

```
Image  -> CLIP ViT (frozen/fine-tuned) -> h_img (512) -> image_proj -> h_i (256)
Text   -> CLIP Text (frozen/fine-tuned) -> h_txt (512) -> text_proj  -> h_t (256)

Gate input: [h_txt; h_img; h_txt * h_img] (1536-d)
Gate:       g = sigmoid(W * gate_input)    (256-d vector)
Fusion:     h_fused = g * h_t + (1-g) * h_i
Output:     classifier(h_fused) -> hateful/not
```

## Project Structure

```
├── data/                     # Hateful Memes dataset (not tracked)
├── features/                 # Cached CLIP embeddings
├── results/                  # Trained model weights + metrics
├── analysis/                 # Gate distribution plots, examples
├── notebooks/
│   ├── explore.ipynb         # Data exploration
│   └── feature_extraction.ipynb
├── dataset.py                # PyTorch Dataset
├── models.py                 # TextOnly, ImageOnly, Concat, GatedFusion
├── train.py                  # Train frozen-CLIP models (1-6)
├── train_finetuned.py        # Train fine-tuned CLIP model (7)
├── evaluate.py               # Metrics + gate analysis (frozen models)
├── analyze_finetuned.py      # Gate analysis (fine-tuned model)
└── RESEARCH_PLAN.md          # Hypotheses + experiment design
```

## Reproducing

```bash
# Install dependencies
uv sync

# Extract CLIP features (run notebook or)
uv run python train.py           # trains models 1-6
uv run python train_finetuned.py # trains model 7
uv run python evaluate.py        # generates analysis plots
uv run python analyze_finetuned.py
```

## References

- Kiela et al. "The Hateful Memes Challenge" (NeurIPS 2020)
- Arevalo et al. "Gated Multimodal Units for Information Fusion" (2017)
- Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021)
- I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts (ICML 2025)
