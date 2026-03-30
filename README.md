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
# setup
cd multimodal-gated-fusion
uv init  # if not already done
uv add torch torchvision transformers datasets scikit-learn matplotlib pillow tqdm jupyter
```

## 2. Get the Dataset

Download from Kaggle: https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset

```bash
# option A: kaggle CLI
uv add kaggle
kaggle datasets download -d parthplc/facebook-hateful-meme-dataset
unzip facebook-hateful-meme-dataset.zip -d data/

# option B: manual download from kaggle website, unzip into data/
```

After this you should have:
```
data/
├── img/          # ~10,000 png files
├── train.jsonl   # 8500 samples
├── dev.jsonl     # 500 samples
└── test.jsonl    # 1000 samples (no labels)
```

# feature extraction (run the notebook)
uv run jupyter notebook  # run notebooks/feature_extraction.ipynb

# train everything
uv run python train.py
uv run python train_finetuned.py

# analyze everything
uv run python evaluate.py
uv run python analyze_finetuned.py

# check results
cat results/results.json
ls analysis/
```

## References

- Kiela et al. "The Hateful Memes Challenge" (NeurIPS 2020)
- Arevalo et al. "Gated Multimodal Units for Information Fusion" (2017)
- Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021)
- I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts (ICML 2025)
