# How to Run This Project (Step by Step)

## 0. Prerequisites

- Python 3.10+
- Mac (MPS) or Linux (CUDA) or CPU
- ~4GB disk space for dataset
- ~600MB for CLIP model (downloads once, cached)

## 1. Setup

```bash
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

Verify:
```bash
ls data/img/ | wc -l    # should be ~10000
head -1 data/train.jsonl # should show {id, img, label, text}
```

## 3. Explore the Data

```bash
uv run jupyter notebook
```

Open `notebooks/explore.ipynb` and run all cells. This shows you:
- What the memes look like
- Class balance (36% hateful, 64% not)
- Why this task needs both modalities

## 4. Extract CLIP Features

Open `notebooks/feature_extraction.ipynb` and run all cells.

This runs CLIP on all 10K memes ONCE and saves 512-d vectors to disk.
Takes ~5-10 min on MPS, ~15 min on CPU.

After this you should have:
```
features/clip_features.pt   # ~40MB file
```

Verify:
```python
import torch
d = torch.load("features/clip_features.pt", weights_only=False)
print(d['train']['img_features'].shape)  # [8500, 512]
```

## 5. Train Models 1-6 (Frozen CLIP)

```bash
uv run python train.py
```

This trains 6 models sequentially on cached features:
1. Text Only
2. Image Only
3. Concat Fusion
4. Scalar Gate
5. Scalar Gate + Interaction
6. Vector Gate + Interaction

Takes ~5-10 min total (features are cached, models are small).

Output:
```
results/
├── results.json          # final AUROC and accuracy for all models
├── histories.json        # epoch-by-epoch training logs
├── text_only.pt          # saved model weights
├── image_only.pt
├── concat.pt
├── scalar_gate.pt
├── scalar_gate_interaction.pt
└── vector_gate_interaction.pt
```

## 6. Train Model 7 (Fine-tuned CLIP)

```bash
uv run python train_finetuned.py
```

This fine-tunes last 2 layers of CLIP alongside the gated fusion head.
Uses raw images (not cached features) so it's slower.

Takes ~20-30 min on MPS.

Output:
```
results/finetuned_gated.pt    # fine-tuned model weights
results/results.json          # updated with model 7 results
results/histories.json        # updated with model 7 history
```

## 7. Analysis (Frozen Models)

```bash
uv run python evaluate.py
```

Generates:
```
analysis/
├── training_curves.png         # loss + AUROC over epochs for all models
├── gate_dist_scalar_gate.png   # gate value histogram (model 4)
├── gate_dist_scalar_gate_interaction.png   # model 5
├── gate_dist_vector_gate_interaction.png   # model 6
├── gate_vs_correctness.png     # correct vs incorrect gate values
├── gate_comparison.png         # with vs without interaction term
├── examples_text_dominant.png  # memes where gate relied on text
├── examples_image_dominant.png # memes where gate relied on image
└── examples_confused.png       # memes where gate couldn't decide
```

## 8. Analysis (Fine-tuned Model)

```bash
uv run python analyze_finetuned.py
```

Generates:
```
analysis/
├── finetuned_gate_analysis.png     # gate distribution + correctness
├── finetuned_text_dominant.png
├── finetuned_image_dominant.png
└── finetuned_confused.png
```

## Full Pipeline (TL;DR)

```bash
# setup
uv sync

# data (manual download from kaggle into data/ folder)

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

## Expected Results

| Model | AUROC | Accuracy |
|-------|-------|----------|
| Text Only | ~0.635 | ~0.600 |
| Image Only | ~0.655 | ~0.592 |
| Concat Fusion | ~0.706 | ~0.636 |
| Scalar Gate | ~0.709 | ~0.628 |
| Scalar Gate + Interaction | ~0.709 | ~0.616 |
| Vector Gate + Interaction | ~0.712 | ~0.640 |
| Fine-tuned + Gate | ~0.727 | ~0.656 |

Numbers may vary slightly due to random initialization.

## Project Files

```
multimodal-gated-fusion/
├── data/                       # dataset (not in git)
├── features/                   # cached CLIP embeddings
├── results/                    # model weights + metrics
├── analysis/                   # plots
├── notebooks/
│   ├── explore.ipynb           # step 3: look at the data
│   └── feature_extraction.ipynb # step 4: extract CLIP features
├── dataset.py                  # pytorch dataset class
├── models.py                   # all model architectures
├── train.py                    # step 5: train models 1-6
├── train_finetuned.py          # step 6: train model 7
├── evaluate.py                 # step 7: analyze frozen models
├── analyze_finetuned.py        # step 8: analyze fine-tuned model
├── RESEARCH_PLAN.md            # hypotheses and experiment design
├── HOW_TO_RUN.md               # this file
└── README.md                   # project overview
```
