# Learning Notes — Concepts Explained During the Build

These are the key concepts explained step-by-step while building the Gated Multimodal Fusion project.

---

## 1. What is CLIP?

CLIP (Contrastive Language-Image Pretraining) by OpenAI maps images and text into the **same 512-dimensional vector space** such that matching image-text pairs are close together.

**How it was trained (simplified):**
```
Given 400M (image, caption) pairs from the internet:
1. Encode image → 512-d vector
2. Encode caption → 512-d vector
3. Push matching pairs CLOSER in vector space
4. Push non-matching pairs APART
```

After training, CLIP gives you two encoders:
- **Vision encoder** (a ViT): any image → 512-d vector
- **Text encoder** (a Transformer): any text → 512-d vector

We use CLIP as the backbone — it already "knows" what images and text mean. We build a small model ON TOP of these features that learns how to combine them for hate detection.

Model used: `openai/clip-vit-base-patch32`

---

## 2. JSONL Format

JSONL = JSON Lines. One JSON object per line. The Hateful Memes dataset format:

```json
{"id": 42953, "img": "img/42953.png", "label": 0, "text": "when you mix bleach and ammonia"}
```

Four fields:
- `id` — unique identifier
- `img` — relative path to the image file
- `label` — 0 (not hateful) or 1 (hateful)
- `text` — the OCR-extracted text from the meme

---

## 3. Batch Size

You have 8,500 training memes. Processing them one at a time is slow and noisy. Processing all at once doesn't fit in memory. Solution: **batches**.

`batch_size = 128` means process 128 memes together in one forward pass.

Benefits:
- **Fast**: parallel computation on GPU/MPS
- **Stable**: average loss over 128 samples is less noisy than 1 sample
- **Memory-friendly**: 128 fits easily in RAM

When batch_size = 128, tensor shapes become:
```python
h_text.shape   = [128, 512]    # 128 memes, each with 512-d text features
h_image.shape  = [128, 512]    # 128 memes, each with 512-d image features
labels.shape   = [128]         # 128 labels (0 or 1)
```

---

## 4. Tensor Dimensions (dim)

Tensors have axes. `dim` tells PyTorch WHICH axis to operate on.

For a 2D tensor with shape `[batch_size, features]`:

```
         dim=1 (features) →
dim=0      ┌─────────────────────┐
(batch)    │ 0.2  0.8  0.1  0.5 │  ← meme 0's features
  ↓        │ 0.7  0.3  0.9  0.2 │  ← meme 1's features
           │ 0.4  0.6  0.5  0.8 │  ← meme 2's features
           └─────────────────────┘
```

- `dim=0` → operate down the column (across memes)
- `dim=1` → operate across the row (across features)
- `dim=-1` → last dimension = same as `dim=1` for 2D tensors. Safer because it works regardless of how many dimensions the tensor has.

**Concatenation examples:**
```python
a = torch.tensor([[1.0, 2.0]])   # shape [1, 2]
b = torch.tensor([[3.0, 4.0]])   # shape [1, 2]

torch.cat([a, b], dim=1)  # [1, 4] → rows wider  (combining features)
torch.cat([a, b], dim=0)  # [2, 2] → columns taller (combining samples)
```

**Normalization:**
```python
x.norm(dim=-1, keepdim=True)  # compute norm of each row independently
x / x.norm(dim=-1, keepdim=True)  # normalize each row to unit length
```

---

## 5. nn.Linear (Linear Layer)

A matrix multiplication + bias:

```
output = input @ W + b

where:
  input  shape: [batch, in_features]
  W      shape: [in_features, out_features]    ← LEARNABLE weights
  b      shape: [out_features]                  ← LEARNABLE bias
  output shape: [batch, out_features]
```

Example: `nn.Linear(1024, 512)` transforms 1024 numbers into 512 numbers.
Think of it as: "Take 1024-d information and compress it into the 512 most useful signals."

---

## 6. Activation Functions

Without activation functions, stacking Linear layers is pointless:
```
Linear(1024→512) followed by Linear(512→1) = one big Linear(1024→1)
```

Activations add **non-linearity** — the ability to learn complex patterns, not just straight lines.

### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
ReLU(-3) = 0
ReLU(2) = 2
```
Kills negative values. Simple and fast. Used in classifier heads.

### Tanh
```
Tanh(x) → squashes to (-1, +1)
Tanh(-100) = -1.0
Tanh(0) = 0.0
Tanh(100) = 1.0
```
Preserves sign of negative values. Used in gate network because CLIP features can be negative — ReLU would destroy half the signal.

### Sigmoid
```
Sigmoid(x) → squashes to (0, 1)
Sigmoid(-100) = 0.0
Sigmoid(0) = 0.5
Sigmoid(100) = 1.0
```
Used as the final gate activation because we need a 0-1 value:
- 0 = trust image fully
- 1 = trust text fully
- 0.5 = equal mix

---

## 7. The Gate Network — Layer by Layer

```python
gate_network = nn.Sequential(
    nn.Linear(1024, 512),   # "Find 512 useful patterns from both modalities"
    nn.Tanh(),               # "Normalize to (-1,1), keep negative signals"
    nn.Linear(512, 1),       # "Compress 512 patterns into one decision"
    nn.Sigmoid()             # "Convert to a 0-1 gate value"
)
```

Full flow:
```
[h_text; h_image] (1024-d)
    ↓
Linear(1024→512)     → find patterns
    ↓
Tanh                 → normalize, preserve negatives
    ↓
Linear(512→1)        → one decision number
    ↓
Sigmoid              → convert to 0-1
    ↓
g = 0.73             → "73% text, 27% image for THIS meme"
```

---

## 8. The Interaction Term

```python
interaction = h_text * h_image    # element-wise product, NOT matrix multiply
```

Each dimension independently:
```
h_text  = [0.8,  -0.3,  0.5,  0.1]
h_image = [0.7,   0.6, -0.4,  0.9]
product = [0.56, -0.18, -0.20, 0.09]
```

What each value means:
- **High positive product**: both modalities agree (both features high or both low)
- **Negative product**: modalities disagree (one high, one low)
- **Near-zero product**: at least one modality has weak signal

**Why it matters for hate detection:**
Imagine dimension 47 = "person's face" and dimension 200 = "insulting language"

```
Non-hateful: face(0.9) * insult(0.1) = 0.09  (low — no dangerous combo)
Hateful:     face(0.9) * insult(0.8) = 0.72  (HIGH — dangerous combo!)
```

The gate with interaction can directly see dangerous modality combinations. Without it, the gate sees features separately and has to learn to multiply them internally (harder).

Gate input with interaction:
```python
gate_input = torch.cat([h_text, h_image, h_text * h_image], dim=1)  # [batch, 1536]
```

---

## 9. MLP (Multi-Layer Perceptron)

Stacked Linear layers with activations:
```python
mlp = nn.Sequential(
    nn.Linear(512, 256),    # layer 1
    nn.ReLU(),               # activation
    nn.Dropout(0.3),         # regularization
    nn.Linear(256, 1)        # output layer
)
```

**Why multiple layers?**
- Single Linear: can only learn straight-line decision boundaries
- Multiple layers with activations: can learn CURVES and complex patterns

---

## 10. Dropout

During training, randomly set 30% of neurons to zero each batch:
```
Before dropout: [0.5, 0.8, 0.2, 0.7, 0.1]
After dropout:  [0.5, 0.0, 0.2, 0.0, 0.1]  (30% randomly zeroed)
```

**Why?** Prevents overfitting. Forces the network to not rely on any single neuron. Like studying with random chapters removed — you learn overall concepts, not memorize specific pages.

During evaluation, dropout is turned OFF (all neurons active).

---

## 11. Feature Normalization

CLIP features are normalized to unit length before use:
```python
h_image = h_image / h_image.norm(dim=-1, keepdim=True)
h_text = h_text / h_text.norm(dim=-1, keepdim=True)
```

**Why?** CLIP was trained with normalized features. Without normalization, features have varying magnitudes — a feature with magnitude 100 would dominate one with magnitude 0.1, regardless of its actual importance.

After normalization, every feature vector has length 1.0, so all samples are on equal footing.

---

## 12. BCEWithLogitsLoss

Binary Cross Entropy loss for classification.

**Why `BCEWithLogitsLoss` instead of `BCELoss`?**
- `BCEWithLogitsLoss` applies sigmoid internally and is numerically stable
- Your model outputs a raw logit (no sigmoid at the end)
- Sigmoid is only applied during inference for getting probabilities

```python
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.85]))
# pos_weight handles class imbalance (35% hateful, 65% not)
# pos_weight = num_negative / num_positive ≈ 1.85
```

---

## 13. Scalar Gate vs Vector Gate

**Scalar gate:** one number per sample
```python
g = 0.73  # same weight for ALL 512 dimensions
h_fused = 0.73 * h_text + 0.27 * h_image
```

**Vector gate:** 512 numbers per sample
```python
g = [0.9, 0.2, 0.5, ...]  # DIFFERENT weight per dimension
h_fused[0] = 0.9 * h_text[0] + 0.1 * h_image[0]   # dim 0: trust text
h_fused[1] = 0.2 * h_text[1] + 0.8 * h_image[1]   # dim 1: trust image
```

Vector gate is more expressive — can learn "for color/style dimensions, trust image; for semantic dimensions, trust text."

---

## 14. Differential Learning Rates (for fine-tuning)

When fine-tuning pretrained CLIP layers alongside new layers:

```python
param_groups = [
    {"params": clip_last_2_layers.parameters(), "lr": 1e-6},   # slow — preserve pretrained knowledge
    {"params": gate.parameters(),                "lr": 1e-3},   # fast — learn from scratch
    {"params": classifier.parameters(),          "lr": 1e-3},   # fast — learn from scratch
]
```

**Why?** CLIP's layers already have good weights from 400M image-text pairs. A high LR would destroy them (catastrophic forgetting). New layers are randomly initialized and need a high LR to learn quickly.

---

## 15. Gate Collapse — A Real Training Bug

During our first training, gate values collapsed to ~0.5 for all samples (std = 0.029). The gate was not making per-sample decisions — it learned to sit at the midpoint and let the classifier do all the work.

**Why it happened:**
CLIP maps images and text into the SAME 512-d vector space (that's what contrastive pre-training does). When you compute `g * h_text + (1-g) * h_image`, you're interpolating between two vectors in the same space. The gate has no reason to prefer one over the other — they carry similar information by design.

**What we tried:**
1. **Entropy regularization** — penalized gate values near 0.5. Result: made AUROC worse because it distorted the classification loss.
2. **Simpler gate** (single linear layer) — followed Arevalo et al. (2017) GMU paper. Result: didn't fix collapse, gates still near 0.5.
3. **Modality-specific projections** — project text and image into SEPARATE spaces before gating. The gate decides between genuinely different representations. Result: gates improved and gated models beat concat fusion.

**Key insight:** For gating to work, the modalities being fused must be in different representation spaces. Gating between vectors in the same space is essentially learning a weighted average, which a linear layer can already do.

---

## 16. Modality-Specific Projections

The fix for gate collapse. Instead of gating raw CLIP features:

```
OLD (doesn't work well):
  h_fused = g * h_text_clip + (1-g) * h_image_clip
  (both in same 512-d CLIP space)

NEW (works):
  h_t = text_projection(h_text_clip)     → 256-d text-specific space
  h_i = image_projection(h_image_clip)   → 256-d image-specific space
  h_fused = g * h_t + (1-g) * h_i
  (different spaces — gate has reason to differentiate)
```

The gate input still uses original CLIP features (shared space is fine for DECIDING), but the features being fused are in separate spaces.

---

## 17. Fine-Tuning vs Feature Extraction

**Feature extraction (Models 1-6):** CLIP is frozen. Extract features once, save to disk, train small heads. Fast (seconds per epoch) but CLIP features aren't adapted to memes.

**Fine-tuning (Model 7):** Unfreeze last 2 CLIP layers. Gradients flow back into CLIP, adapting its features to meme-domain data. Slow (minutes per epoch) but features become meme-specific.

Fine-tuning requires:
- Raw images each batch (can't cache features since CLIP changes during training)
- Differential learning rates (CLIP: 1e-6, new layers: 1e-3)
- Weight decay (AdamW) to prevent fine-tuned weights from growing too large
- Smaller batch size (16 vs 128) due to memory

---

## 18. CLIP Processor — What It Does

`CLIPProcessor` converts raw inputs into tensors:

**For images:**
```
Original image (any size, 0-255 pixels)
  → Resize to 224×224
  → Normalize to 0.0-1.0
  → Standardize with CLIP's mean/std
  → Output: [batch, 3, 224, 224] tensor
```

**For text:**
```
"its their character not their color"
  → Tokenize: [<START>, its, their, character, not, their, color, <END>]
  → Convert to IDs: [49406, 902, 911, 4009, 783, 911, 3140, 49407]
  → Output: input_ids [batch, seq_len], attention_mask [batch, seq_len]
```

**Attention mask:** 1 = real token, 0 = padding. When batching texts of different lengths, shorter texts get padded. The mask tells CLIP to ignore padding positions.

---

## 19. Early Stopping

Stop training when validation performance stops improving:

```
Epoch 10: AUROC = 0.70  ← best so far, save model
Epoch 11: AUROC = 0.69  ← worse, patience counter = 1
Epoch 12: AUROC = 0.68  ← worse, patience counter = 2
...
Epoch 15: AUROC = 0.67  ← patience = 5, STOP
```

Use the saved model from epoch 10, not epoch 15. Without early stopping, the model overfits — it memorizes training data and performs worse on unseen data.

`model.state_dict()` saves a snapshot of all weights. `copy.deepcopy()` ensures it's a true copy (not a reference that changes with further training).

---

## 20. Reproducibility — Setting Seeds

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

Neural network training involves randomness (weight initialization, data shuffling, dropout). Setting seeds makes results repeatable — same seed = same random numbers = same results.

We reset the seed before each model experiment so all models start from the same random state. This ensures differences in results are due to architecture, not random luck.

---

*More concepts will be added as we progress through the build.*
