# Application Answer Drafts

IMPORTANT: These are drafts based on your real experience. Rewrite them
in YOUR voice before submitting. The professor explicitly said "articulate
that by yourself, using AI only for grammar edits."

---

## Technical Challenge #1 (Max 300 words)

**Describe a technically challenging task you have completed.**

I built a multimodal hate speech detection system that learns to dynamically
weight visual and textual features per-sample using a gated fusion mechanism.

The task: given a meme (image + overlaid text), classify it as hateful or not.
This is fundamentally multimodal — the same text can be hateful or benign
depending on the image context, and vice versa. I used the Facebook Hateful
Memes benchmark (8,500 training memes), which was specifically designed so
neither modality alone is sufficient.

I implemented 7 model variants as an ablation study:
- Unimodal baselines (text-only: 0.635 AUROC, image-only: 0.655)
- Concatenation fusion (0.706)
- Gated fusion with scalar and vector gates (0.709-0.712)
- Fine-tuned CLIP with gated fusion (0.727)

The core contribution is a learned gating mechanism that decides how much to
trust text vs image for each individual meme. The gate uses cross-modal
interaction features (element-wise product of text and image embeddings) to
detect when modalities agree or disagree.

The main technical challenge was gate collapse — the gate converged to 0.5
for all samples, making no per-sample decisions. I diagnosed this as a
consequence of CLIP mapping both modalities into the same embedding space:
the gate couldn't differentiate between two vectors in an identical space.
I fixed this by adding modality-specific projection layers that map text and
image features into separate representation spaces before gating. This
allowed gated models to surpass concatenation fusion.

Fine-tuning CLIP's last two layers with differential learning rates
(1e-6 for CLIP, 1e-3 for new layers) gave the best result (0.727 AUROC),
outperforming VisualBERT (0.71) with a simpler architecture.

**Proof of Work:** [Your GitHub repo URL]

---

## Describe a specific bug or error you encountered during training

During training of the gated fusion model, I observed that the gate values
collapsed to approximately 0.5 for all samples (standard deviation < 0.03).
The model was achieving reasonable AUROC (0.70+) but the gate mechanism was
effectively bypassed — the model learned to use the classifier head for all
discrimination while the gate produced a fixed 50/50 mix.

I verified this by extracting gate values for all 500 dev samples and
plotting the distribution — it was a narrow spike centered at 0.5 with
no text-dominant or image-dominant samples.

---

## What did you try first to fix it, and were you right?

My first attempt was entropy regularization — adding a penalty term to the
loss that punishes gate values near 0.5. This was wrong. The entropy penalty
distorted the classification loss, causing AUROC to drop from 0.712 to 0.687.
The model spent its capacity trying to push gates to extreme values instead
of learning to classify correctly.

My second attempt was simplifying the gate architecture from a two-layer MLP
to a single linear layer (following Arevalo et al., 2017). This also didn't
fix the collapse.

The fix that worked was adding modality-specific projection layers. I
realized the root cause: CLIP maps both modalities into the same 512-d space
by design (contrastive pre-training). Gating between two vectors in an
identical space is just a weighted average — the gate has no reason to
prefer one over the other. By projecting each modality into its own 256-d
space before fusion, the gate had genuinely different representations to
choose between. After this change, gated models consistently outperformed
concatenation.

---

## Your validation loss was decreasing but your validation metric (e.g. AUROC) was not improving. What would be the first three things you check?

1. **Class imbalance exploitation.** The dataset is 36% hateful, 64% not
   hateful. The model can reduce loss by confidently predicting the majority
   class for borderline samples. Loss decreases because the model is more
   confident, but AUROC doesn't improve because it's not learning to
   distinguish the classes better — it's just exploiting the base rate.
   I would check the per-class prediction distribution.

2. **Overfitting to easy samples.** The model might be reducing loss on
   training samples it already classifies correctly (making confident
   predictions more confident) while not improving on hard samples
   (cross-modal confounders where the label depends on text-image
   interaction). I would check if the loss reduction comes uniformly
   across samples or is concentrated on already-correct predictions.

3. **Threshold mismatch.** AUROC is threshold-free, but if I were tracking
   accuracy instead, a decreasing loss with flat accuracy could mean the
   model is improving its probability calibration in a range that doesn't
   cross the 0.5 decision threshold. I would plot the predicted probability
   distribution and check if it's shifting but not crossing the threshold
   for misclassified samples.

---

## What dataset did you use? How many samples?

Facebook Hateful Memes dataset (Kiela et al., NeurIPS 2020). 10,000 memes
total, each consisting of an image with overlaid text and OCR-extracted text.
Binary labels: hateful (1) or not hateful (0). The dataset was specifically
designed with "benign confounders" — swapping the image or text in a hateful
meme produces a non-hateful version, ensuring neither modality alone is
sufficient for classification.

---

## What was your train/validation/test split? Why did you choose that split?

Train: 8,500 / Dev: 500 / Test: 1,000

I used the official dataset splits provided by the Hateful Memes Challenge
organizers. This is important for two reasons: (1) it enables direct
comparison with published baselines (VisualBERT, ViLBERT, etc.) that used
the same splits, and (2) the organizers carefully constructed the splits to
ensure benign confounders are distributed appropriately — a random split
might accidentally separate a meme from its confounder, making evaluation
artificially easier.

---

## What was your best validation metric value, and what was your baseline?

Best validation AUROC: 0.727 (fine-tuned CLIP with vector gated fusion)

Baselines:
- Random: 0.500 AUROC
- Mean predictor (always predict majority class): 0.500 AUROC
- Text-only (my model): 0.635 AUROC
- Image-only (my model): 0.655 AUROC
- Concatenation fusion (my model): 0.706 AUROC
- Published VisualBERT baseline: ~0.710 AUROC

My best model outperforms VisualBERT while using a simpler architecture
(CLIP + learned gate vs. a full cross-modal Transformer).

---

## What is one thing you would redo if you had more time, and why?

I would replace the gating mechanism with cross-attention between image
patch tokens and text tokens, rather than gating at the feature level.

My current approach operates on CLIP's [CLS] token — a single 512-d vector
summarizing the entire image or text. This discards spatial information
(where in the image is the relevant content?) and token-level information
(which specific words interact with the image?).

Cross-attention would let the model learn: "the word 'ugly' attends to the
person's face in the image → hateful" vs "the word 'ugly' attends to a
cartoon character → not hateful." This token-level interaction is what makes
models like VisualBERT and FLAVA stronger — they fuse modalities at the
token level, not the feature level.

Additionally, I would explore the relationship between gate behavior and
the benign confounder structure of the dataset: do confounders (same image,
different text) produce systematically different gate values?

---

## Paper Review: I2MoE (ICML 2025)

**Paper:** I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts

### One thing I liked (100 words)

The paper's decomposition of multimodal interactions into distinct types
(unimodal dominance, synergy, redundancy) and assigning specialized experts
to each is elegant. Rather than forcing a single fusion mechanism to handle
all interaction patterns, the Mixture-of-Experts framework lets each expert
specialize. The routing mechanism reveals which interaction type dominates
each sample — this interpretability is valuable. In my own work on gated
fusion, I observed that a single gate struggles to capture diverse
interaction patterns. I2MoE's multi-expert approach is a principled solution
to this limitation.

### One thing I disliked (100 words)

The MoE framework introduces substantial architectural complexity — multiple
expert networks, a routing mechanism with load balancing, and interaction
type classification — for gains that are sometimes modest over simpler fusion
methods. On some benchmarks, the improvement over concatenation or attention
fusion is within a few percentage points. For binary classification tasks
like hateful meme detection, the overhead of maintaining and training
multiple experts may not justify the marginal gains. The paper would benefit
from a clearer analysis of when MoE-based fusion is necessary versus when
simpler approaches (like learned gating) achieve comparable results with
far fewer parameters.

### One thing I would improve (100 words)

I would investigate adaptive expert count — letting the model learn how
many experts are needed per dataset rather than fixing it as a hyperparameter.
Some datasets may have only two dominant interaction types (text-dominant
and image-dominant), while others may require finer decomposition. A
mechanism that starts with a single expert and progressively splits when
routing entropy is high would reduce unnecessary complexity on simpler tasks.
Additionally, I would test I2MoE against a simple learned vector gate (as
in my project) to establish when the full MoE framework becomes necessary
versus when lightweight gating captures sufficient interaction patterns.

---

## Technical Challenge #2 and #3

You need two more technical challenges. These should be from your OTHER
projects or experiences. They do NOT need to be multimodal or related to
this project. Think about:

- Any ML/DL project where you trained a model
- A systems/backend project with technical depth
- A competitive programming achievement
- Any project where you hit a hard bug and solved it

Write these from your own experience. I cannot help draft these since
they must be about work you actually did.
