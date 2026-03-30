import json, os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from transformers import CLIPProcessor
from PIL import Image
from train_finetuned import HatefulMemesRawDataset, collate_fn, FineTunedGatedFusion

DATA_DIR = "data"
ANALYSIS_DIR = "analysis"


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # load model
    model = FineTunedGatedFusion(num_unfrozen=2, dropout=0.4)
    model.load_state_dict(torch.load("results/finetuned_gated.pt", weights_only=False))
    model = model.to(device).eval()

    # load data
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dev_ds = HatefulMemesRawDataset(os.path.join(DATA_DIR, "dev.jsonl"), DATA_DIR)
    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, processor), num_workers=0)
    with open(os.path.join(DATA_DIR, "dev.jsonl")) as f:
        dev_raw = [json.loads(line) for line in f]

    # extract gates
    print("Extracting gate values...")
    all_gates, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for inputs, labels in dev_loader:
            pv = inputs['pixel_values'].to(device)
            ids = inputs['input_ids'].to(device)
            mask = inputs['attention_mask'].to(device)
            logits = model(pv, ids, mask)
            all_gates.append(model._last_gate.cpu())
            all_preds.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.tolist())

    gates = torch.cat(all_gates, dim=0)
    values = gates.mean(dim=1).numpy() if gates.dim() == 2 and gates.shape[1] > 1 else gates.squeeze().numpy()

    # stats
    print(f"\nGate stats: mean={values.mean():.4f}, std={values.std():.4f}, "
          f"min={values.min():.4f}, max={values.max():.4f}, range={values.max()-values.min():.4f}")

    binary = [1 if p > 0.5 else 0 for p in all_preds]
    correct_g = [values[i] for i in range(len(all_labels)) if binary[i] == all_labels[i]]
    incorrect_g = [values[i] for i in range(len(all_labels)) if binary[i] != all_labels[i]]
    print(f"Correct:   mean={np.mean(correct_g):.4f}, std={np.std(correct_g):.4f}")
    print(f"Incorrect: mean={np.mean(incorrect_g):.4f}, std={np.std(incorrect_g):.4f}")
    print(f"(Frozen model had: mean=0.466, std=0.017)")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(values, bins=50, edgecolor='black', alpha=0.7, color='forestgreen')
    axes[0].set_title("Fine-tuned: Gate Distribution")
    axes[0].set_xlabel("Gate value"); axes[0].axvline(x=0.5, color='red', linestyle='--')
    axes[1].hist(correct_g, bins=30, alpha=0.6, label=f"Correct ({len(correct_g)})", edgecolor='black')
    axes[1].hist(incorrect_g, bins=30, alpha=0.6, label=f"Incorrect ({len(incorrect_g)})", edgecolor='black')
    axes[1].set_title("Fine-tuned: Gate vs Correctness")
    axes[1].set_xlabel("Gate value"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "finetuned_gate_analysis.png"), dpi=150)
    plt.close()
    print(f"\nSaved: analysis/finetuned_gate_analysis.png")

    # examples
    high_t = np.percentile(values, 90)
    low_t = np.percentile(values, 10)
    median = np.median(values)
    print(f"Thresholds — text: >{high_t:.3f}, image: <{low_t:.3f}")

    examples = {'text_dominant': [], 'image_dominant': [], 'confused': []}
    for i in range(len(values)):
        correct = binary[i] == all_labels[i]
        g = values[i]
        if g >= high_t and correct and len(examples['text_dominant']) < 3:
            examples['text_dominant'].append(i)
        elif g <= low_t and correct and len(examples['image_dominant']) < 3:
            examples['image_dominant'].append(i)
        elif abs(g - median) < 0.005 and not correct and len(examples['confused']) < 3:
            examples['confused'].append(i)

    for cat, idxs in examples.items():
        if not idxs: print(f"  No examples for {cat}"); continue
        n = len(idxs)
        fig, axs = plt.subplots(1, n, figsize=(5*n, 5))
        if n == 1: axs = [axs]
        for ax, idx in zip(axs, idxs):
            s = dev_raw[idx]
            if os.path.exists(os.path.join(DATA_DIR, s['img'])):
                ax.imshow(Image.open(os.path.join(DATA_DIR, s['img'])))
            ax.set_title(f"Gate: {values[idx]:.3f}\nLabel: {'hateful' if all_labels[idx]==1 else 'not hateful'}\n"
                         f"Pred: {'hateful' if binary[idx]==1 else 'not hateful'}", fontsize=10)
            ax.set_xlabel(f"{s['text'][:60]}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle(f"Fine-tuned: {cat.replace('_',' ').title()}")
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"finetuned_{cat}.png"), dpi=150)
        plt.close()
        print(f"  Saved: analysis/finetuned_{cat}.png")

    # report
    print(f"\n{classification_report(all_labels, binary, target_names=['Not Hateful', 'Hateful'])}")
    print(f"AUROC: {roc_auc_score(all_labels, all_preds):.4f}")


if __name__ == "__main__":
    main()
