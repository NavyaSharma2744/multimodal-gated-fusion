import json, os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from PIL import Image
from dataset import HatefulMemesDataset
from models import GatedFusionModel

DATA_DIR = "data"
RESULTS_DIR = "results"
ANALYSIS_DIR = "analysis"


def extract_gate_values(model, dataset, device):
    model = model.to(device).eval()
    all_gates, all_preds, all_labels = [], [], []
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            img = batch['img_features'].to(device)
            txt = batch['txt_features'].to(device)
            gates = model.get_gate_values(img, txt).cpu()
            logits = model(img, txt)
            probs = torch.sigmoid(logits).cpu()
            all_gates.append(gates)
            all_preds.extend(probs.tolist())
            all_labels.extend(batch['label'].tolist())
    return torch.cat(all_gates, dim=0), all_preds, all_labels


def get_gate_mean(gate_values):
    if gate_values.dim() == 2 and gate_values.shape[1] > 1:
        return gate_values.mean(dim=1).numpy()
    return gate_values.squeeze().numpy()


def plot_gate_dist(values, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Gate value")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.axvline(x=0.5, color='red', linestyle='--', label='g=0.5')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_gate_vs_correctness(values, preds, labels, path):
    binary = [1 if p > 0.5 else 0 for p in preds]
    correct = [values[i] for i in range(len(labels)) if binary[i] == labels[i]]
    incorrect = [values[i] for i in range(len(labels)) if binary[i] != labels[i]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(correct, bins=30, alpha=0.6, label=f"Correct (n={len(correct)})", edgecolor='black')
    ax.hist(incorrect, bins=30, alpha=0.6, label=f"Incorrect (n={len(incorrect)})", edgecolor='black')
    ax.set_xlabel("Gate value")
    ax.set_ylabel("Count")
    ax.set_title("Gate Value: Correct vs Incorrect")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    print(f"  Correct   — mean: {np.mean(correct):.3f}, std: {np.std(correct):.3f}")
    print(f"  Incorrect — mean: {np.mean(incorrect):.3f}, std: {np.std(incorrect):.3f}")


def plot_gate_comparison(v1, v2, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(v1, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_title("Model 4: Scalar Gate (no interaction)")
    axes[0].set_xlabel("Gate value")
    axes[0].axvline(x=0.5, color='red', linestyle='--')
    axes[1].hist(v2, bins=50, edgecolor='black', alpha=0.7, color='darkorange')
    axes[1].set_title("Model 6: Vector Gate + Interaction")
    axes[1].set_xlabel("Gate value")
    axes[1].axvline(x=0.5, color='red', linestyle='--')
    plt.suptitle("Effect of Interaction Term on Gate Behavior")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_examples(values, preds, labels, dev_raw, save_dir):
    binary = [1 if p > 0.5 else 0 for p in preds]
    high_t = np.percentile(values, 90)
    low_t = np.percentile(values, 10)
    median = np.median(values)

    print(f"  Gate range: [{values.min():.3f}, {values.max():.3f}]")
    print(f"  Thresholds — text: >{high_t:.3f}, image: <{low_t:.3f}")

    cats = {'text_dominant': [], 'image_dominant': [], 'confused': []}
    for i in range(len(values)):
        correct = binary[i] == labels[i]
        g = values[i]
        if g >= high_t and correct and len(cats['text_dominant']) < 3:
            cats['text_dominant'].append(i)
        elif g <= low_t and correct and len(cats['image_dominant']) < 3:
            cats['image_dominant'].append(i)
        elif abs(g - median) < 0.005 and not correct and len(cats['confused']) < 3:
            cats['confused'].append(i)

    for cat, idxs in cats.items():
        if not idxs:
            print(f"  No examples for {cat}")
            continue
        n = len(idxs)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
        if n == 1: axes = [axes]
        for ax, idx in zip(axes, idxs):
            s = dev_raw[idx]
            img_path = os.path.join(DATA_DIR, s['img'])
            if os.path.exists(img_path):
                ax.imshow(Image.open(img_path))
            ax.set_title(f"Gate: {values[idx]:.2f}\n"
                         f"Label: {'hateful' if labels[idx]==1 else 'not hateful'}\n"
                         f"Pred: {'hateful' if binary[idx]==1 else 'not hateful'}", fontsize=10)
            ax.set_xlabel(f"Text: {s['text'][:60]}...", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle(f"Examples: {cat.replace('_',' ').title()}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"examples_{cat}.png"), dpi=150)
        plt.close()
        print(f"  Saved: {save_dir}/examples_{cat}.png")


def plot_training_curves(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, hist in histories.items():
        epochs = [h['epoch'] for h in hist]
        axes[0].plot(epochs, [h['train_loss'] for h in hist], label=name, marker='.')
        axes[1].plot(epochs, [h['dev_auroc'] for h in hist], label=name, marker='.')
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Training Loss")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUROC"); axes[1].set_title("Dev AUROC")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")

    features = torch.load("features/clip_features.pt", weights_only=False)
    dev_ds = HatefulMemesDataset(features['dev'])
    with open(os.path.join(DATA_DIR, "dev.jsonl")) as f:
        dev_raw = [json.loads(line) for line in f]

    # training curves
    print("--- Training Curves ---")
    histories = json.load(open(os.path.join(RESULTS_DIR, "histories.json")))
    plot_training_curves(histories, os.path.join(ANALYSIS_DIR, "training_curves.png"))

    # load gated models
    print("\n--- Gate Analysis ---")
    configs = [
        ("scalar_gate",             False, False),
        ("scalar_gate_interaction",  True,  False),
        ("vector_gate_interaction",  True,  True),
    ]
    gates_dict = {}
    for key, interaction, vector in configs:
        m = GatedFusionModel(use_interaction=interaction, vector_gate=vector)
        m.load_state_dict(torch.load(f"{RESULTS_DIR}/{key}.pt", weights_only=False))
        g, p, l = extract_gate_values(m, dev_ds, device)
        gates_dict[key] = (g, p, l)

    # gate distributions
    print("\n--- Gate Distributions ---")
    for key in gates_dict:
        g, _, _ = gates_dict[key]
        plot_gate_dist(get_gate_mean(g), key, os.path.join(ANALYSIS_DIR, f"gate_dist_{key}.png"))

    # gate vs correctness (best frozen model)
    print("\n--- Gate vs Correctness ---")
    g6, p6, l6 = gates_dict["vector_gate_interaction"]
    v6 = get_gate_mean(g6)
    plot_gate_vs_correctness(v6, p6, l6, os.path.join(ANALYSIS_DIR, "gate_vs_correctness.png"))

    # comparison
    print("\n--- Interaction Effect ---")
    g4, _, _ = gates_dict["scalar_gate"]
    plot_gate_comparison(get_gate_mean(g4), v6, os.path.join(ANALYSIS_DIR, "gate_comparison.png"))

    # examples
    print("\n--- Sample Examples ---")
    save_examples(v6, p6, l6, dev_raw, ANALYSIS_DIR)

    # classification report
    print("\n--- Classification Report (Model 6) ---")
    print(classification_report(l6, [1 if p > 0.5 else 0 for p in p6],
                                target_names=['Not Hateful', 'Hateful']))


if __name__ == "__main__":
    main()
