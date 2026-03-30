import copy, json, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from dataset import HatefulMemesDataset
from models import TextOnlyModel, ImageOnlyModel, ConcatFusionModel, GatedFusionModel

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            img = batch['img_features'].to(device)
            txt = batch['txt_features'].to(device)
            logits = model(img, txt)
            probs = torch.sigmoid(logits).cpu()
            all_preds.extend(probs.tolist())
            all_labels.extend(batch['label'].tolist())
    auroc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    return auroc, acc


def train_model(model, train_loader, dev_loader, device,
                epochs=30, lr=1e-3, patience=5, name="model"):
    model = model.to(device)

    # pos_weight for class imbalance
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['label'].tolist())
    num_pos = sum(all_labels)
    num_neg = len(all_labels) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"{name} | params: {n_params:,} | pos_weight: {pos_weight.item():.2f}")
    print(f"{'='*60}")

    best_auroc, best_state = 0.0, None
    no_improve = 0
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss, n_batch = 0, 0

        for batch in train_loader:
            img = batch['img_features'].to(device)
            txt = batch['txt_features'].to(device)
            labels = batch['label'].to(device)

            logits = model(img, txt)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batch += 1

        avg_loss = total_loss / n_batch
        auroc, acc = evaluate(model, dev_loader, device)

        history.append({'epoch': epoch+1, 'train_loss': avg_loss,
                        'dev_auroc': auroc, 'dev_accuracy': acc})

        marker = '  *best*' if auroc > best_auroc else ''
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
              f"AUROC: {auroc:.4f} | Acc: {acc:.4f}{marker}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    final_auroc, final_acc = evaluate(model, dev_loader, device)
    print(f"Best {name}: AUROC={final_auroc:.4f}, Acc={final_acc:.4f}")
    return model, final_auroc, final_acc, history


def main():
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    data = torch.load("features/clip_features.pt", weights_only=False)
    train_ds = HatefulMemesDataset(data['train'])
    dev_ds = HatefulMemesDataset(data['dev'])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False)
    print(f"Train: {len(train_ds)}, Dev: {len(dev_ds)}")

    os.makedirs("results", exist_ok=True)
    results, histories = {}, {}

    experiments = [
        ("text_only",                "Model 1: Text Only",                TextOnlyModel()),
        ("image_only",               "Model 2: Image Only",               ImageOnlyModel()),
        ("concat",                   "Model 3: Concat Fusion",            ConcatFusionModel()),
        ("scalar_gate",              "Model 4: Scalar Gate",              GatedFusionModel(use_interaction=False, vector_gate=False)),
        ("scalar_gate_interaction",  "Model 5: Scalar Gate + Interaction", GatedFusionModel(use_interaction=True, vector_gate=False)),
        ("vector_gate_interaction",  "Model 6: Vector Gate + Interaction", GatedFusionModel(use_interaction=True, vector_gate=True)),
    ]

    for key, name, model in experiments:
        set_seed(SEED)
        model, auroc, acc, hist = train_model(model, train_loader, dev_loader, device, name=name)
        results[key] = {'auroc': auroc, 'accuracy': acc}
        histories[key] = hist
        torch.save(model.state_dict(), f"results/{key}.pt")

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(f"{'Model':<40} {'AUROC':>10} {'Acc':>10}")
    print("-" * 65)
    for key, name, _ in experiments:
        r = results[key]
        print(f"{name:<40} {r['auroc']:>10.4f} {r['accuracy']:>10.4f}")

    with open("results/results.json", "w") as f: json.dump(results, f, indent=2)
    with open("results/histories.json", "w") as f: json.dump(histories, f, indent=2)
    print("\nSaved results and model weights to results/")


if __name__ == "__main__":
    main()
