import copy, json, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm

SEED = 42
DATA_DIR = "data"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)


class HatefulMemesRawDataset(Dataset):
    def __init__(self, jsonl_path, data_dir):
        with open(jsonl_path) as f:
            self.samples = [json.loads(line) for line in f]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(os.path.join(self.data_dir, s['img'])).convert('RGB')
        return img, s['text'], s.get('label', -1)


def collate_fn(batch, processor):
    images, texts, labels = zip(*batch)
    inputs = processor(text=list(texts), images=list(images),
                       return_tensors="pt", padding=True, truncation=True)
    return inputs, torch.tensor(labels, dtype=torch.float32)


class FineTunedGatedFusion(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32",
                 num_unfrozen=2, proj_dim=256, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)

        # freeze everything first
        for param in self.clip.parameters():
            param.requires_grad = False

        # unfreeze last N layers of both encoders
        for layer in self.clip.vision_model.encoder.layers[-num_unfrozen:]:
            for param in layer.parameters(): param.requires_grad = True
        for layer in self.clip.text_model.encoder.layers[-num_unfrozen:]:
            for param in layer.parameters(): param.requires_grad = True

        # unfreeze projection heads too
        for param in self.clip.visual_projection.parameters(): param.requires_grad = True
        for param in self.clip.text_projection.parameters(): param.requires_grad = True

        input_dim = 512
        self.text_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.gate_linear = nn.Linear(input_dim * 3, proj_dim)  # with interaction
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, pixel_values, input_ids, attention_mask):
        img_feat = self.clip.get_image_features(pixel_values=pixel_values)
        txt_feat = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        if not isinstance(img_feat, torch.Tensor): img_feat = img_feat.pooler_output
        if not isinstance(txt_feat, torch.Tensor): txt_feat = txt_feat.pooler_output

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        h_t = self.text_proj(txt_feat)
        h_i = self.image_proj(img_feat)

        interaction = txt_feat * img_feat
        gate_input = torch.cat([txt_feat, img_feat, interaction], dim=1)
        g = torch.sigmoid(self.gate_linear(gate_input))
        self._last_gate = g

        h_fused = g * h_t + (1 - g) * h_i
        return self.classifier(h_fused).squeeze(-1)

    def get_param_groups(self, clip_lr=1e-6, head_lr=1e-3):
        clip_params, head_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            if name.startswith('clip.'): clip_params.append(param)
            else: head_params.append(param)
        return [{"params": clip_params, "lr": clip_lr},
                {"params": head_params, "lr": head_lr}]


def evaluate_finetuned(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            pv = inputs['pixel_values'].to(device)
            ids = inputs['input_ids'].to(device)
            mask = inputs['attention_mask'].to(device)
            logits = model(pv, ids, mask)
            all_preds.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.tolist())
    auroc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    return auroc, acc


def train_finetuned(model, train_loader, dev_loader, device,
                    clip_lr=1e-6, head_lr=1e-3, epochs=15, patience=5):
    model = model.to(device)

    num_pos = sum(1 for _, _, l in train_loader.dataset if l == 1)
    num_neg = len(train_loader.dataset) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.get_param_groups(clip_lr, head_lr), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    clip_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith('clip.'))
    print(f"\nModel 7: Fine-tuned CLIP + Vector Gate + Interaction")
    print(f"CLIP params: {clip_params:,} | Head params: {total_params-clip_params:,} | Total: {total_params:,}")
    print(f"CLIP lr: {clip_lr}, Head lr: {head_lr}, pos_weight: {pos_weight.item():.2f}")
    print("=" * 60)

    best_auroc, best_state, no_improve = 0.0, None, 0
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss, n_batch = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            pv = inputs['pixel_values'].to(device)
            ids = inputs['input_ids'].to(device)
            mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)

            loss = loss_fn(model(pv, ids, mask), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / n_batch
        auroc, acc = evaluate_finetuned(model, dev_loader, device)
        scheduler.step(auroc)

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
    final_auroc, final_acc = evaluate_finetuned(model, dev_loader, device)
    print(f"\nBest Model 7: AUROC={final_auroc:.4f}, Acc={final_acc:.4f}")
    return model, final_auroc, final_acc, history


def main():
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    train_ds = HatefulMemesRawDataset(os.path.join(DATA_DIR, "train.jsonl"), DATA_DIR)
    dev_ds = HatefulMemesRawDataset(os.path.join(DATA_DIR, "dev.jsonl"), DATA_DIR)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, processor), num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, processor), num_workers=0)
    print(f"Train: {len(train_ds)}, Dev: {len(dev_ds)}")

    model = FineTunedGatedFusion(num_unfrozen=2, dropout=0.4)
    model, auroc, acc, history = train_finetuned(model, train_loader, dev_loader, device)

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/finetuned_gated.pt")

    # append to existing results
    rpath = "results/results.json"
    results = json.load(open(rpath)) if os.path.exists(rpath) else {}
    results['finetuned_vector_gate_interaction'] = {'auroc': auroc, 'accuracy': acc}
    json.dump(results, open(rpath, "w"), indent=2)

    hpath = "results/histories.json"
    histories = json.load(open(hpath)) if os.path.exists(hpath) else {}
    histories['finetuned_vector_gate_interaction'] = history
    json.dump(histories, open(hpath, "w"), indent=2)

    print("\n" + "=" * 65)
    print("ALL RESULTS")
    print("=" * 65)
    for name, r in results.items():
        print(f"  {name:<45} AUROC={r['auroc']:.4f}  Acc={r['accuracy']:.4f}")


if __name__ == "__main__":
    main()
