import torch
import torch.nn as nn


class TextOnlyModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, img_features, txt_features):
        return self.classifier(txt_features).squeeze(-1)


class ImageOnlyModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, img_features, txt_features):
        return self.classifier(img_features).squeeze(-1)


class ConcatFusionModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, img_features, txt_features):
        combined = torch.cat([txt_features, img_features], dim=1)
        return self.classifier(combined).squeeze(-1)


class GatedFusionModel(nn.Module):
    # gated fusion with modality-specific projections
    # CLIP puts both modalities in same space so we project them
    # into separate spaces first, then gate decides how to combine
    def __init__(self, input_dim=512, proj_dim=256, hidden_dim=128,
                 dropout=0.3, use_interaction=False, vector_gate=False):
        super().__init__()
        self.use_interaction = use_interaction
        self.vector_gate = vector_gate

        # project each modality into its own space
        self.text_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))

        # gate uses original CLIP features for deciding
        gate_input_dim = input_dim * 2
        if use_interaction:
            gate_input_dim = input_dim * 3
        gate_output_dim = proj_dim if vector_gate else 1
        self.gate_linear = nn.Linear(gate_input_dim, gate_output_dim)

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, img_features, txt_features):
        h_t = self.text_proj(txt_features)
        h_i = self.image_proj(img_features)

        if self.use_interaction:
            interaction = txt_features * img_features
            gate_input = torch.cat([txt_features, img_features, interaction], dim=1)
        else:
            gate_input = torch.cat([txt_features, img_features], dim=1)

        g = torch.sigmoid(self.gate_linear(gate_input))
        self._last_gate = g
        h_fused = g * h_t + (1 - g) * h_i
        return self.classifier(h_fused).squeeze(-1)

    def gate_entropy(self):
        g = self._last_gate
        eps = 1e-7
        entropy = -(g * torch.log(g + eps) + (1 - g) * torch.log(1 - g + eps))
        return entropy.mean()

    def get_gate_values(self, img_features, txt_features):
        with torch.no_grad():
            if self.use_interaction:
                interaction = txt_features * img_features
                gate_input = torch.cat([txt_features, img_features, interaction], dim=1)
            else:
                gate_input = torch.cat([txt_features, img_features], dim=1)
            return torch.sigmoid(self.gate_linear(gate_input))
