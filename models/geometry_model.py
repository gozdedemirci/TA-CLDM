# --- Add these imports if needed ---
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# --- ADD: Siamese Encoder (Helper for GeomHead) ---
class SiameseEncoder(nn.Module):
    def __init__(self, embedding_dim=512, input_channels=3): # Adjust input_channels (e.g., 1 for grayscale)
        super().__init__()
        # Example using ResNet18 stem
        resnet = models.resnet18(pretrained=True)
        # Modify first layer if input channels != 3
        if input_channels != 3:
             self.original_conv1_weights = resnet.conv1.weight.clone() # Keep original weights if needed
             resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
             # Optional: Initialize weights reasonably (e.g., average original RGB weights)
             # resnet.conv1.weight.data = self.original_conv1_weights.mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1)

        self.features = nn.Sequential(*list(resnet.children())[:-1]) # Remove final FC layer
        self.projector = nn.Linear(resnet.fc.in_features, embedding_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.projector(x))
        return x

# --- ADD: The Geometry Head Network ---
class GeomHead(nn.Module):
    def __init__(self, embedding_dim=512, hidden_dim=256, input_channels=3):
        super().__init__()
        self.encoder = SiameseEncoder(embedding_dim, input_channels)

        # Comparison head
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim) # Takes concatenated features
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 3) # Output: [θ (radians), dx, dy]  

    def forward(self, img_cond, img_gen):
        features_in = self.encoder(img_cond)
        features_gen = self.encoder(img_gen)
        combined_features = torch.cat((features_in, features_gen), dim=1)
        x = self.relu(self.fc1(combined_features))
        logits = self.fc2(x) # T_pred
        return logits