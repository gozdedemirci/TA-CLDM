import torch
import torch.nn as nn
import numpy as np
from skimage.morphology import skeletonize
from transformers import CLIPTokenizer, CLIPTextModel

class PositionalEncoder(nn.Module):
    """Fourier feature positional encoding for vessel coordinates."""
    def __init__(self, L=6):
        super().__init__()
        self.L = L  # Frequency bands
        self.scale = 2 ** torch.arange(L).float()  # [2^0, 2^1, ..., 2^{L-1}]
        
    def forward(self, coords):
        """
        Args:
            coords: (N, 2) tensor of (x, y) coordinates (normalized to [-1, 1])
        Returns:
            pe: (N, 2*2L) positional encoding
        """
        x = coords[:, 0].unsqueeze(-1)  # (N, 1)
        y = coords[:, 1].unsqueeze(-1)  # (N, 1)

        self.scale = self.scale.to(coords.device)
        
        # Compute Fourier features for x and y
        pe_x = torch.cat([torch.sin(self.scale * np.pi * x), 
                        torch.cos(self.scale * np.pi * x)], dim=-1)  # (N, 2L)
        pe_y = torch.cat([torch.sin(self.scale * np.pi * y), 
                        torch.cos(self.scale * np.pi * y)], dim=-1)  # (N, 2L)
        
        return torch.cat([pe_x, pe_y], dim=-1)  # (N, 4L)
    
class VesselPEEncoder(nn.Module):
    """Encodes vessel mask into positional encoding."""
    def __init__(self, L=6):
        super().__init__()
        self.pos_encoder = PositionalEncoder(L=L)
        self.L = L
        
    def forward(self, vessel_mask):
        """
        Args:
            vessel_mask: (B, 1, H, W) soft mask [0, 1]
        Returns:
            pe_global: (B, 4L) aggregated positional encoding
        """
        B = vessel_mask.shape[0]
        pe_global = torch.zeros(B, 4 * self.L).to(vessel_mask.device)
        
        for i in range(B):
            # Threshold and skeletonize
            binary_mask = (vessel_mask[i, 0] > 0.5).float().cpu().numpy()
            skeleton = skeletonize(binary_mask)  # (H, W)
            coords = np.argwhere(skeleton)  # (N, 2)
            
            if len(coords) == 0:
                continue  # No vessels detected
                
            # Normalize coordinates to [-1, 1]
            H, W = binary_mask.shape
            coords = coords.astype(np.float32)
            coords[:, 0] = (coords[:, 0] / (H-1)) * 2 - 1  # y-coordinate
            coords[:, 1] = (coords[:, 1] / (W-1)) * 2 - 1  # x-coordinate
            
            # Compute PE and aggregate
            coords_tensor = torch.tensor(coords).to(vessel_mask.device)  # (N, 2)
            self.pos_encoder = self.pos_encoder.to(vessel_mask.device)
            pe = self.pos_encoder(coords_tensor)  # (N, 4L)
            pe_global[i] = pe.mean(dim=0)  # Global average
            
        return pe_global  # (B, 4L)
    
class PosEncEmbedder(nn.Module):
    """Combines text embeddings and vessel PE."""
    def __init__(self, text_dim=512, pe_dim=24, emb_dim=256):
        super().__init__()
        self.emb_dim = emb_dim
        self.text_embedder = CLIPTextConditioner(output_dim=text_dim)

        # OD image branch (same as before)
        self.od_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, emb_dim//2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.vessel_pe_encoder = VesselPEEncoder(L=pe_dim//4)  # pe_dim=4L=24 when L=6

        self.proj = nn.Sequential(
            nn.Linear(text_dim*3, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)  # Project back to emb_dim
        )
        
    def forward(self, od_image, text_condition, vessel_masks):
        # OD image encoding.
        od_feat = self.od_encoder(od_image).squeeze(-1).squeeze(-1)  # shape: [B, emb_dim//2]# OD image encoding.
        # Text condition encoding.
        text_feat = self.text_embedder(text_condition)       # shape: [B, emb_dim//2]
        # Vessel Positional Encoding.
        self.vessel_pe_encoder = self.vessel_pe_encoder.to(od_image.device)
        vessel_pe = self.vessel_pe_encoder(vessel_masks)  # shape: [B, pe_dim]

        # Concatenate and project.
        combined = torch.cat([od_feat, text_feat, vessel_pe], dim=1)             # shape: [B, emb_dim]
        context = self.proj(combined)                                 # shape: [B, emb_dim]

        return context  # (B, text_dim)
    
class CLIPTextConditioner(nn.Module):
    def __init__(self, output_dim=32, pretrained_model="openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
        # Project CLIP’s text embedding to the desired dimension.
        self.proj = nn.Linear(self.text_encoder.config.hidden_size, output_dim)
        
    def forward(self, text_list):
        # text_list: list of strings, length B
        inputs = self.tokenizer(text_list, padding=True, return_tensors="pt")
        inputs = inputs.to(self.text_encoder.device)
        outputs = self.text_encoder(**inputs)
        # Use the last hidden state of the [EOS] token (or mean pooling)
        text_embedding = outputs.last_hidden_state[:, -1, :]  # [B, hidden_size]
        return self.proj(text_embedding)  # [B, output_dim]

class LabelImageTextEmbedder(nn.Module):
    def __init__(self, emb_dim=32, text_hidden_dim=16): # text_hidden_dim=emb_dim//2
        super(LabelImageTextEmbedder, self).__init__()
        self.emb_dim = emb_dim
        # Text embedder to process the condition string (e.g. "left eye, nasal")
        self.text_embedder = CLIPTextConditioner(output_dim=text_hidden_dim)
        
        # OD image branch (same as before)
        self.od_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, emb_dim//2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        # Final fusion: concatenate OD image features and text features, then project.
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, od_image, text_condition_indices):
        # OD image encoding.
        od_feat = self.od_encoder(od_image).squeeze(-1).squeeze(-1)  # shape: [B, emb_dim//2]
        # Text condition encoding.
        text_feat = self.text_embedder(text_condition_indices)       # shape: [B, emb_dim//2]
        # Concatenate and project.
        combined = torch.cat([od_feat, text_feat], dim=1)             # shape: [B, emb_dim]
        context = self.proj(combined)                                 # shape: [B, emb_dim]
        return context

if __name__ == "__main__":
    od_img = torch.randn(1, 3, 256, 256)
    label = ['left eye, nasal view']
    vessel_mask = torch.randn(1, 1, 256, 256)

    text_embedder = PosEncEmbedder(emb_dim=1024, text_dim=512, pe_dim=512)
    text_cond = text_embedder(od_img, label, vessel_mask)
    print(text_cond.shape)

    text_embedder = LabelImageTextEmbedder(emb_dim=1024, text_hidden_dim=512)
    text_cond = text_embedder(od_img,label)
    print(text_cond.shape)
    