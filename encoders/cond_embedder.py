import torch.nn as nn

class ImageEncoderEmbedder(nn.Module):
    """
    This is a sample image encoder to show how we can use conditioned image (OD center) into our architecture. 
    """
    def __init__(self, emb_dim=32):
        super(ImageEncoderEmbedder, self).__init__()
        self.emb_dim = emb_dim

        # OD image branch
        self.od_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, emb_dim, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        # Final fusion: concatenate OD image features and text features, then project.
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, od_image, text_condition_indices):
        
        # OD image encoding.
        od_feat = self.od_encoder(od_image).squeeze(-1).squeeze(-1)  # shape: [B, emb_dim]
       
        context = self.proj(od_feat)                                 # shape: [B, emb_dim]

        return context
