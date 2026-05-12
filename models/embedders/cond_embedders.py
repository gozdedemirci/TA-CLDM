
import torch.nn as nn
import torch 
from monai.networks.layers.utils import get_act_layer
from transformers import CLIPTokenizer, CLIPTextModel

class LabelEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_classes, emb_dim)

        # self.embedding = nn.Embedding(num_classes, emb_dim//4)
        # self.emb_net = nn.Sequential(
        #     nn.Linear(1, emb_dim),
        #     get_act_layer(act_name),
        #     nn.Linear(emb_dim, emb_dim)
        # )

    def forward(self, condition):
        c = self.embedding(condition) #[B,] -> [B, C]
        # c = self.emb_net(c)
        # c = self.emb_net(condition[:,None].float())
        # c = (2*condition-1)[:, None].expand(-1, self.emb_dim).type(torch.float32)
        return c

class LabelImageEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_classes, emb_dim)

        self.od_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, emb_dim//2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.label_embed = nn.Embedding(num_classes, emb_dim//2)  # 2 classes: nasal/temporal

        self.proj = nn.Linear(emb_dim, emb_dim)  # Project combined features

    def forward(self, od_image, label):
        # Encode OD image
        if label is not None:
            od_feat = self.od_encoder(od_image).squeeze(-1).squeeze(-1)  # (B, 256)
            
            # Encode label
            lbl_feat = self.label_embed(label)  # (B, 256)
            
            # Combine features
            combined = torch.cat([od_feat, lbl_feat], dim=1)  # (B, 512)
            context = self.proj(combined)  # (B, 1, 512)
        else:
            od_feat = self.od_encoder(od_image).squeeze(-1).squeeze(-1)
            context = self.proj(torch.cat([od_feat, od_feat], dim=1))
        return context
    
class LabelImageTextEmbedder(nn.Module):
    def __init__(self, emb_dim=32, text_hidden_dim=16): # text_hidden_dim=emb_dim//2
        super(LabelImageTextEmbedder, self).__init__()
        self.emb_dim = emb_dim
        # Text embedder to process the condition string (e.g. "left eye, nasal")
        self.text_embedder = CLIPTextConditioner(output_dim=text_hidden_dim)
        
        # OD image branch (same as before)
        self.od_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), # rop -- 3, xray -- 1
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
    
import numpy as np
def drop_strings_numpy(strings, threshold=0.15):
    strings = np.array(strings, dtype=object)  # keep as object array
    rand_vals = np.random.rand(len(strings))   # generate random numbers
    mask = rand_vals >= threshold              # keep only >= threshold
    strings[~mask] = ""                        # replace with ""
    return strings.tolist(), rand_vals


class LabelImageTextPoseEncoderEmbedder(nn.Module):
    def __init__(self, emb_dim=32, text_hidden_dim=16): # text_hidden_dim=emb_dim//2
        super(LabelImageTextPoseEncoderEmbedder, self).__init__()
        self.emb_dim = emb_dim
        # Text embedder to process the condition string (e.g. "left eye, nasal")
        self.text_embedder = CLIPTextConditioner(output_dim=text_hidden_dim)

        # SE2 pose predictor - it will predict three outputs: θ (radians), dx, dy
        self.se2_predictor = nn.Sequential(  
            nn.Linear(text_hidden_dim, 256),  # CLIP embedding dim = 512  
            nn.ReLU(),  
            nn.Linear(256, 3)     # Output: [θ (radians), dx, dy]  
        )  
        # Project SE(2) + text embeddings to diffusion dimension  
        self.rle = nn.Linear(3 + text_hidden_dim, text_hidden_dim)  # relative position encoding
        
        # OD image branch (same as before)
        self.od_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), # rop -- 3, xray -- 1
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, emb_dim//2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        # Final fusion: concatenate OD image features and text features, then project.
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, od_image, text_condition):
        if isinstance(text_condition, str):
            text_condition = [text_condition]
        
        # OD image encoding.
        od_feat = self.od_encoder(od_image).squeeze(-1).squeeze(-1)  # shape: [B, emb_dim//2]
        
        # dropout in text during training for robustness
        # try:
        #     text_condition_indices, _ = drop_strings_numpy(text_condition, threshold=0.15)
        # except:
        #     import pdb; pdb.set_trace()
        text_condition_indices = text_condition#[0]
        # Text condition encoding.
        text_feat = self.text_embedder(text_condition_indices)       # shape: [B, emb_dim//2]

        # SE2 pose prediction
        pose = self.se2_predictor(text_feat) # shape: [B, 3]
        text_positioned = torch.cat([text_feat, pose], dim=1)         # shape: [B, emb_dim//2 + 3]
        fused_text = self.rle(text_positioned)                       # shape: [B, emb_dim//2]

        # Concatenate and project.
        combined = torch.cat([od_feat, fused_text], dim=1)             # shape: [B, emb_dim]
        context = self.proj(combined)                                 # shape: [B, emb_dim]

        # return context, pose
        return context


class Label2ImageTextPoseEncoderEmbedder(nn.Module):
    def __init__(self, emb_dim=32, text_hidden_dim=16):
        super(Label2ImageTextPoseEncoderEmbedder, self).__init__()
        self.emb_dim = emb_dim
        # Text embedder to process the condition string (e.g. "left eye, nasal")
        self.text_embedder = CLIPTextConditioner(output_dim=text_hidden_dim)

        # OD image branch
        self.od_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, emb_dim//4, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Peripheral image branch (same architecture)
        self.peripheral_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, emb_dim//4, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Cross-modal attention fusion
        self.image_proj = nn.Linear(emb_dim//2, text_hidden_dim)
        self.text_gate = nn.Sequential(
            nn.Linear(text_hidden_dim, emb_dim//2),
            nn.Sigmoid()
        )
        
        # Final fusion: concatenate and project
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, od_image, peripheral_image, text_condition):
        if isinstance(text_condition, str):
            text_condition = [text_condition]
        
        # Image encoding
        od_feat = self.od_encoder(od_image).squeeze(-1).squeeze(-1)  # [B, emb_dim//4]
        periph_feat = self.peripheral_encoder(peripheral_image).squeeze(-1).squeeze(-1)  # [B, emb_dim//4]
        
        # Concatenate image features
        image_feat = torch.cat([od_feat, periph_feat], dim=1)  # [B, emb_dim//2]
        
        # Text condition encoding
        text_feat = self.text_embedder(text_condition)  # [B, text_hidden_dim]
        
        # Cross-modal gating: text gates image features
        gate = self.text_gate(text_feat)  # [B, emb_dim//2]
        gated_image = image_feat * gate  # [B, emb_dim//2]
        
        # Project and concatenate
        image_proj = self.image_proj(gated_image)  # [B, text_hidden_dim]
        combined = torch.cat([image_proj, text_feat], dim=1)  # [B, emb_dim]
        context = self.proj(combined)  # [B, emb_dim]
        
        return context




if __name__ == "__main__":
    import torch
    # fov_label = torch.randint(0, 2, (16,))
    # embedder = LabelEmbedder(emb_dim=1024, num_classes=2)
    # cond_emb = embedder(fov_label)
    # print(cond_emb.shape)

    # od_img = torch.randn(16, 3, 256, 256)
    # label = torch.randint(0, 2, (16,))
    # joint_encoder = LabelImageEmbedder(emb_dim=1024, num_classes=2)
    # cond = joint_encoder(od_img, label)
    # print(cond.shape)

    od_img = torch.randn(1, 3, 256, 256)
    peri_img = torch.randn(1, 3, 256, 256)
    label = ['left eye, nasal view']

    text_embedder = LabelImageTextPoseEncoderEmbedder(emb_dim=1024, text_hidden_dim=512)
    text_cond = text_embedder(od_img,label)[0]
    print(text_cond.shape)

    # text_embedder = Label2ImageTextPoseEncoderEmbedder(emb_dim=1024, text_hidden_dim=512)
    # text_cond = text_embedder(od_img, peri_img,label)
    # print(text_cond.shape)

