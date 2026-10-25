import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=[192, 256], patch_size=8, in_channels=1, embed_dim=64, project=True):
        super().__init__()
        h, w = img_size
        self.num_patches = (h // patch_size) * (w // patch_size)
        self.project = project  

        if project:
            self.projection = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_size * patch_size * in_channels, embed_dim)  
            )
        else:
            self.patch_size = patch_size
            self.out_dim = patch_size * patch_size * in_channels

    def forward(self, x):
        if self.project:
            return self.projection(x)
        else:
            return rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, mlp_dim=512, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, 
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(mlp_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        #rama atención
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        #rama MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size =[192,256], patch_size=8, num_classes=4, embed_dim=64, 
                 depth = 6, num_heads=8, project = True, mlp_dim=1024, dropout=0.0):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size,  1, embed_dim, project=project)
        num_patches = self.patch_embed.num_patches

        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

       
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]

        #patch embedding: (B, N, D)
        x = self.patch_embed(x)

        #añadir CLS token al inicio de la secuencia
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = torch.cat([cls, x], dim=1)            # (B, N+1, D)

        #suma de codificación posicional
        x = x + self.pos_embedding
        x = self.dropout(x)

        #pasar por el encoder
        x = self.encoder(x)
        x = self.norm(x)

        #coger solo el CLS token para clasificar
        cls_output = x[:, 0]                      # (B, D)
        return self.head(cls_output)              # (B, num_classes)