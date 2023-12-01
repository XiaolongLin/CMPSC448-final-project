from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

from .transformer import Transformer


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class PositionalEmbedding1D(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
    
    def forward(self, x):
        return x + self.pos_embedding


class ViT(nn.Module):
    def __init__(
        self, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout_rate: float = 0.1,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3, 
        image_size: int = 224,
        num_classes: int = 90,
    ):
        super().__init__()               
        h, w = as_tuple(image_size)
        fh, fw = as_tuple(patches)  
        gh, gw = h // fh, w // fw  
        seq_len = gh * gw

        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1
        
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()
        
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate)
        
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)

        self.init_weights()

        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) 
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)  
        nn.init.constant_(self.class_token, 0)


    def forward(self, x):
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  
        x = x.flatten(2).transpose(1, 2)  
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  
        if hasattr(self, 'positional_embedding'): 
            x = self.positional_embedding(x) 
        x = self.transformer(x) 
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0] 
            x = self.fc(x)  
        return x
