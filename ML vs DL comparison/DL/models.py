import torch.nn as nn
import torch
from torchvision import models
from einops import rearrange

# Transformer implementation: https://github.com/lucidrains/vit-pytorch
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


################################
### CNN-Transformer
################################

class CNN_Transformer(nn.Module):
    def __init__(self, *, backbone_name, seq_length, n_classes, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        self.backbone_name = backbone_name
        self.seq_length = seq_length
        self.n_classes = n_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dim_head = dim_head
        self.drop = dropout
        self.emb_dropout = emb_dropout

        if self.backbone_name == 'resnet18':
            self.conv_model = models.resnet18(pretrained=True)
            num_ftrs = self.conv_model.fc.in_features  # 512
            self.conv_model.fc = nn.Linear(num_ftrs, self.dim)

        elif self.backbone_name == 'resnet50':
            self.conv_model = models.resnet50(pretrained=True)
            num_ftrs = self.conv_model.fc.in_features  # 2048
            
            self.conv_model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),  # First layer reduces the feature size from 2048 to 512
                nn.ReLU(),  # Activation function to introduce non-linearity
                nn.Linear(512, self.dim)
            )# Second layer reduces the feature size from 512 to 64
        
           
        elif self.backbone_name == 'resnext50_32x4d':
            self.conv_model = models.resnext50_32x4d(pretrained=True)
            num_ftrs = self.conv_model.fc.in_features  # 2048
            
            self.conv_model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),  # First layer reduces the feature size from 2048 to 512
                nn.ReLU(),  # Activation function to introduce non-linearity
                nn.Linear(512, self.dim)
            )


        for param in self.conv_model.parameters():
            param.requires_grad = True

        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_length, self.dim))
        self.dropout = nn.Dropout(self.emb_dropout)
        self.transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.drop)
        self.output_layer = nn.Sequential(nn.Linear(self.seq_length * self.dim, self.n_classes))

    def forward(self, t):
        batch_size, timesteps, channels, h, w = t.shape
        t = t.view(batch_size * timesteps, channels, h, w)
        t = self.conv_model(t)
        t = t.view(batch_size, timesteps, -1)

        t += self.pos_embedding[:, :timesteps]
        t = self.dropout(t)
        t = self.transformer(t)
        t = t.reshape(batch_size, -1)
        t = self.output_layer(t)

        return t

    
################################
### CNN-TransformerBimodal
################################

class CNN_TransformerBimodal(nn.Module):
    def __init__(self, *, backbone_name, seq_length, n_classes, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        self.backbone_name = backbone_name
        self.seq_length = seq_length
        self.n_classes = n_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dim_head = dim_head
        self.drop = dropout
        self.emb_dropout = emb_dropout

        if self.backbone_name == 'resnet18':
            self.conv_model = models.resnet18(pretrained=True)
            num_ftrs = self.conv_model.fc.in_features  # 512
            self.conv_model.fc = nn.Linear(num_ftrs, self.dim)

        elif self.backbone_name == 'resnet50':
            self.conv_model = models.resnet50(pretrained=True)
            num_ftrs = self.conv_model.fc.in_features  # 2048
            
            self.conv_model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),  # First layer reduces the feature size from 2048 to 512
                nn.ReLU(),  # Activation function to introduce non-linearity
                nn.Linear(512, self.dim)
            )# Second layer reduces the feature size from 512 to 64
        
           
        elif self.backbone_name == 'resnext50_32x4d':
            self.conv_model = models.resnext50_32x4d(pretrained=True)
            num_ftrs = self.conv_model.fc.in_features  # 2048
            
            self.conv_model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),  # First layer reduces the feature size from 2048 to 512
                nn.ReLU(),  # Activation function to introduce non-linearity
                nn.Linear(512, self.dim)
            )



        for param in self.conv_model.parameters():
            param.requires_grad = True

        # Linear layer for processing (x, y) coordinates
        self.coord_fc = nn.Sequential(
            nn.Linear(2, self.dim),
            nn.ReLU()
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_length, self.dim * 2))
        self.dropout = nn.Dropout(self.emb_dropout)
        self.transformer = Transformer(self.dim * 2, self.depth, self.heads, self.dim_head, self.mlp_dim, self.drop)
        self.output_layer = nn.Sequential(nn.Linear(self.seq_length * self.dim * 2, self.n_classes))


    def forward(self, images, coords):
        batch_size, timesteps, channels, h, w = images.shape
        images = images.view(batch_size * timesteps, channels, h, w)
        images = self.conv_model(images)
        images = images.view(batch_size, timesteps, -1)

        coords = self.coord_fc(coords)

        # Concatenate CNN features with coordinate features
        combined_features = torch.cat((images, coords), dim=-1)
        combined_features += self.pos_embedding[:, :timesteps]
        combined_features = self.dropout(combined_features)
        combined_features = self.transformer(combined_features)
        combined_features = combined_features.reshape(batch_size, -1)
        combined_features = self.output_layer(combined_features)

        return combined_features   
        
    
