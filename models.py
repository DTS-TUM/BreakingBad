import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import VOCAB_SIZE


class FEMNISTNet(torch.nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 62)

    def forward(self, x):

        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ShakespeareLSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=8, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)     
        out, _ = self.lstm(x) 
        return self.fc(out[:, -1])   


class DropPath(nn.Module):


    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device)
        noise.bernoulli_(keep_prob)
        return x * noise / keep_prob  


class CCTEncoderLayer(nn.Module):


    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        drop_path_prob: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path_prob)
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        h = self.attn_drop(h)
        x = x + self.drop_path1(h)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class CCT(nn.Module):

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 7,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.tokenizer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.GELU(),                                   # Fix 1
            nn.MaxPool2d(2),                             # 16×16
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),                                   # Fix 1
            nn.MaxPool2d(2),                             # 8×8
        )
        seq_len = 8 * 8  

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dpr = [
            drop_path_rate * i / (num_layers - 1)
            for i in range(num_layers)
        ]

        self.layers = nn.ModuleList([
            CCTEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path_prob=dpr[i], 
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.pool_attn = nn.Linear(embed_dim, 1)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.zeros_(self.pool_attn.weight)
        nn.init.zeros_(self.pool_attn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        attn_weights = torch.softmax(self.pool_attn(x), dim=1)
        x = (attn_weights * x).sum(dim=1) 
        return self.head(x)


DEIT_NAME = "deit_tiny_patch16_224"

DEIT_WEIGHTS = os.environ.get(
    "DEIT_WEIGHTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained", DEIT_NAME + ".pth"),
)


def ensure_local_deit_weights(path: str = None) -> str:

    import timm
    path = path or DEIT_WEIGHTS
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[DeiT] local weights not found, downloading {DEIT_NAME} once to {path}")
    reference = timm.create_model(DEIT_NAME, pretrained=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(reference.state_dict(), tmp)
    os.replace(tmp, path)
    return path


class DeiTTiny(nn.Module):

    def __init__(self, num_classes: int = 100, img_size: int = 112,
                 pretrained: bool = True, drop_path_rate: float = 0.1):
        super().__init__()
        import timm  # local import so the CCT and FEMNIST paths do not require timm

        self.img_size = img_size
        extra = {}
        if pretrained:
            extra["pretrained_cfg_overlay"] = dict(file=ensure_local_deit_weights())
        self.backbone = timm.create_model(
            DEIT_NAME,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            **extra,
        )

        nn.init.zeros_(self.backbone.head.weight)
        nn.init.zeros_(self.backbone.head.bias)

        cfg = getattr(self.backbone, "pretrained_cfg", None) or {}
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std = cfg.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bicubic", align_corners=False)
            x = x.clamp(0.0, 1.0) 
        x = (x - self.mean) / self.std
        return self.backbone(x)


def uses_deit(data_source: str) -> bool:
    return data_source == "CIFAR100"


def build_model(data_source: str, pretrained: bool = True, img_size: int = 112) -> nn.Module:

    if data_source == "CIFAR10":
        return CCT()
    if data_source == "CIFAR100":
        return DeiTTiny(num_classes=100, img_size=img_size, pretrained=pretrained)
    if data_source == "FEMNIST":
        return FEMNISTNet()
    if data_source == "Shakespeare":
        return ShakespeareLSTM()
    raise ValueError(f"Unknown data source: {data_source}")
