"""
Các kiến trúc mô hình Deep Learning (PyTorch `nn.Module`) dùng trong khóa học:
1. `SimpleMLP`: Mạng truyền thẳng đa tầng (Multi-Layer Perceptron).
2. `SimpleCNN`: Mạng tích chập cơ bản (Convolutional Neural Network).
3. `ResidualCNN`: Mạng tích chập có kết nối tắt (Residual / Skip Connection).
4. `MiniUNet`: Kiến trúc Encoder-Decoder có skip connection cho bài toán Phân vùng ảnh (Segmentation).
5. `ConvAutoencoder`: Bộ tự mã hóa tích chập cho bài toán Tái tạo & Phát hiện bất thường (Anomaly Detection).
6. `SiameseEmbeddingNet`: Mạng trích xuất vector nhúng chuẩn hóa L2 cho Metric Learning & Image Retrieval.
7. `MiniViT`: Mô hình Vision Transformer thu nhỏ minh họa Patch Embedding, Positional Encoding và Self-Attention.

Quy tắc quan trọng trong PyTorch:
- Các mô hình phân loại đa lớp trả về `logits` thô (chưa qua `Softmax`) để dùng trực tiếp với `nn.CrossEntropyLoss()`.
- Mô hình phân vùng nhị phân `MiniUNet` trả về `logits` thô (chưa qua `Sigmoid`) để dùng với `nn.BCEWithLogitsLoss()`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """
    Mạng Multi-Layer Perceptron (MLP) cho phân loại ảnh.
    - Đầu vào: `(N, C, H, W)`
    - Đầu ra: `logits` có shape `(N, num_classes)`
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        input_dim = in_channels * image_size * image_size
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = self.flatten(x)
        logits = self.classifier(x_flat)
        return logits


class SimpleCNN(nn.Module):
    """
    Mạng tích chập (CNN) gồm 3 khối tích chập + Global Average Pooling (`AdaptiveAvgPool2d(1)`).
    - Đầu vào: `(N, C, H, W)`
    - Đầu ra: `logits` có shape `(N, num_classes)`
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        base_channels: int = 16,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Khối 1: giữ kích thước không gian nhờ padding=1, sau đó giảm một nửa bằng MaxPool2d(2)
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Khối 2
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Khối 3
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.features(x)
        pooled = self.pool(feat_map)
        flat = self.flatten(pooled)
        logits = self.fc(self.dropout(flat))
        return logits


class ResidualBlock(nn.Module):
    """
    Khối phần dư (Residual Block): `y = ReLU(F(x) + shortcut(x))`.
    Giúp đạo hàm (gradient) truyền ngược trực tiếp qua nhánh tắt `shortcut`, chống triệt tiêu đạo hàm.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResidualCNN(nn.Module):
    """
    Mạng CNN sử dụng `ResidualBlock` (kiểu ResNet thu nhỏ) phục vụ bài học CNN và Transfer Learning/Fine-tuning.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 4, base_channels: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResidualBlock(base_channels, base_channels, stride=1)
        self.stage2 = ResidualBlock(base_channels, base_channels * 2, stride=2)
        self.stage3 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.fc(feats)


class MiniUNet(nn.Module):
    """
    Kiến trúc U-Net thu nhỏ cho bài toán Phân vùng ảnh (Semantic Segmentation).
    - Đầu vào: `(N, in_channels, H, W)` với `H, W` chia hết cho 4.
    - Đầu ra: `logits` `(N, out_channels, H, W)` (chưa qua Sigmoid/Softmax).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


class ConvAutoencoder(nn.Module):
    """
    Mô hình Convolutional Autoencoder nén ảnh về vector ẩn `latent_dim` rồi tái tạo lại ảnh `[0, 1]`.
    Dùng cho học biểu diễn không giám sát và phát hiện bất thường (Anomaly Detection).
    """

    def __init__(self, in_channels: int = 3, image_size: int = 48, latent_dim: int = 32) -> None:
        super().__init__()
        self.image_size = image_size
        feat_hw = image_size // 4
        self.feat_hw = feat_hw
        flat_dim = 32 * feat_hw * feat_hw

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.to_latent = nn.Linear(flat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x).flatten(1)
        return self.to_latent(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.from_latent(z))
        h = h.view(-1, 32, self.feat_hw, self.feat_hw)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class SiameseEmbeddingNet(nn.Module):
    """
    Mạng nhúng (Embedding Network) dùng chung trọng số cho Siamese / Triplet Learning.
    Trả về vector đặc trưng đã chuẩn hóa độ dài L2 bằng 1 (`F.normalize(z, p=2, dim=1)`).
    """

    def __init__(self, in_channels: int = 3, embedding_dim: int = 32) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return F.normalize(emb, p=2, dim=1)


class MiniViT(nn.Module):
    """
    Vision Transformer (ViT) thu nhỏ phục vụ mục đích giảng dạy:
    1. Chia ảnh `(N, C, H, W)` thành lưới các mảnh (`patches`) kích thước `patch_size x patch_size`.
    2. Chiếu tuyến tính mỗi patch thành vector `embed_dim` thông qua `Conv2d(kernel_size=patch_size, stride=patch_size)`.
    3. Ghép thêm token phân loại `[CLS]` và cộng mã hóa vị trí học được (`pos_embed`).
    4. Đưa qua chuỗi tầng `TransformerEncoderLayer` (Multi-Head Self-Attention + MLP).
    5. Lấy trạng thái của token `[CLS]` ở đầu ra để dự đoán `logits`.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 4,
        embed_dim: int = 48,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size phải chia hết cho patch_size")
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.num_patches, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout_p)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N_patches) -> (B, N_patches, D)
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        encoded = self.norm(self.encoder(tokens))
        cls_out = encoded[:, 0]
        return self.head(cls_out)
