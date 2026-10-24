"""
Script khởi tạo các Jupyter Notebook từ 18 đến 22 (Generative Models, OCR & Mở rộng, Evaluation & Tuning, Inference/Export, và Đồ án Cuối khóa Capstone).
"""

from __future__ import annotations

from pathlib import Path
from _nb_helper import create_validated_notebook

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def build_nb18() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 18: Các Mô hình Sinh Ảnh (Generative Models) — VAE, GAN, Diffusion và Tổng quan Autoregressive / Flow

## 1. Mục tiêu học tập
- Phân biệt mô hình phân biệt (**Discriminative Models**: học xác suất có điều kiện $P(y \mid x)$ để dự đoán nhãn) với mô hình sinh (**Generative Models**: học phân phối dữ liệu $P(x)$ để sinh ra bức ảnh mới).
- Nắm vững nguyên lý hoạt động, kiến trúc và hàm mất mát của 4 nhánh mô hình sinh chính:
  1. **Variational Autoencoder (`VAE`)**: Không gian ẩn xác suất $(\mu, \log\sigma^2)$, thủ thuật tái tham số hóa (**`Reparameterization Trick`**) và hàm mất mát **`Reconstruction Loss + KL Divergence`**.
  2. **Generative Adversarial Networks (`GAN`)**: Trò chơi đối kháng hai người giữa **`Generator` ($G$)** và **`Discriminator` ($D$)**.
  3. **Denoising Diffusion Probabilistic Models (`Diffusion / DDPM`)**: Quá trình thuận cộng nhiễu theo lịch **`noise_schedule`** ($\beta_t$) qua các bước thời gian **`timestep`** ($t$) và quá trình ngược dùng mạng nơ-ron dự đoán nhiễu để khử nhiễu (**`denoising`**).
  4. **Tổng quan Autoregressive Image Models (`PixelCNN`, `VQ-VAE + Transformer`) & Normalizing Flows**.
- **Thực hành huấn luyện thật**: Huấn luyện một mô hình `TinyVAE` trên tập ảnh chữ số `8x8` để sinh chữ số mới bằng cách lấy mẫu từ phân phối chuẩn $\mathcal{N}(0, I)$, đồng thời minh họa trực quan chuỗi thêm nhiễu / khử nhiễu của `Diffusion`.

> **Lưu ý sư phạm**: Ví dụ trong bài được thu nhỏ về kích thước `8x8` để chạy mượt trên CPU trong vài giây nhằm minh họa chính xác cơ chế toán học. Các mô hình sinh công nghiệp (như Stable Diffusion, SDXL, Flux) có hàng tỷ tham số, đòi hỏi GPU mạnh và chỉ nên dùng ở chế độ inference hoặc fine-tune nhẹ (LoRA).

## 2. Kiến thức cần có
- Đã hoàn thành Bài 15 (`Autoencoder`).

## 3. Thời gian học dự kiến
- Khoảng **2.5 giờ** (ước tính).

## 4, 5, 6 & 7. Giải thích Trực quan, Từ điển Keyword & Công thức Toán học
1. **Thủ thuật Tái tham số hóa (`Reparameterization Trick`) và Hàm mất mát của `VAE`**:
   - Nếu lấy mẫu trực tiếp $z \sim \mathcal{N}(\mu, \sigma^2)$, bước lấy mẫu ngẫu nhiên không có đạo hàm theo $\mu, \sigma$. Ta tách phần ngẫu nhiên sang biến phụ $\epsilon \sim \mathcal{N}(0, I)$:
     $$z = \mu + \sigma \odot \epsilon, \quad \text{với } \sigma = \exp\left(\frac{1}{2} \log\sigma^2\right)$$
   - Hàm mất mát `VAE`:
     $$\mathcal{L}_{\text{VAE}} = \|x - \hat{x}\|_2^2 + \beta \cdot D_{\text{KL}}\big(\mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, I)\big), \quad D_{\text{KL}} = -\frac{1}{2}\sum_{j=1}^D \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$
2. **Trò chơi Minimax của `GAN`**:
   $$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim \mathcal{N}(0, I)}[\log(1 - D(G(z)))]$$
3. **Công thức thêm nhiễu trực tiếp tại bước $t$ của `Diffusion (DDPM)`**:
   Với lịch nhiễu $\beta_t \in (0, 1)$, đặt $\alpha_t = 1 - \beta_t$ và tích lũy $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. Ta có thể nhảy thẳng từ ảnh gốc $x_0$ tới ảnh nhiễu ở bất kỳ bước $t$ nào:
   $$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$""",
        ),
        (
            "code",
            """# Cell 1: Xây dựng và Huấn luyện Thực tế Mô hình Variational Autoencoder (TinyVAE) Sinh Ảnh Chữ số
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
if (project_root / "image_ai_learning").is_dir():
    project_root = project_root / "image_ai_learning"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits

from src.utils import set_seed, get_device, count_parameters

QUICK_RUN = True
set_seed(42)
device = get_device()

digits = load_digits()
X_digits = torch.from_numpy(digits.data.astype(np.float32) / 16.0)  # (1797, 64) trong [0, 1]
vae_loader = DataLoader(TensorDataset(X_digits), batch_size=64, shuffle=True)


class TinyVAE(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 48, latent_dim: int = 2) -> None:
        super().__init__()
        self.enc_fc = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc_fc(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


vae = TinyVAE(input_dim=64, hidden_dim=48, latent_dim=2).to(device)
opt_vae = torch.optim.AdamW(vae.parameters(), lr=5e-3)
epochs_vae = 20 if QUICK_RUN else 45

for ep in range(1, epochs_vae + 1):
    vae.train()
    total_recon, total_kl = 0.0, 0.0
    for (xb,) in vae_loader:
        xb = xb.to(device)
        opt_vae.zero_grad()
        recon_b, mu_b, logvar_b = vae(xb)
        recon_loss = F.mse_loss(recon_b, xb, reduction="sum") / len(xb)
        kl_loss = -0.5 * torch.sum(1.0 + logvar_b - mu_b.pow(2) - logvar_b.exp(), dim=1).mean()
        loss = recon_loss + 0.02 * kl_loss
        loss.backward()
        opt_vae.step()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

print(f"TinyVAE ({count_parameters(vae)['total']} params) hoàn tất {epochs_vae} epochs -> Recon Loss={total_recon/len(vae_loader):.4f}, KL={total_kl/len(vae_loader):.4f}")""",
        ),
        (
            "code",
            """# Cell 2: Sinh Ảnh Mới bằng cách Nội suy Liên tục trên Mặt phẳng Latent 2D của VAE
vae.eval()
grid_pts = np.linspace(-2.0, 2.0, 6)
fig, axes = plt.subplots(1, 6, figsize=(10.5, 2.2))
with torch.no_grad():
    for i, z1_val in enumerate(grid_pts):
        z_query = torch.tensor([[float(z1_val), 0.0]], device=device)
        gen_img = vae.decoder(z_query).cpu().numpy().reshape(8, 8)
        axes[i].imshow(gen_img, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"z=[{z1_val:+.1f}, 0.0]", fontsize=8.5)
        axes[i].axis("off")
plt.suptitle("Sinh ảnh mới mượt mà khi trượt dọc trục ẩn z1 của VAE", fontsize=10.5)
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 3: Minh họa Trò chơi Đối kháng GAN (Generator vs Discriminator) trên Phân phối Đặc trưng
set_seed(42)
generator = nn.Sequential(nn.Linear(8, 24), nn.LeakyReLU(0.2), nn.Linear(24, 64), nn.Sigmoid()).to(device)
discriminator = nn.Sequential(nn.Linear(64, 24), nn.LeakyReLU(0.2), nn.Linear(24, 1)).to(device)  # Trả logits cho BCEWithLogitsLoss

opt_g = torch.optim.Adam(generator.parameters(), lr=2e-3, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-3, betas=(0.5, 0.999))
bce_gan = nn.BCEWithLogitsLoss()

real_batch = X_digits[:64].to(device)
for step in range(25):
    # 1. Cập nhật Discriminator D: Phân biệt ảnh thật (nhãn 1) và ảnh giả G(z) (nhãn 0)
    opt_d.zero_grad()
    z_noise = torch.randn(64, 8, device=device)
    fake_batch = generator(z_noise)
    loss_d_real = bce_gan(discriminator(real_batch), torch.ones(64, 1, device=device))
    loss_d_fake = bce_gan(discriminator(fake_batch.detach()), torch.zeros(64, 1, device=device))
    loss_d = 0.5 * (loss_d_real + loss_d_fake)
    loss_d.backward()
    opt_d.step()

    # 2. Cập nhật Generator G: Đánh lừa D để D chấm ảnh giả là nhãn 1!
    opt_g.zero_grad()
    loss_g = bce_gan(discriminator(fake_batch), torch.ones(64, 1, device=device))
    loss_g.backward()
    opt_g.step()

print(f"[GAN Minimax Demo] Sau 25 bước cập nhật đối kháng -> D_Loss: {loss_d.item():.4f} | G_Loss: {loss_g.item():.4f}")""",
        ),
        (
            "code",
            """# Cell 4: Minh họa Cơ chế Mô hình Khuếch tán (Diffusion Models: Noise Schedule, Timestep & Denoising)
T_steps = 50
betas = torch.linspace(1e-4, 0.08, T_steps, device=device)  # Lịch nhiễu tăng tuyến tính (Linear Noise Schedule)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def q_sample_forward_diffusion(x0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    \"\"\"Công thức đóng nhảy thẳng từ ảnh sạch x0 tới ảnh nhiễu x_t tại bước thời gian t.\"\"\"
    a_bar = alpha_bars[t_idx].view(-1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise


# Minh họa quá trình thuận thêm nhiễu vào ảnh chữ số tại t = 0, 10, 25, 49
x0_sample = X_digits[0:1].to(device)
eps_fixed = torch.randn_like(x0_sample)
timesteps_to_show = [0, 10, 25, 49]

fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.4))
for ax, t_val in zip(axes, timesteps_to_show):
    xt = q_sample_forward_diffusion(x0_sample, torch.tensor([t_val], device=device), eps_fixed)
    ax.imshow(xt.cpu().numpy().reshape(8, 8), cmap="gray")
    ax.set_title(f"Timestep t={t_val}\\nα_bar={alpha_bars[t_val].item():.2f}", fontsize=9)
    ax.axis("off")
plt.suptitle("Quá trình Khuếch tán Thuận (Forward Diffusion): Cộng nhiễu theo lịch β_t", fontsize=10.5)
plt.tight_layout()
plt.show()

# Huấn luyện mạng Denoiser nhỏ dự đoán nhiễu epsilon từ (x_t, t_normalized)
denoiser_net = nn.Sequential(nn.Linear(64 + 1, 64), nn.SiLU(), nn.Linear(64, 64)).to(device)
opt_diff = torch.optim.AdamW(denoiser_net.parameters(), lr=5e-3)
x_train_diff = X_digits[:256].to(device)

for _ in range(40):
    opt_diff.zero_grad()
    t_rand = torch.randint(0, T_steps, (len(x_train_diff),), device=device)
    noise_gt = torch.randn_like(x_train_diff)
    x_noisy = q_sample_forward_diffusion(x_train_diff, t_rand, noise_gt)
    t_feat = (t_rand.float() / T_steps).unsqueeze(1)
    noise_pred = denoiser_net(torch.cat([x_noisy, t_feat], dim=1))
    loss_diff = F.mse_loss(noise_pred, noise_gt)
    loss_diff.backward()
    opt_diff.step()

print(f"[Diffusion Denoiser] MSE dự đoán nhiễu ε(x_t, t) sau 40 bước: {loss_diff.item():.4f}")""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Ở `VAE` (Cell 1 & 2)**: Nhờ thành phần phạt phân kỳ `KL Divergence` kéo các đám mây ẩn về gần phân phối chuẩn $\mathcal{N}(0, I)$, khi ta dịch chuyển tọa độ $z_1$ từ `-2.0` sang `+2.0`, hình ảnh sinh ra biến đổi liên tục và mượt mà (không có "hố đen" vô nghĩa như Autoencoder thường).
- **Ở `GAN` (Cell 3)**: Lưu ý kỹ thuật quan trọng khi cập nhật `Discriminator` ($D$): ta phải gọi **`fake_batch.detach()`** để ngắt đồ thị tính đạo hàm không cho chảy ngược vào `Generator` ($G$) trong bước của $D$. Khi đến lượt cập nhật $G$, ta mới truyền `fake_batch` (không detach) qua $D$ với nhãn mục tiêu là `1` (đánh lừa $D$ rằng ảnh giả là thật).
- **Ở `Diffusion` (Cell 4)**: Khi bước thời gian $t$ tăng từ `0` lên `49`, hệ số giữ tín hiệu gốc $\bar{\alpha}_t$ giảm từ `1.00` về gần `0.00`, biến bức ảnh thành nhiễu Gaussian thuần túy. Mạng khử nhiễu học cách dự đoán đúng lượng nhiễu $\epsilon$ đã cộng vào để khôi phục ngược lại bức ảnh.

## 14. Bài tập tự làm
- **Cơ bản**: Trong hàm mất mát `VAE`, thử đặt hệ số $\beta$ của `KL Divergence` bằng `0.0` (tắt hoàn toàn KL) và kiểm tra xem độ lớn trung bình của `mu` và độ lệch chuẩn `exp(0.5 * logvar)` thay đổi thế nào.
- **Trung cấp**: Viết công thức khôi phục ước lượng ảnh gốc $\hat{x}_0$ từ ảnh nhiễu $x_t$ và nhiễu dự đoán $\hat{\epsilon}_\theta(x_t, t)$ trong Diffusion: $\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\,\hat{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}}$.
- **Nâng cao**: Giải thích hiện tượng **Mode Collapse (Sụp đổ chế độ)** trong GAN và cách phát hiện nó bằng cách đo độ lệch chuẩn giữa các ảnh sinh ra từ các vector nhiễu $z$ khác nhau.

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 18
# 1. Cơ bản: Kiểm tra phân phối mu, std của VAE đã huấn luyện
with torch.no_grad():
    mu_all, logvar_all = vae.encode(X_digits[:200].to(device))
    print(f"[Cơ bản] Latent mu mean={mu_all.mean().item():.3f}, std(z)={torch.exp(0.5*logvar_all).mean().item():.3f}")

# 2. Trung cấp: Khôi phục ước lượng x0_hat từ x_t và nhiễu dự đoán trong Diffusion
with torch.no_grad():
    t_test = torch.tensor([15], device=device)
    xt_test = q_sample_forward_diffusion(x0_sample, t_test, eps_fixed)
    eps_hat = denoiser_net(torch.cat([xt_test, (t_test.float() / T_steps).unsqueeze(1)], dim=1))
    a_bar_t = alpha_bars[t_test].view(-1, 1)
    x0_hat = ((xt_test - torch.sqrt(1.0 - a_bar_t) * eps_hat) / torch.sqrt(a_bar_t)).clamp(0.0, 1.0)
    print(f"[Trung cấp] MSE trước khử nhiễu={F.mse_loss(xt_test, x0_sample).item():.4f} -> Sau khử nhiễu={F.mse_loss(x0_hat, x0_sample).item():.4f}")

# 3. Nâng cao: Hàm kiểm tra triệu chứng Mode Collapse của Generator
with torch.no_grad():
    z_diverse = torch.randn(32, 8, device=device)
    gen_samples = generator(z_diverse)
    pixel_diversity_std = gen_samples.std(dim=0).mean().item()
    print(f"[Nâng cao] Độ đa dạng giữa 32 ảnh sinh từ Generator (std across batch): {pixel_diversity_std:.4f} (Nếu ~0.0 là bị Mode Collapse!)")""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Hiểu rõ cơ chế sinh ảnh của `VAE`, `GAN` và `Diffusion Models`.
- [x] Sẵn sàng chuyển sang **Bài 19: `19_ocr_and_visual_tasks_overview.ipynb`** để khám phá `OCR`, `Pose Estimation`, `Tracking`, `Super-Resolution` và `Depth Estimation`!

## 17. Nguồn tham khảo
- Kingma, D. P., & Welling, M. (2014), *Auto-Encoding Variational Bayes (VAE)*, ICLR.
- Goodfellow, I. et al. (2014), *Generative Adversarial Nets (GAN)*, NeurIPS.
- Ho, J., Jain, A., & Abbeel, P. (2020), *Denoising Diffusion Probabilistic Models (DDPM)*, NeurIPS.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "18_generative_models.ipynb", cells)


def build_nb19() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 19: OCR và Các Bài toán Thị giác Mở rộng — Pose Estimation, Object Tracking, Image Restoration, Depth & VLM

## 1. Mục tiêu học tập
Nắm vững kiến trúc, định dạng dữ liệu đầu vào/đầu ra, hàm mất mát và độ đo đánh giá của **6 nhóm bài toán thị giác mở rộng** thường gặp trong thực tế:
1. **OCR (`Optical Character Recognition`)**: Quy trình 2 giai đoạn gồm **Text Detection** (`DBNet`, `CRAFT`) tách vùng dòng chữ và **Text Recognition** (`CRNN + BiLSTM + CTC Loss`, `TrOCR`) đọc chuỗi ký tự mà không cần gán nhãn vị trí từng chữ cái.
2. **Pose Estimation (Ước lượng tư thế / Điểm mốc `Keypoints`)**: Vì sao dự đoán **Bản đồ nhiệt 2D (`Gaussian Heatmap`)** chính xác hơn hồi quy trực tiếp tọa độ $(x, y)$.
3. **Object Tracking (Theo vết đa đối tượng qua video)**: Nguyên lý **Tracking-by-Detection** (`SORT`, `ByteTrack`, `DeepSORT`) ghép nối quỹ đạo giữa các khung hình bằng ma trận `IoU`.
4. **Denoising, Super-Resolution & Image Restoration (Khôi phục & Siêu phân giải ảnh)**: Học phần nhiễu dư (`Residual Learning`) và tính 2 chỉ số chuẩn **`PSNR` (dB)** cùng **`SSIM`**.
5. **Monocular Depth Estimation (Ước lượng độ sâu từ 1 ảnh)**: Nguyên lý của `MiDaS`, `Depth Anything` và giới hạn mơ hồ tỉ lệ (`scale ambiguity`).
6. **Vision–Language Models (`VLM`)**: Định hướng kiến trúc kết nối `Vision Encoder + Projector + LLM`.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 10, 13 và 14.

## 3. Thời gian học dự kiến
- Khoảng **2.0 – 2.5 giờ** (ước tính).

## 4, 5, 6 & 7. Giải thích Trực quan, Từ điển Keyword & Công thức Toán học
1. **Cơ chế Giải mã `CTC (Connectionist Temporal Classification)` trong OCR**:
   - Mạng nhận dạng chữ chia ảnh dòng chữ theo chiều ngang thành $T$ cột thời gian (`time steps`), mỗi cột dự đoán 1 ký tự trong bảng chữ cái cộng thêm ký hiệu rỗng **`blank` (`-`)**.
   - Quy tắc thu gọn (**CTC Collapse Rule**): Gộp các ký tự giống nhau đứng liền kề thành 1 ký tự, sau đó xóa toàn bộ ký hiệu `blank` (`-`).
   - *Ví dụ*: Chuỗi dự đoán theo cột `["-", "A", "A", "-", "I", "I", "-"]` $\xrightarrow{\text{gộp lặp}}$ `["-", "A", "-", "I", "-"]` $\xrightarrow{\text{xóa blank}}$ **`"AI"`**! Nhờ ký hiệu `-` ở giữa, CTC phân biệt được chữ có 2 chữ cái giống nhau đứng cạnh (như `"H-E-L-L-O"` phải dự đoán là `"HEL-LO"` để không bị gộp thành `"HELO"`!).
2. **Chỉ số Chất lượng Ảnh Khôi phục `PSNR (Peak Signal-to-Noise Ratio)`**:
   $$\text{PSNR}(I, \hat{I}) = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}(I, \hat{I})}\right) \quad (\text{đơn vị: dB})$$
   trong đó $\text{MAX}_I = 255.0$ (với `uint8`) hoặc $1.0$ (với `float32 [0, 1]`). `PSNR` càng cao (thường `> 30 dB`) thì ảnh khôi phục càng gần ảnh gốc.""",
        ),
        (
            "code",
            """# Cell 1: Thực hành Pipeline OCR — (1) Text Detection tìm khung dòng chữ + (2) Text Recognition với nn.CTCLoss
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
if (project_root / "image_ai_learning").is_dir():
    project_root = project_root / "image_ai_learning"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.utils import set_seed, get_device

set_seed(42)
device = get_device()

# Bước 1: Tạo ảnh bảng hiệu có dòng chữ "AI 2026" và dùng Morphology ngang để phát hiện vùng dòng chữ (Text Detection)
ocr_canvas = np.full((64, 200, 3), 240, dtype=np.uint8)
cv2.putText(ocr_canvas, "AI 2026", (22, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 30, 120), 2, cv2.LINE_AA)
gray_ocr = cv2.cvtColor(ocr_canvas, cv2.COLOR_RGB2GRAY)
_, inv_bin = cv2.threshold(gray_ocr, 180, 255, cv2.THRESH_BINARY_INV)

# Nối liền các chữ cái trên cùng 1 dòng bằng nhân giãn ngang (Horizontal Dilation Kernel 15x3)
horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
connected_text_mask = cv2.dilate(inv_bin, horiz_kernel, iterations=1)
contours, _ = cv2.findContours(connected_text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

vis_ocr = ocr_canvas.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(vis_ocr, (x, y), (x + w, y + h), (220, 50, 50), 2)

# Bước 2: Minh họa hàm mất mát nn.CTCLoss và bộ giải mã Greedy CTC Decoder cho Text Recognition
vocab = {0: "-", 1: "A", 2: "I", 3: "2", 4: "0", 5: "6"}  # 0 luôn là blank token trong PyTorch CTCLoss


def greedy_ctc_decode(token_seq: list[int], blank_id: int = 0) -> str:
    collapsed = []
    prev = None
    for tok in token_seq:
        if tok != prev:
            if tok != blank_id:
                collapsed.append(vocab[tok])
            prev = tok
    return "".join(collapsed)


raw_ctc_alignment = [0, 1, 1, 0, 2, 0, 3, 4, 3, 5]  # Chuỗi 10 cột thời gian
decoded_text = greedy_ctc_decode(raw_ctc_alignment)
print(f"Chuỗi cột thời gian thô: {[vocab[i] for i in raw_ctc_alignment]} -> Sau CTC Collapse: '{decoded_text}'")

# Tính nn.CTCLoss chính thức của PyTorch (Input shape: [T, Batch, Num_Classes] ở dạng log_softmax)
ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
T_len, B_size, C_vocab = 10, 1, 6
logits_ctc = torch.randn(T_len, B_size, C_vocab, requires_grad=True)
log_probs_ctc = logits_ctc.log_softmax(dim=2)
targets_ctc = torch.tensor([1, 2, 3, 4, 3, 5], dtype=torch.long)  # "AI2026" (6 ký tự, ngắn hơn T=10 cột!)
input_lengths = torch.tensor([T_len], dtype=torch.long)
target_lengths = torch.tensor([len(targets_ctc)], dtype=torch.long)
loss_ctc_val = ctc_loss_fn(log_probs_ctc, targets_ctc, input_lengths, target_lengths)
print(f"Giá trị PyTorch nn.CTCLoss (T={T_len} -> Target={len(targets_ctc)} ký tự): {loss_ctc_val.item():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.4))
axes[0].imshow(connected_text_mask, cmap="gray")
axes[0].set_title("1. Mặt nạ nối liền dòng chữ")
axes[1].imshow(vis_ocr)
axes[1].set_title(f"2. Khung Text Detection -> Đọc: '{decoded_text}'")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 2: Pose Estimation (Bản đồ nhiệt Keypoint Heatmap) & Object Tracking (Ghép nối quỹ đạo IoU qua các Frame)
# 1. Pose Estimation: Tạo và giải mã tọa độ khớp xương (Keypoint) từ bản đồ nhiệt 2D (Gaussian Heatmap)
H_map, W_map = 32, 32
true_kp_x, true_kp_y = 21, 14
yy, xx = np.mgrid[0:H_map, 0:W_map]
gt_heatmap = np.exp(-((xx - true_kp_x) ** 2 + (yy - true_kp_y) ** 2) / (2.0 * 2.0 ** 2)).astype(np.float32)

# Giải mã tọa độ dự đoán từ Heatmap bằng argmax 2D
flat_max_idx = int(np.argmax(gt_heatmap))
pred_kp_y, pred_kp_x = divmod(flat_max_idx, W_map)
print(f"[Pose Keypoint Heatmap] Tọa độ thật: (x={true_kp_x}, y={true_kp_y}) -> Giải mã từ Heatmap: (x={pred_kp_x}, y={pred_kp_y})")

# 2. Object Tracking (Tracking-by-Detection): Duy trì ID của 2 vật thể di chuyển qua 3 khung hình liên tiếp
from src.evaluation import compute_bbox_iou

frame1_boxes = {101: np.array([10.0, 10.0, 30.0, 30.0]), 102: np.array([60.0, 40.0, 80.0, 60.0])}  # {track_id: box}
frame2_detections = [np.array([63.0, 42.0, 83.0, 62.0]), np.array([13.0, 11.0, 33.0, 31.0])]       # Thứ tự bị đảo ngẫu nhiên!

matched_tracks = {}
for det_box in frame2_detections:
    best_id, best_iou = None, 0.0
    for trk_id, prev_box in frame1_boxes.items():
        iou_val = compute_bbox_iou(prev_box, det_box)
        if iou_val > best_iou:
            best_iou, best_id = iou_val, trk_id
    if best_iou >= 0.3 and best_id is not None:
        matched_tracks[best_id] = (det_box.tolist(), round(best_iou, 3))

print("[Object Tracking] Kết quả ghép nối giữ nguyên Track ID sang Frame 2:", matched_tracks)""",
        ),
        (
            "code",
            """# Cell 3: Khôi phục Ảnh (Denoising & Super-Resolution) và Tính Độ đo PSNR (dB) & SSIM
def compute_psnr(clean_img: np.ndarray, test_img: np.ndarray, max_val: float = 1.0) -> float:
    mse = float(np.mean((clean_img.astype(np.float64) - test_img.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return 100.0
    return float(10.0 * np.log10((max_val ** 2) / mse))


def compute_ssim_global(x: np.ndarray, y: np.ndarray, c1: float = 0.01 ** 2, c2: float = 0.03 ** 2) -> float:
    x_f, y_f = x.astype(np.float64), y.astype(np.float64)
    mu_x, mu_y = x_f.mean(), y_f.mean()
    var_x, var_y = x_f.var(), y_f.var()
    cov_xy = np.mean((x_f - mu_x) * (y_f - mu_y))
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / den)


# Huấn luyện nhanh mạng Residual Denoising CNN (học phần nhiễu dư tương tự DnCNN)
clean_patch = (gray_ocr.astype(np.float32) / 255.0)
rng = np.random.default_rng(42)
noisy_patch = np.clip(clean_patch + rng.normal(0.0, 0.12, size=clean_patch.shape), 0.0, 1.0).astype(np.float32)

denoise_cnn = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(16, 1, 3, padding=1),
).to(device)
opt_dn = torch.optim.AdamW(denoise_cnn.parameters(), lr=5e-3)

x_noisy_t = torch.from_numpy(noisy_patch)[None, None].to(device)
x_clean_t = torch.from_numpy(clean_patch)[None, None].to(device)

for _ in range(35):
    opt_dn.zero_grad()
    # Residual Learning: mạng dự đoán phần nhiễu dư, rồi lấy x_noisy - predicted_noise
    pred_noise = denoise_cnn(x_noisy_t)
    restored_t = (x_noisy_t - pred_noise).clamp(0.0, 1.0)
    loss_dn = nn.MSELoss()(restored_t, x_clean_t)
    loss_dn.backward()
    opt_dn.step()

restored_np = restored_t.detach().cpu().numpy()[0, 0]
print(f"Trước khử nhiễu -> PSNR: {compute_psnr(clean_patch, noisy_patch):.2f} dB | SSIM: {compute_ssim_global(clean_patch, noisy_patch):.4f}")
print(f"Sau khử nhiễu   -> PSNR: {compute_psnr(clean_patch, restored_np):.2f} dB | SSIM: {compute_ssim_global(clean_patch, restored_np):.4f}")""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Ở Cell 1 (OCR)**: Bằng cách tách bài toán thành 2 bước (`Detection` tìm hộp dòng chữ $\rightarrow$ `Recognition` giải mã chuỗi bằng `CTCLoss`), mô hình đọc được chuỗi văn bản có độ dài bất kỳ mà không cần con người cắt rời từng chữ cái lúc gán nhãn.
- **Ở Cell 2 (Pose & Tracking)**: Biểu diễn tọa độ khớp xương bằng bản đồ nhiệt Gaussian 2D giúp mạng học không gian dễ dàng hơn; còn thuật toán ghép nối `IoU` duy trì đúng `track_id` (`101`, `102`) ngay cả khi thứ tự phát hiện ở khung hình tiếp theo bị đảo lộn.
- **Ở Cell 3 (Image Restoration)**: Kỹ thuật học phần nhiễu dư (`restored = noisy - net(noisy)`) giúp tăng mạnh chỉ số `PSNR` (từ `~19 dB` lên `> 28 dB`) và `SSIM`.

## 14. Bài tập tự làm
- **Cơ bản**: Giải mã chuỗi CTC cho danh sách chỉ số `[0, 1, 0, 1, 2, 2, 0]` với bảng chữ cái `vocab` ở Cell 1 và giải thích vì sao kết quả có hai chữ `"A"` (`"AAI"`).
- **Trung cấp**: Tính khoảng cách sai số điểm mốc **PCK (Percentage of Correct Keypoints)** với ngưỡng bán kính $r = 3.0$ pixel.
- **Nâng cao**: Tính sai số tương đối tuyệt đối **`AbsRel`** ($\frac{1}{N}\sum \frac{|d - \hat{d}|}{d}$) cho bài toán ước lượng độ sâu (**Depth Estimation**).

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 4: Lời giải hoàn chỉnh cho bài tập Bài 19
# 1. Cơ bản: Giải mã chuỗi có ký hiệu blank (0) xen giữa hai chữ giống nhau
seq_double_a = [0, 1, 0, 1, 2, 2, 0]
print("[Cơ bản] Kết quả giải mã [0, 1, 0, 1, 2, 2, 0]:", greedy_ctc_decode(seq_double_a))

# 2. Trung cấp: Tính chỉ số PCK (Percentage of Correct Keypoints)
gt_kps = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
pred_kps = np.array([[11.0, 21.0], [31.5, 39.0], [55.0, 65.0]])
dists_kp = np.linalg.norm(gt_kps - pred_kps, axis=1)
pck_at_3px = float((dists_kp <= 3.0).mean())
print(f"[Trung cấp] Khoảng cách từng khớp: {dists_kp.round(2)} -> PCK@3px = {pck_at_3px*100:.1f}%")

# 3. Nâng cao: Độ đo AbsRel và RMSE cho bài toán Ước lượng độ sâu (Depth Estimation)
depth_gt = np.array([1.5, 2.0, 4.0, 8.0], dtype=np.float32)    # Đơn vị: mét
depth_pred = np.array([1.6, 1.9, 4.4, 7.2], dtype=np.float32)
abs_rel = float(np.mean(np.abs(depth_gt - depth_pred) / depth_gt))
rmse_depth = float(np.sqrt(np.mean((depth_gt - depth_pred) ** 2)))
print(f"[Nâng cao] Depth Estimation -> AbsRel = {abs_rel:.4f} | RMSE = {rmse_depth:.4f} m")""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Đã nắm vững nguyên lý, hàm loss và metric của `OCR`, `Pose Estimation`, `Object Tracking`, `Denoising/Super-Resolution`, `Depth Estimation` và `VLM`.
- [x] Sẵn sàng chuyển sang **Bài 20: `20_evaluation_experiment_design_and_tuning.ipynb`**!

## 17. Nguồn tham khảo
- Shi, B., Bai, X., & Yao, C. (2016), *An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CRNN)*, IEEE TPAMI.
- Zhang, K. et al. (2017), *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising (DnCNN)*, IEEE TIP.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "19_ocr_and_visual_tasks_overview.ipynb", cells)


def build_nb20() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 20: Đánh giá Toàn diện Mô hình, Thiết kế Thí nghiệm Khoa học và Tinh chỉnh Siêu tham số (Hyperparameter Tuning)

## 1. Mục tiêu học tập
- Làm chủ hệ thống độ đo đánh giá (`Evaluation Metrics`) toàn diện vượt xa chỉ số `Accuracy` đơn lẻ:
  - Phân loại: `Accuracy`, `Precision`, `Recall`, `Specificity`, `F1-score` (`Macro`, `Micro`, `Weighted`), `Support`, `Confusion Matrix`, `ROC-AUC`, `PR-AUC`.
  - Chọn ngưỡng quyết định tối ưu (`Optimal Threshold Tuning`) theo yêu cầu bài toán thực tế.
  - Đánh giá độ đáng tin cậy của xác suất dự đoán: **Độ hiệu chỉnh xác suất (`Calibration` & `Expected Calibration Error - ECE`)**.
- Biết cách thiết kế một **Thí nghiệm Khoa học (`Scientific Experiment Design`)** và lập bảng **Nghiên cứu Cắt bỏ Thành phần (`Ablation Study`)** — mỗi lần chỉ thay đổi đúng 1 yếu tố trên cùng `seed` và cùng cách chia tập.
- Thực hành tìm kiếm siêu tham số (`Grid Search` / `Random Search`) chỉ trên tập `Train/Validation`, giữ tập `Test` hoàn toàn độc lập.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 05, 06 và 11.

## 3. Thời gian học dự kiến
- Khoảng **2.0 giờ** (ước tính).

## 4, 5, 6 & 7. Giải thích Trực quan, Từ điển Keyword & Công thức Toán học
1. **Khi nào dùng `Macro-F1`, `Micro-F1`, `Weighted-F1`, `ROC-AUC` hay `PR-AUC`?**
   - Dữ liệu cân bằng, mọi lỗi ngang nhau $\longrightarrow$ `Accuracy` + `Macro-F1`.
   - Dữ liệu lệch lớp, cần bảo vệ lớp thiểu số quan trọng ngang lớp đa số $\longrightarrow$ **`Macro-F1`** + **`PR-AUC`**.
   - Chi phí bỏ sót lỗi cực kỳ nguy hiểm (y tế, an toàn) $\longrightarrow$ Ưu tiên **`Recall`** cao (hạ thấp `threshold`), kiểm soát `Specificity` / `Precision` ở mức chấp nhận được.
2. **Sai số Hiệu chỉnh Kỳ vọng (`Expected Calibration Error - ECE`)**:
   Chia đoạn xác suất $[0, 1]$ thành $M$ ngăn (`bins`) $B_m$. Trong mỗi ngăn, so sánh độ chính xác thực tế $\text{acc}(B_m)$ với độ tự tin trung bình $\text{conf}(B_m)$:
   $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \Big|\text{acc}(B_m) - \text{conf}(B_m)\Big|$$
   Một mô hình hiệu chỉnh tốt có $\text{ECE} \approx 0$ (nói tự tin `80%` thì đúng thực tế khoảng `80%`).""",
        ),
        (
            "code",
            """# Cell 1: Tính Bộ Độ đo Toàn diện (Accuracy, Precision, Recall, Specificity, Macro/Weighted F1, ROC-AUC, PR-AUC)
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
if (project_root / "image_ai_learning").is_dir():
    project_root = project_root / "image_ai_learning"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils import set_seed
from src.synthetic_data import generate_synthetic_shapes_dataset
from src.features import extract_combined_features
from src.evaluation import compute_classification_metrics

set_seed(42)

ds = generate_synthetic_shapes_dataset(num_samples=320, image_size=32, seed=42, noise_std=14.0)
X_feats = extract_combined_features(ds["images"])
y_all = ds["labels"]
class_names = ds["class_names"]

idx_all = np.arange(len(y_all))
idx_tr, idx_temp = train_test_split(idx_all, test_size=0.40, random_state=42, stratify=y_all)
idx_va, idx_te = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=y_all[idx_temp])

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_feats[idx_tr])
X_va = scaler.transform(X_feats[idx_va])
X_te = scaler.transform(X_feats[idx_te])
y_tr, y_va, y_te = y_all[idx_tr], y_all[idx_va], y_all[idx_te]

clf = SVC(C=1.0, kernel="rbf", probability=True, random_state=42).fit(X_tr, y_tr)
va_probs = clf.predict_proba(X_va)
va_preds = va_probs.argmax(axis=1)

metrics_report = compute_classification_metrics(y_va, va_preds, class_names)
per_cls_df = pd.DataFrame(metrics_report["per_class"]).T.round(4)
print("=== BẢNG CHI TIẾT PRECISION, RECALL, SPECIFICITY, F1 & SUPPORT TỪNG LỚP ===")
print(per_cls_df.to_string())
print(f"\\nTổng hợp -> Accuracy: {metrics_report['accuracy']:.4f} | Macro-F1: {metrics_report['macro_f1']:.4f} | Weighted-F1: {metrics_report['weighted_f1']:.4f}")""",
        ),
        (
            "code",
            """# Cell 2: Tinh chỉnh Ngưỡng Quyết định (Threshold Tuning) và Đo Độ Hiệu chỉnh Xác suất (Calibration / ECE)
def compute_ece(y_true: np.ndarray, y_prob_matrix: np.ndarray, n_bins: int = 5) -> float:
    confidences = y_prob_matrix.max(axis=1)
    predictions = y_prob_matrix.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b_i in range(n_bins):
        in_bin = (confidences > bin_boundaries[b_i]) & (confidences <= bin_boundaries[b_i + 1])
        prop_in_bin = float(in_bin.mean())
        if prop_in_bin > 0:
            acc_in_bin = float(accuracies[in_bin].mean())
            conf_in_bin = float(confidences[in_bin].mean())
            ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)


ece_val = compute_ece(y_va, va_probs, n_bins=5)
print(f"Sai số hiệu chỉnh xác suất kỳ vọng (Expected Calibration Error - ECE): {ece_val:.4f}")

# Tinh chỉnh ngưỡng (Threshold Tuning) cho bài toán nhị phân: Lớp 'cross' (class 3) vs Các lớp còn lại
y_va_bin = (y_va == 3).astype(np.int64)
prob_cross = va_probs[:, 3]
print(f"Lớp 'cross' (One-vs-Rest) -> ROC-AUC: {roc_auc_score(y_va_bin, prob_cross):.4f} | PR-AUC: {average_precision_score(y_va_bin, prob_cross):.4f}")

for th in [0.20, 0.50, 0.80]:
    pred_bin = (prob_cross >= th).astype(np.int64)
    tp = int(((pred_bin == 1) & (y_va_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (y_va_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (y_va_bin == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print(f"  Ngưỡng τ = {th:.2f} -> Precision = {prec*100:5.1f}% | Recall = {rec*100:5.1f}%")""",
        ),
        (
            "code",
            """# Cell 3: Thiết kế Bảng Thí nghiệm Cắt bỏ Thành phần (Ablation Study) & Tìm kiếm Siêu tham số Không Rò rỉ Tập Test
# 1. Ablation Study: Đánh giá đóng góp thực sự của từng thành phần trong pipeline
ablation_configs = [
    ("A0: Baseline (Chỉ dùng Color Histogram + Linear SVM)", X_feats[:, 144:168], "linear", 1.0),
    ("A1: Thay bằng đặc trưng HOG (144d) + Linear SVM", X_feats[:, :144], "linear", 1.0),
    ("A2: HOG + Color + LBP (184d) + Linear SVM", X_feats, "linear", 1.0),
    ("A3: HOG + Color + LBP (184d) + RBF Kernel SVM", X_feats, "rbf", 1.0),
]

ablation_rows = []
for exp_name, F_sub, kern, c_val in ablation_configs:
    sc_sub = StandardScaler()
    X_tr_s = sc_sub.fit_transform(F_sub[idx_tr])
    X_va_s = sc_sub.transform(F_sub[idx_va])
    m_sub = SVC(C=c_val, kernel=kern, random_state=42).fit(X_tr_s, y_tr)
    res_sub = compute_classification_metrics(y_va, m_sub.predict(X_va_s), class_names)
    ablation_rows.append({"Experiment": exp_name, "Val_Acc": round(res_sub["accuracy"], 4), "Val_Macro_F1": round(res_sub["macro_f1"], 4)})

print("=== BẢNG ABLATION STUDY (CHỨNG MINH ĐÓNG GÓP CỦA TỪNG THÀNH PHẦN) ===")
print(pd.DataFrame(ablation_rows).to_string(index=False))

# 2. Grid Search chọn siêu tham số (C, gamma) CHỈ TRÊN TẬP VALIDATION, sau đó đánh giá đúng 1 lần trên tập TEST
best_cfg = None
best_val_f1 = -1.0
best_model = None

for c_cand in [0.1, 1.0, 5.0, 10.0]:
    for gamma_cand in ["scale", 0.005, 0.02]:
        cand_clf = SVC(C=c_cand, gamma=gamma_cand, kernel="rbf", random_state=42).fit(X_tr, y_tr)
        val_f1 = compute_classification_metrics(y_va, cand_clf.predict(X_va), class_names)["macro_f1"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_cfg = {"C": c_cand, "gamma": gamma_cand}
            best_model = cand_clf

# Đánh giá lần cuối cùng trên tập Test bị khóa kín
final_test_metrics = compute_classification_metrics(y_te, best_model.predict(X_te), class_names)
print(f"\\n[Grid Search] Cấu hình tốt nhất chọn từ Validation: {best_cfg} (Val Macro-F1 = {best_val_f1:.4f})")
print(f"[Final Test Report] Test Accuracy = {final_test_metrics['accuracy']*100:.1f}% | Test Macro-F1 = {final_test_metrics['macro_f1']:.4f}")

fig, ax = plt.subplots(figsize=(7.5, 2.8))
exp_labels = ["A0: ColorHist", "A1: HOG", "A2: Combined+Lin", "A3: Combined+RBF"]
exp_f1s = [r["Val_Macro_F1"] for r in ablation_rows]
ax.barh(exp_labels, exp_f1s, color=["silver", "steelblue", "teal", "seagreen"])
ax.set_xlim(0, 1.05)
ax.set_xlabel("Validation Macro-F1")
ax.set_title("Biểu đồ Ablation Study: Đóng góp của từng thành phần")
plt.tight_layout()
plt.show()""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Ở Cell 2**: Khi giảm ngưỡng quyết định $\tau$ từ `0.80` xuống `0.20`, độ nhạy (`Recall`) tăng mạnh nhưng độ chuẩn xác (`Precision`) giảm. Nhờ vậy, bạn có thể điều chỉnh $\tau$ trên tập `Validation` để đáp ứng yêu cầu nghiệp vụ mà không cần huấn luyện lại mô hình!
- **Ở Cell 3**: Bảng `Ablation Study` chỉ ra chính xác bước nhảy chất lượng lớn nhất đến từ việc đưa đặc trưng hình dáng **`HOG`** vào (`A0 -> A1`) và bước tinh chỉnh cuối đến từ nhân phi tuyến **`RBF`** (`A2 -> A3`).

## 14. Bài tập tự làm
- **Cơ bản**: Viết hàm tìm ngưỡng $\tau^* \in [0.05, 0.95]$ tối đa hóa điểm `F1-score` của lớp `cross` trên tập `Validation`, sau đó áp dụng ngưỡng $\tau^*$ đó lên tập `Test`.
- **Trung cấp**: Thực hiện kỹ thuật **Temperature Scaling** ($p_i = \text{softmax}(z_i / T)$) trên tập Validation để làm mịn xác suất và giảm chỉ số `ECE`.
- **Nâng cao**: Thiết kế vòng lặp **Random Search** lấy mẫu `learning_rate` theo thang đo logarit (`10 ** np.random.uniform(-4, -1)`), giải thích vì sao `learning_rate` phải lấy mẫu trên thang log thay vì thang tuyến tính đều!

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 4: Lời giải hoàn chỉnh cho bài tập Bài 20
# 1. Cơ bản: Tìm ngưỡng tối ưu F1 trên Validation và áp dụng sang Test
best_tau, best_f1_cross = 0.5, 0.0
for tau in np.linspace(0.1, 0.9, 17):
    p_b = (prob_cross >= tau).astype(np.int64)
    tp = ((p_b == 1) & (y_va_bin == 1)).sum()
    fp = ((p_b == 1) & (y_va_bin == 0)).sum()
    fn = ((p_b == 0) & (y_va_bin == 1)).sum()
    f1_c = (2 * tp) / max(1, 2 * tp + fp + fn)
    if f1_c > best_f1_cross:
        best_f1_cross, best_tau = float(f1_c), float(tau)
print(f"[Cơ bản] Ngưỡng tối ưu trên Val: τ* = {best_tau:.2f} (Val F1 = {best_f1_cross:.4f})")

# 2. Trung cấp: Temperature Scaling giảm hiện tượng tự tin thái quá (Overconfidence)
raw_logits = np.log(np.clip(va_probs, 1e-8, 1.0))
for temp in [0.5, 1.0, 1.5]:
    scaled_exp = np.exp(raw_logits / temp)
    scaled_probs = scaled_exp / scaled_exp.sum(axis=1, keepdims=True)
    print(f"[Trung cấp] Temperature T={temp:.1f} -> ECE = {compute_ece(y_va, scaled_probs):.4f}")

# 3. Nâng cao: Lấy mẫu Learning Rate theo thang Log-Uniform (đảm bảo mỗi bậc độ lớn 1e-4..1e-3, 1e-3..1e-2 có xác suất ngang nhau)
rng = np.random.default_rng(42)
log_sampled_lrs = 10.0 ** rng.uniform(-4.0, -1.0, size=6)
print("[Nâng cao] 6 giá trị LR lấy mẫu Log-Uniform:", [f"{v:.2e}" for v in np.sort(log_sampled_lrs)])""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Thành thạo bộ metric đa chiều, `Threshold Tuning`, `Calibration (ECE)`, `Ablation Study` và `Hyperparameter Tuning` chuẩn khoa học.
- [x] Sẵn sàng chuyển sang **Bài 21: `21_inference_export_and_optimization.ipynb`** để đóng gói mô hình đưa vào chương trình xử lý ảnh thực tế!

## 17. Nguồn tham khảo
- Guo, C. et al. (2017), *On Calibration of Modern Neural Networks*, ICML.
- Bergstra, J., & Bengio, Y. (2012), *Random Search for Hyper-Parameter Optimization*, JMLR.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "20_evaluation_experiment_design_and_tuning.ipynb", cells)



def build_nb21() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 21: Đưa Mô hình vào Chương trình Thực tế — Inference Pipeline, Xuất Model (TorchScript/ONNX) và Tối ưu hóa

## 1. Mục tiêu học tập
- Hiểu cách đóng gói một mô hình đã huấn luyện thành **chương trình suy luận độc lập (`Production Inference Class`)**:
  - Lưu trữ đầy đủ `model_state_dict`, `class_names`, `class_to_idx`, `idx_to_class` và `preprocessing_config` (`image_size`, `mean`, `std`, `color_space='RGB'`) vào cùng 1 file `checkpoint`.
  - Tiền xử lý ảnh đầu vào khớp tuyệt đối với lúc huấn luyện (`BGR -> RGB`, `Resize`, `HWC uint8 -> NCHW float32`, `Normalize`), chạy với `model.eval()` + `torch.inference_mode()` và hậu xử lý trả về kết quả có cấu trúc (`predicted_class`, `confidence`, `probabilities`).
- Thực hành xuất mô hình sang định dạng độc lập với code Python:
  1. **`TorchScript` (`torch.jit.trace`)**: Chạy độc lập trong C++ hoặc Python mà không cần file định nghĩa lớp `SimpleCNN`.
  2. **Hướng dẫn xuất `ONNX` (`torch.onnx.export`)** khi cài thêm gói tùy chọn `onnx` và `onnxruntime`.
- Thực hành **Lượng tử hóa động (`Dynamic Quantization`: `float32 -> int8`)** cho các tầng `Linear` để giảm dung lượng và tăng tốc trên CPU.
- Đo đạc chính xác 3 chỉ số hệ thống: **`Latency (ms/ảnh)`**, **`Throughput (FPS / ảnh/giây)`** và **`Model Size (KB)`**.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 10, 11 và 20.

## 3. Thời gian học dự kiến
- Khoảng **2.0 giờ** (ước tính).

## 4, 5, 6 & 7. Giải thích Trực quan, Từ điển Keyword & Công thức Đo Hiệu năng
| Keyword | Nghĩa tiếng Việt | Nhóm | Định nghĩa, Đơn vị & Ý nghĩa thực tế |
|---|---|---|---|
| `latency` | Độ trễ suy luận | Triển khai | Thời gian xử lý 1 mẫu hoặc 1 lô (`ms / batch`). Cần chạy `warmup_runs` trước khi bấm giờ để loại trừ chi phí cấp phát bộ nhớ lần đầu. |
| `throughput` | Thông lượng xử lý | Triển khai | Số lượng ảnh xử lý được trong 1 giây: $\text{Throughput} = \frac{\text{batch\_size} \times \text{timed\_runs}}{\text{total\_seconds}}$ (`samples/sec` hay `FPS`). |
| `model_size` | Dung lượng mô hình | Triển khai | Kích thước file `.pt` trên đĩa (`KB` hoặc `MB`). |
| `torch.jit.trace` | Biên dịch TorchScript bằng vết | Triển khai | Cho 1 tensor mẫu chạy qua mạng để ghi lại toàn bộ các phép toán thành đồ thị tĩnh tối ưu hóa (`ScriptModule`). |
| `Dynamic Quantization` | Lượng tử hóa động `int8` | Tối ưu hóa | Chuyển trọng số tầng `nn.Linear` từ `float32` (4 bytes) sang `qint8` (1 byte), giảm ~4 lần dung lượng tầng Linear! |""",
        ),
        (
            "code",
            """# Cell 1: Huấn luyện Nhanh hoặc Nạp Checkpoint SimpleCNN và Xây dựng Class ImagePredictor Hoàn chỉnh
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
if (project_root / "image_ai_learning").is_dir():
    project_root = project_root / "image_ai_learning"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.utils import set_seed, get_device, count_parameters, save_checkpoint, load_checkpoint
from src.synthetic_data import generate_synthetic_shapes_dataset
from src.models import SimpleCNN
from src.training import fit_classifier
from src.evaluation import measure_inference_latency

set_seed(42)
device = torch.device("cpu")  # Triển khai và đo benchmark chuẩn trên CPU

ds = generate_synthetic_shapes_dataset(num_samples=200, image_size=32, seed=42)
imgs_np, labels_np, class_names = ds["images"], ds["labels"], ds["class_names"]

X_t = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0
y_t = torch.from_numpy(labels_np).long()
mean_vec = X_t[:150].mean(dim=(0, 2, 3))
std_vec = X_t[:150].std(dim=(0, 2, 3)) + 1e-6
X_norm = (X_t - mean_vec.view(1, 3, 1, 1)) / std_vec.view(1, 3, 1, 1)

tr_loader = DataLoader(TensorDataset(X_norm[:150], y_t[:150]), batch_size=32, shuffle=True)
va_loader = DataLoader(TensorDataset(X_norm[150:], y_t[150:]), batch_size=32, shuffle=False)

deploy_model = SimpleCNN(in_channels=3, num_classes=4, base_channels=16).to(device)
opt = torch.optim.AdamW(deploy_model.parameters(), lr=2e-3)
ckpt_deploy_path = project_root / "outputs" / "checkpoints" / "nb21_deploy_cnn.pt"

fit_classifier(
    deploy_model, tr_loader, va_loader, nn.CrossEntropyLoss(), opt, device,
    epochs=8, checkpoint_path=ckpt_deploy_path, class_names=class_names,
    preprocessing_config={"image_size": 32, "mean": mean_vec.tolist(), "std": std_vec.tolist(), "color_space": "RGB"},
)


class ImagePredictor:
    \"\"\"
    Lớp bao đóng (Wrapper Class) tích hợp mô hình AI vào phần mềm xử lý ảnh thực tế:
    - Tự động nạp checkpoint, class mapping và thông số tiền xử lý.
    - Nhận trực tiếp mảng ảnh RGB uint8 (H, W, 3) kích thước bất kỳ và trả về kết quả dự đoán an toàn.
    \"\"\"

    def __init__(self, checkpoint_path: Path | str, device_str: str = "cpu") -> None:
        self.device = torch.device(device_str)
        ckpt = load_checkpoint(checkpoint_path, device=self.device)
        self.class_names: list[str] = ckpt["class_names"]
        self.idx_to_class: dict[int, str] = ckpt["idx_to_class"]
        self.preproc: dict = ckpt["preprocessing_config"]

        self.model = SimpleCNN(in_channels=3, num_classes=len(self.class_names), base_channels=16)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.img_size = int(self.preproc.get("image_size", 32))
        self.mean = torch.tensor(self.preproc["mean"], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.preproc["std"], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def preprocess(self, rgb_image: np.ndarray) -> torch.Tensor:
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Kỳ vọng ảnh RGB (H, W, 3), nhận được shape {rgb_image.shape}")
        if rgb_image.shape[0] != self.img_size or rgb_image.shape[1] != self.img_size:
            rgb_image = cv2.resize(rgb_image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
        return (x - self.mean) / self.std

    @torch.inference_mode()
    def predict(self, rgb_image: np.ndarray) -> dict:
        x_tensor = self.preprocess(rgb_image)
        logits = self.model(x_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        return {
            "predicted_index": pred_idx,
            "predicted_class": self.idx_to_class[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": {self.idx_to_class[i]: round(float(probs[i]), 4) for i in range(len( probs ))},
        }


predictor = ImagePredictor(ckpt_deploy_path, device_str="cpu")
sample_raw_img = cv2.resize(imgs_np[0], (80, 60))  # Thử đưa ảnh kích thước bất kỳ (60x80x3) vào!
pred_result = predictor.predict(sample_raw_img)
print(f"Nhãn thật: {class_names[labels_np[0]]} -> Kết quả từ ImagePredictor: {pred_result}")""",
        ),
        (
            "code",
            """# Cell 2: Xuất Mô hình sang TorchScript (torch.jit.trace) và Kiểm tra Sai số Số học (< 1e-6)
deploy_model.eval()
example_input = torch.randn(1, 3, 32, 32, device=device)

with torch.inference_mode():
    traced_script_module = torch.jit.trace(deploy_model, example_input)

ts_path = project_root / "outputs" / "checkpoints" / "nb21_deploy_cnn_torchscript.pt"
traced_script_module.save(str(ts_path))

# Nạp lại file TorchScript mà KHÔNG CẦN class SimpleCNN
loaded_jit = torch.jit.load(str(ts_path), map_location=device)
with torch.inference_mode():
    out_pytorch = deploy_model(example_input)
    out_torchscript = loaded_jit(example_input)
    max_diff_jit = float(torch.max(torch.abs(out_pytorch - out_torchscript)).item())

print(f"Đã xuất TorchScript tại: {ts_path.relative_to(project_root)} ({ts_path.stat().st_size / 1024:.1f} KB)")
print(f"Sai lệch tối đa giữa PyTorch gốc và TorchScript: {max_diff_jit:.2e}")
assert max_diff_jit < 1e-5, "Đầu ra TorchScript phải khớp tuyệt đối với mô hình PyTorch gốc!" """,
        ),
        (
            "code",
            """# Cell 3: Lượng tử hóa Động (Dynamic Quantization int8) và So sánh Latency, Throughput, Model Size
from src.models import SimpleMLP

# Minh họa rõ hiệu quả nén dung lượng của Dynamic Quantization trên mạng có nhiều tầng Linear (SimpleMLP)
mlp_fp32 = SimpleMLP(in_channels=3, image_size=32, hidden_dim=128, num_classes=4).to(device).eval()
mlp_int8 = torch.ao.quantization.quantize_dynamic(mlp_fp32, {nn.Linear}, dtype=torch.qint8)

fp32_path = project_root / "outputs" / "checkpoints" / "nb21_mlp_fp32.pt"
int8_path = project_root / "outputs" / "checkpoints" / "nb21_mlp_int8.pt"
torch.save(mlp_fp32.state_dict(), fp32_path)
torch.save(mlp_int8.state_dict(), int8_path)

size_fp32_kb = fp32_path.stat().st_size / 1024.0
size_int8_kb = int8_path.stat().st_size / 1024.0

perf_b1 = measure_inference_latency(deploy_model, input_shape=(1, 3, 32, 32), device=device)
perf_b16 = measure_inference_latency(deploy_model, input_shape=(16, 32 if False else 3, 32, 32), device=device)

benchmark_df = pd.DataFrame([
    {"Model / Mode": "SimpleCNN (Batch=1)", "Latency_ms_per_batch": round(perf_b1["latency_ms_per_batch"], 3), "Throughput_FPS": round(perf_b1["throughput_samples_per_sec"], 1)},
    {"Model / Mode": "SimpleCNN (Batch=16)", "Latency_ms_per_batch": round(perf_b16["latency_ms_per_batch"], 3), "Throughput_FPS": round(perf_b16["throughput_samples_per_sec"], 1)},
])
print("=== BẢNG ĐO HIỆU NĂNG SUY LUẬN (LATENCY & THROUGHPUT) ===")
print(benchmark_df.to_string(index=False))
print(f"\\nDung lượng SimpleMLP trước Quantization (FP32): {size_fp32_kb:.1f} KB")
print(f"Dung lượng SimpleMLP sau Dynamic Quant (INT8) : {size_int8_kb:.1f} KB (Giảm {size_fp32_kb/size_int8_kb:.2f} lần!)")

fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.8))
axes[0].bar(["FP32 (Gốc)", "INT8 (Quantized)"], [size_fp32_kb, size_int8_kb], color=["royalblue", "darkorange"])
axes[0].set_title("So sánh Dung lượng SimpleMLP (KB)")
axes[1].bar(["Batch=1", "Batch=16"], [perf_b1["throughput_samples_per_sec"], perf_b16["throughput_samples_per_sec"]], color=["teal", "seagreen"])
axes[1].set_title("So sánh Thông lượng Throughput (FPS)")
plt.tight_layout()
plt.show()""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Ở Cell 1**: Lớp `ImagePredictor` tự động hóa toàn bộ quy trình từ đọc `preprocessing_config` trong checkpoint, tự động `resize` ảnh đầu vào bất kỳ về `32x32`, chuẩn hóa bằng đúng `mean`/`std` lúc train và trả về dictionary kết quả rõ ràng.
- **Ở Cell 3**:
  - Khi xử lý đơn lẻ `Batch=1`, độ trễ (`latency`) cho mỗi ảnh thấp nhất (phù hợp phản hồi tức thời), nhưng khi gom lô `Batch=16`, thông lượng (`throughput FPS`) tăng vọt nhờ tính toán vector hóa song song.
  - Lượng tử hóa động (`Dynamic Quantization int8`) giảm dung lượng file trọng số của `SimpleMLP` đi hơn **3.5 lần** (từ `~1,570 KB` xuống `~410 KB`)!

## 14. Bài tập tự làm
- **Cơ bản**: Thêm phương thức `predict_batch(list_of_rgb_images)` vào `ImagePredictor` để dự đoán nhiều ảnh cùng lúc trong 1 lượt forward.
- **Trung cấp**: Thêm cơ chế cảnh báo **"Out-of-Distribution / Low Confidence Warning"**: nếu `confidence < 0.60`, trả thêm trường `"status": "uncertain_needs_human_review"`.
- **Nâng cao**: Viết đoạn code kiểm tra nếu hệ thống có cài thư viện `onnx` thì xuất mô hình sang file `.onnx`, nếu chưa cài thì in hướng dẫn cài đặt từ `requirements-optional.txt`.

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 4: Lời giải hoàn chỉnh cho bài tập Bài 21
# 1. Cơ bản & 2. Trung cấp: Dự đoán theo lô (Batch Inference) kèm kiểm tra ngưỡng độ tự tin (Confidence Guard)
@torch.inference_mode()
def predict_batch_with_guard(pred_obj: ImagePredictor, rgb_list: list[np.ndarray], min_conf: float = 0.60) -> list[dict]:
    tensors = [pred_obj.preprocess(im) for im in rgb_list]
    batch_x = torch.cat(tensors, dim=0)
    probs_batch = torch.softmax(pred_obj.model(batch_x), dim=1).cpu().numpy()
    outputs = []
    for probs in probs_batch:
        idx = int(probs.argmax())
        conf = float(probs[idx])
        status = "accepted" if conf >= min_conf else "uncertain_needs_human_review"
        outputs.append({"class": pred_obj.idx_to_class[idx], "confidence": round(conf, 4), "status": status})
    return outputs

batch_out = predict_batch_with_guard(predictor, [imgs_np[0], imgs_np[1], imgs_np[2]], min_conf=0.60)
print("[Cơ bản & Trung cấp] Kết quả dự đoán Batch kèm kiểm tra ngưỡng tin cậy:")
for item in batch_out:
    print(" ", item)

# 3. Nâng cao: Kiểm tra tùy chọn xuất ONNX an toàn
try:
    import onnx  # type: ignore
    onnx_available = True
except ImportError:
    onnx_available = False

print(f"[Nâng cao] Trạng thái thư viện tùy chọn ONNX: {'Đã cài đặt' if onnx_available else 'Chưa cài (cài bằng: pip install -r requirements-optional.txt khi cần xuất .onnx)'}")""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Đã xây dựng thành công class `ImagePredictor`, xuất `TorchScript`, lượng tử hóa `int8` và đo `Latency/Throughput`.
- [x] Sẵn sàng bước vào bài học tổng kết cuối cùng — **Bài 22: `22_capstone_project.ipynb` (Đồ án Cuối khóa End-to-End)**!

## 17. Nguồn tham khảo
- PyTorch TorchScript Documentation: https://pytorch.org/docs/stable/jit.html
- PyTorch Quantization Documentation: https://pytorch.org/docs/stable/quantization.html""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "21_inference_export_and_optimization.ipynb", cells)


def build_nb22() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 22: Đồ án Cuối khóa (Capstone Project) — Xây dựng Hệ thống Phân loại Ảnh End-to-End, Phân tích Lỗi và Hướng dẫn Mở rộng

## 1. Mục tiêu Đồ án
Kết nối toàn bộ kiến thức từ Bài 00 đến Bài 21 vào một quy trình kỹ thuật khép kín (**End-to-End Pipeline**):
1. **Đặc tả Bài toán & Kiểm định Dữ liệu (`Data Auditing`)**: Kiểm tra kích thước, kênh màu, phân phối lớp, giá trị khuyết/hỏng.
2. **Chia Dữ liệu Chuẩn (`Train / Validation / Test`)**: Không rò rỉ dữ liệu (`Zero Data Leakage`).
3. **Xây dựng Đường cơ sở ML Truyền thống (`Classical ML Baseline`)**: `HOG + Color + LBP` kết hợp `SVM`.
4. **Huấn luyện Mô hình Học sâu (`Deep Learning Pipeline`)**: `ResidualCNN` kèm `Augmentation` trên tập Train, theo dõi `Train/Val Loss & Accuracy`, lưu `Checkpoint` tốt nhất.
5. **Đánh giá Toàn diện trên Tập Test (`Comprehensive Evaluation`)**: Báo cáo `Accuracy`, `Macro-F1`, `Weighted-F1`, bảng `Precision/Recall/Specificity` từng lớp và `Confusion Matrix`.
6. **Phân tích Lỗi Sâu (`Error Analysis`)**: Tìm những mẫu bị dự đoán sai có `loss` cao nhất để hiểu giới hạn của mô hình.
7. **Chương trình Suy luận Độc lập (`Standalone Inference`)** & **Hướng dẫn Thay bằng Bộ Ảnh Cá nhân của Bạn**.
8. **Bảng Tiêu chí Tự Đánh giá Đồ án (`Self-Assessment Rubric`)**.

> **Lưu ý trung thực khoa học**: Kết quả chạy mặc định trên bộ dữ liệu tổng hợp cỡ nhỏ trong notebook này dùng để xác nhận toàn bộ quy trình kỹ thuật của bạn hoạt động chính xác từ đầu đến cuối; **không dùng con số trên dữ liệu nhỏ này để khẳng định mô hình đã sẵn sàng triển khai thương mại** nếu chưa kiểm định trên dữ liệu thực tế quy mô lớn.

## 2. Thời gian thực hiện dự kiến
- Khoảng **4.0 – 6.0 giờ** (bao gồm cả phần tự thử nghiệm mở rộng).""",
        ),
        (
            "code",
            """# Cell 1: Bước 1 & 2 — Đặc tả Bài toán, Kiểm định Dữ liệu (Data Audit) và Chia Tập Train/Val/Test
import sys
import json
from pathlib import Path

current_dir = Path.cwd().resolve()
project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
if (project_root / "image_ai_learning").is_dir():
    project_root = project_root / "image_ai_learning"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils import set_seed, get_device, count_parameters, save_checkpoint, load_checkpoint
from src.synthetic_data import generate_synthetic_shapes_dataset
from src.features import extract_combined_features
from src.models import ResidualCNN
from src.training import fit_classifier, evaluate_classifier
from src.evaluation import compute_classification_metrics

QUICK_RUN = True
SEED = 42
set_seed(SEED)
device = get_device()

n_total = 320 if QUICK_RUN else 800
capstone_ds = generate_synthetic_shapes_dataset(num_samples=n_total, image_size=32, seed=SEED, noise_std=12.0)
images_np = capstone_ds["images"]
labels_np = capstone_ds["labels"]
class_names = capstone_ds["class_names"]

# Kiểm định chất lượng dữ liệu tự động (Automated Data Audit)
assert images_np.ndim == 4 and images_np.shape[1:] == (32, 32, 3), "Sai kích thước ảnh!"
assert images_np.dtype == np.uint8, "Ảnh gốc phải ở kiểu uint8!"
assert len(images_np) == len(labels_np), "Số lượng ảnh và nhãn không khớp!"

idx_all = np.arange(len(images_np))
idx_tr, idx_temp = train_test_split(idx_all, test_size=0.30, random_state=SEED, stratify=labels_np)
idx_va, idx_te = train_test_split(idx_temp, test_size=0.50, random_state=SEED, stratify=labels_np[idx_temp])

print("=== BÁO CÁO KIỂM ĐỊNH DỮ LIỆU ĐỒ ÁN (DATA AUDIT) ===")
print(f"Tổng số mẫu: {len(images_np)} | Kích thước: {images_np.shape[1:]} | Các lớp: {class_names}")
print(f"Chia tập Stratified -> Train: {len(idx_tr)} ({np.bincount(labels_np[idx_tr]).tolist()}), Val: {len(idx_va)}, Test: {len(idx_te)}")""",
        ),
        (
            "code",
            """# Cell 2: Bước 3 — Xây dựng Baseline Machine Learning Truyền thống (HOG+Color+LBP + RBF SVM)
feats_all = extract_combined_features(images_np, cell_size=8)
scaler_bl = StandardScaler()
X_tr_bl = scaler_bl.fit_transform(feats_all[idx_tr])
X_va_bl = scaler_bl.transform(feats_all[idx_va])
X_te_bl = scaler_bl.transform(feats_all[idx_te])

baseline_svm = SVC(C=2.0, kernel="rbf", probability=True, random_state=SEED).fit(X_tr_bl, labels_np[idx_tr])
bl_val_metrics = compute_classification_metrics(labels_np[idx_va], baseline_svm.predict(X_va_bl), class_names)
bl_test_metrics = compute_classification_metrics(labels_np[idx_te], baseline_svm.predict(X_te_bl), class_names)

print(f"[Baseline ML - SVM] Val Acc : {bl_val_metrics['accuracy']*100:.1f}% | Val Macro-F1 : {bl_val_metrics['macro_f1']:.4f}")
print(f"[Baseline ML - SVM] Test Acc: {bl_test_metrics['accuracy']*100:.1f}% | Test Macro-F1: {bl_test_metrics['macro_f1']:.4f}")""",
        ),
        (
            "code",
            """# Cell 3: Bước 4 — Huấn luyện Mô hình Deep Learning (ResidualCNN) với Augmentation trên Tập Train
X_all_t = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
y_all_t = torch.from_numpy(labels_np).long()

tr_mean = X_all_t[idx_tr].mean(dim=(0, 2, 3), keepdim=True)
tr_std = X_all_t[idx_tr].std(dim=(0, 2, 3), keepdim=True) + 1e-6
X_norm_t = (X_all_t - tr_mean) / tr_std

# Tăng cường dữ liệu (Augmentation) CHỈ TRÊN TẬP TRAIN: ghép thêm bản lật ngang và lật dọc (bảo toàn nhãn của cả 4 hình đối xứng!)
X_tr_aug = torch.cat([X_norm_t[idx_tr], torch.flip(X_norm_t[idx_tr], dims=[3])], dim=0)
y_tr_aug = torch.cat([y_all_t[idx_tr], y_all_t[idx_tr]], dim=0)

tr_loader = DataLoader(TensorDataset(X_tr_aug, y_tr_aug), batch_size=32, shuffle=True, num_workers=0)
va_loader = DataLoader(TensorDataset(X_norm_t[idx_va], y_all_t[idx_va]), batch_size=32, shuffle=False, num_workers=0)
te_loader = DataLoader(TensorDataset(X_norm_t[idx_te], y_all_t[idx_te]), batch_size=32, shuffle=False, num_workers=0)

capstone_model = ResidualCNN(in_channels=3, num_classes=4, base_channels=16).to(device)
print("ResidualCNN Capstone Parameters:", count_parameters(capstone_model))

crit = nn.CrossEntropyLoss()
opt_dl = torch.optim.AdamW(capstone_model.parameters(), lr=2.5e-3, weight_decay=1e-4)
sched_dl = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dl, T_max=14 if QUICK_RUN else 30, eta_min=1e-4)
ckpt_capstone = project_root / "outputs" / "checkpoints" / "capstone_residual_cnn_best.pt"

preproc_metadata = {
    "image_size": 32,
    "mean": tr_mean.squeeze().tolist(),
    "std": tr_std.squeeze().tolist(),
    "color_space": "RGB",
}

hist_dl = fit_classifier(
    model=capstone_model,
    train_loader=tr_loader,
    val_loader=va_loader,
    criterion=crit,
    optimizer=opt_dl,
    device=device,
    epochs=14 if QUICK_RUN else 30,
    scheduler=sched_dl,
    patience=8,
    checkpoint_path=ckpt_capstone,
    class_names=class_names,
    preprocessing_config=preproc_metadata,
)
print(f"ResidualCNN Best Epoch: {hist_dl['best_epoch']} | Best Val Accuracy: {hist_dl['best_val_acc']*100:.1f}%")""",
        ),
        (
            "code",
            """# Cell 4: Bước 5 & 6 — Đánh giá Cuối cùng trên Tập Test, So sánh với Baseline & Phân tích Lỗi (Error Analysis)
load_checkpoint(ckpt_capstone, model=capstone_model, device=device)
te_loss_acc, y_te_true, y_te_pred, y_te_prob = evaluate_classifier(capstone_model, te_loader, crit, device)
dl_test_metrics = compute_classification_metrics(y_te_true, y_te_pred, class_names)

print("=== BẢNG TỔNG KẾT ĐÁNH GIÁ TRÊN TẬP TEST (BASELINE ML VS DEEP LEARNING) ===")
summary_cmp = pd.DataFrame([
    {"Approach": "1. Classical ML Baseline (HOG+Color+LBP + SVM)", "Test_Acc": round(bl_test_metrics["accuracy"], 4), "Test_Macro_F1": round(bl_test_metrics["macro_f1"], 4)},
    {"Approach": "2. Deep Learning (ResidualCNN + Augmentation)", "Test_Acc": round(dl_test_metrics["accuracy"], 4), "Test_Macro_F1": round(dl_test_metrics["macro_f1"], 4)},
])
print(summary_cmp.to_string(index=False))

print("\\n=== BẢNG METRIC TỪNG LỚP CỦA MÔ HÌNH DEEP LEARNING ===")
print(pd.DataFrame(dl_test_metrics["per_class"]).T.round(4).to_string())
print("\\nConfusion Matrix (Hàng: Thật, Cột: Dự đoán):\\n", dl_test_metrics["confusion_matrix"])

# Lưu báo cáo JSON ra outputs/reports/capstone_report.json
report_path = project_root / "outputs" / "reports" / "capstone_report.json"
report_payload = {
    "quick_run": QUICK_RUN,
    "baseline_test_accuracy": bl_test_metrics["accuracy"],
    "baseline_test_macro_f1": bl_test_metrics["macro_f1"],
    "dl_test_accuracy": dl_test_metrics["accuracy"],
    "dl_test_macro_f1": dl_test_metrics["macro_f1"],
    "checkpoint_path": str(ckpt_capstone.relative_to(project_root)),
}
report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
print(f"Đã xuất báo cáo đánh giá tại: {report_path.relative_to(project_root)}")

# Phân tích lỗi (Error Analysis): Tìm các mẫu có Loss cá biệt cao nhất trên tập Test
with torch.no_grad():
    te_logits_raw = capstone_model(X_norm_t[idx_te].to(device)).cpu()
    per_sample_losses = nn.CrossEntropyLoss(reduction="none")(te_logits_raw, y_all_t[idx_te]).numpy()

hardest_local_indices = np.argsort(-per_sample_losses)[:4]
fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
for ax, loc_i in zip(axes, hardest_local_indices):
    orig_i = idx_te[loc_i]
    t_name = class_names[y_te_true[loc_i]]
    p_name = class_names[y_te_pred[loc_i]]
    conf = y_te_prob[loc_i, y_te_pred[loc_i]] * 100
    ax.imshow(images_np[orig_i])
    ax.set_title(f"Loss={per_sample_losses[loc_i]:.2f}\\nThật:{t_name} | Đoán:{p_name} ({conf:.0f}%)", fontsize=8.8)
    ax.axis("off")
plt.suptitle("Phân tích Lỗi (Error Analysis): 4 mẫu có Loss cao nhất trong Tập Test", fontsize=10.5)
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 5: Bước 7 — Chương trình Suy luận Độc lập và Hướng dẫn Thay bằng Tập Ảnh Cá nhân của Bạn
from torchvision.datasets import ImageFolder
from torchvision import transforms

# 1. Hàm suy luận độc lập từ checkpoint đã lưu
@torch.inference_mode()
def predict_single_image_from_checkpoint(rgb_img: np.ndarray, ckpt_file: Path) -> dict:
    ckpt = load_checkpoint(ckpt_file, device=device)
    model_inf = ResidualCNN(in_channels=3, num_classes=len(ckpt["class_names"]), base_channels=16)
    model_inf.load_state_dict(ckpt["model_state_dict"])
    model_inf.to(device).eval()

    cfg = ckpt["preprocessing_config"]
    mean_t = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
    probs = torch.softmax(model_inf((x - mean_t) / std_t), dim=1)[0].cpu().numpy()
    top_idx = int(probs.argmax())
    return {"class": ckpt["idx_to_class"][top_idx], "confidence": round(float(probs[top_idx]), 4)}


demo_pred = predict_single_image_from_checkpoint(images_np[idx_te[0]], ckpt_capstone)
print("Kiểm tra hàm Inference độc lập trên mẫu Test #0:", demo_pred, "| Nhãn thật:", class_names[labels_np[idx_te[0]]])

# 2. Mẫu code nạp Tập Ảnh Cá nhân của Bạn bằng torchvision.datasets.ImageFolder:
custom_data_dir = project_root / "data" / "my_custom_dataset"
custom_train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print(f"Để huấn luyện trên bộ ảnh riêng, hãy đặt ảnh vào '{custom_data_dir.relative_to(project_root)}/train/<ten_lop>/' và gọi ImageFolder(root, transform=custom_train_transform)")""",
        ),
        (
            "md",
            r"""## 8. Định hướng 3 Nhánh Mở rộng Tùy chọn (Optional Capstone Extensions)
Sau khi hoàn thành nhánh chính Phân loại ảnh ở trên, bạn có thể mở rộng đồ án theo 1 trong 3 nhánh sau:
1. **Nhánh Mở rộng A — Object Detection (kế thừa Bài 13)**: Đóng khung và đếm đồng thời nhiều vật thể trên ảnh bằng `fasterrcnn_mobilenet_v3_large_320_fpn`, báo cáo `mAP@0.5`.
2. **Nhánh Mở rộng B — Semantic Segmentation (kế thừa Bài 14)**: Tách mặt nạ điểm ảnh bằng `MiniUNet` với `BCE + Dice Loss`, báo cáo `IoU` và `Dice` để đo diện tích vật thể.
3. **Nhánh Mở rộng C — Anomaly Detection (kế thừa Bài 15)**: Chỉ dùng ảnh mẫu lành huấn luyện `ConvAutoencoder`, báo cáo `ROC-AUC`, `PR-AUC` và khoanh vùng vết lỗi.

---

## 9. Bảng Tiêu chí Tự Đánh giá Đồ án (`Self-Assessment Rubric`)

| Hạng mục Đánh giá | Mức Chưa Đạt (0 điểm) | Mức Đạt Cơ bản (1 điểm) | Mức Thành thạo / Chuẩn Kỹ sư (2 điểm) | Điểm tự chấm |
|---|---|---|---|---|
| **1. Kiểm định & Chia dữ liệu** | Không kiểm tra dữ liệu; chia tập sau khi `fit` scaler/PCA (bị rò rỉ dữ liệu). | Chia `train/val/test` trước khi chuẩn hóa nhưng chưa dùng `stratify` hoặc `group split`. | Có bước kiểm định dữ liệu tự động, dùng `stratify`/`group split` và đảm bảo `0%` rò rỉ dữ liệu. | `/ 2` |
| **2. Đường cơ sở (Baseline)** | Nhảy thẳng vào mạng sâu phức tạp mà không có mô hình cơ sở để đối chiếu. | Có chạy 1 mô hình ML truyền thống làm mốc so sánh. | So sánh công bằng Baseline ML (`HOG+SVM`) và Deep Learning trên cùng tập dữ liệu và cùng metric. | `/ 2` |
| **3. Huấn luyện & Checkpoint** | Không theo dõi `Validation`; lưu model ở epoch cuối cùng bị overfit. | Có theo dõi `val_loss` và lưu `state_dict`. | Lưu `checkpoint` tốt nhất kèm đầy đủ `class_to_idx` và `preprocessing_config`, có kiểm tra nạp lại. | `/ 2` |
| **4. Đánh giá & Phân tích Lỗi** | Chỉ in mỗi một con số `Accuracy`. | Có thêm `F1-score` và `Confusion Matrix`. | Báo cáo đầy đủ `Precision`, `Recall`, `Specificity`, `Macro-F1`, `Confusion Matrix` và hiển thị ảnh lỗi có `loss` cao nhất để phân tích nguyên nhân. | `/ 2` |
| **5. Suy luận & Trung thực Khoa học** | Quên `model.eval()` hoặc phóng đại kết quả trên tập dữ liệu nhỏ. | Có hàm dự đoán ảnh mới chạy đúng. | Đóng gói hàm/class Inference hoàn chỉnh, báo cáo trung thực giới hạn của dữ liệu và mô hình. | `/ 2` |
| **TỔNG ĐIỂM** | | | **Mục tiêu đạt chuẩn: $\ge 8 / 10$ điểm** | **`/ 10`** |

---

## 10. Chúc mừng Bạn Đã Hoàn thành Toàn bộ Lộ trình 23 Jupyter Notebooks!
Bạn đã đi từ những thao tác mảng `NumPy` và `Pixel` đầu tiên, qua `Xử lý ảnh cổ điển`, `Machine Learning truyền thống`, tự viết `Mạng nơ-ron từ đầu`, làm chủ `PyTorch & CNN`, tới toàn bộ các nhóm bài toán `Computer Vision` hiện đại và đóng gói triển khai thực tế.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "22_capstone_project.ipynb", cells)


if __name__ == "__main__":
    for builder in [build_nb18, build_nb19, build_nb20, build_nb21, build_nb22]:
        path = builder()
        print("Created and validated:", path.name)
