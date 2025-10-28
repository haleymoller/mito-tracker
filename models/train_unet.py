import os
import glob
import torch
import torch.nn as nn
import numpy as np
import random
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    A = None
    ToTensorV2 = None
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


class EMSet(Dataset):
    def __init__(self, root, aug=False, crop_size=256):
        if A is None or ToTensorV2 is None:
            raise ImportError(
                "Albumentations is required for EMSet. Please install 'albumentations' and 'albumentations[pytorch]'.")
        all_imgs = sorted(glob.glob(os.path.join(root, "images", "*")))
        imgs, masks = [], []
        for p in all_imgs:
            m = p.replace("images", "masks").rsplit(".", 1)[0]+".png"
            if os.path.exists(m):
                imgs.append(p)
                masks.append(m)
        if len(imgs) == 0:
            raise RuntimeError(
                f"No valid image/mask pairs found in {root} (expected masks/*.png for each images/*)")
        self.imgs, self.masks = imgs, masks
        # Keep augmentations modest; ensure mask integrity
        aug_list = [
            A.HorizontalFlip(p=0.5 if aug else 0.0),
            A.VerticalFlip(p=0.5 if aug else 0.0),
            A.RandomRotate90(p=0.2 if aug else 0.0),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10,
                               border_mode=cv2.BORDER_REFLECT_101, p=0.4 if aug else 0.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3 if aug else 0.0),
            A.GaussNoise(var_limit=(5, 15), p=0.3 if aug else 0.0),
            A.GaussianBlur(blur_limit=(3, 3), p=0.2 if aug else 0.0),
        ]
        if crop_size is not None:
            # Use deterministic resize to enforce equal HxW and avoid any mismatch
            aug_list = aug_list + \
                [A.Resize(height=crop_size, width=crop_size,
                          interpolation=cv2.INTER_AREA)]
        aug_list += [A.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)), ToTensorV2()]
        # Disable strict shape checks since we explicitly resize masks to image size
        self.tf = A.Compose(aug_list, is_check_shapes=False)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        im = np.array(Image.open(self.imgs[i]).convert("L"))
        mk_g = np.array(Image.open(self.masks[i]).convert("L"))
        # Ensure image and mask have identical HxW
        if mk_g.shape != im.shape:
            mk_g = cv2.resize(
                mk_g, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        mk = (mk_g > 127).astype(np.float32)
        # Construct 3-channel features: intensity, Sobel edge magnitude, Laplacian-of-Gaussian
        im_f = im.astype(np.float32)
        sobelx = cv2.Sobel(im_f, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(im_f, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        if sobel_mag.max() > 0:
            sobel_mag = sobel_mag / (sobel_mag.max()+1e-6) * 255.0
        blur = cv2.GaussianBlur(im_f, (3, 3), 0)
        lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
        lap_abs = np.abs(lap)
        if lap_abs.max() > 0:
            lap_abs = lap_abs / (lap_abs.max()+1e-6) * 255.0
        im3 = np.stack([im_f, sobel_mag, lap_abs], axis=-1).astype(np.uint8)
        out = self.tf(image=im3, mask=mk)
        x = out["image"]  # [3,H,W]
        y = out["mask"].unsqueeze(0)
        y = (y > 0.5).float()
        return x, y


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1), nn.BatchNorm2d(
                c_out), nn.ReLU(True),
            nn.Conv2d(c_out, c_out, 3, 1, 1), nn.BatchNorm2d(
                c_out), nn.ReLU(True)
        )

    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, c=32, in_channels=3):
        super().__init__()
        self.d1 = ConvBlock(in_channels, c)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = ConvBlock(c, c*2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = ConvBlock(c*2, c*4)
        self.p3 = nn.MaxPool2d(2)
        self.b = ConvBlock(c*4, c*8)
        self.u3 = nn.ConvTranspose2d(c*8, c*4, 2, 2)
        self.c3 = ConvBlock(c*8, c*4)
        self.u2 = nn.ConvTranspose2d(c*4, c*2, 2, 2)
        self.c2 = ConvBlock(c*4, c*2)
        self.u1 = nn.ConvTranspose2d(c*2, c, 2, 2)
        self.c1 = ConvBlock(c*2, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        b = self.b(self.p3(d3))
        x = self.u3(b)
        x = torch.cat([x, d3], 1)
        x = self.c3(x)
        x = self.u2(x)
        x = torch.cat([x, d2], 1)
        x = self.c2(x)
        x = self.u1(x)
        x = torch.cat([x, d1], 1)
        x = self.c1(x)
        return self.out(x)


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2*(pred*target).sum() + eps
    den = pred.sum()+target.sum()+eps
    return 1 - num/den


def dice_coeff_from_logits(pred, target, thr=0.5, eps=1e-6):
    p = (torch.sigmoid(pred) >= thr).float()
    t = (target >= 0.5).float()
    inter = (p*t).sum() * 2
    denom = p.sum() + t.sum() + eps
    return (inter + eps) / denom


def boundary_dice_loss(pred, target, eps=1e-6):
    # emphasize borders via morphological gradient band
    import torch.nn.functional as F
    with torch.no_grad():
        y = (target >= 0.5).float()
        # dilation and erosion via pooling
        dil = F.max_pool2d(y, kernel_size=3, stride=1, padding=1)
        ero = 1 - F.max_pool2d(1 - y, kernel_size=3, stride=1, padding=1)
        band = (dil - ero).clamp(min=0.0, max=1.0)  # [B,1,H,W]
    pred_sig = torch.sigmoid(pred)
    num = 2*(pred_sig*target*band).sum() + eps
    den = (pred_sig*band).sum() + (target*band).sum() + eps
    return 1 - num/den


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(train_root="data/train", val_root="data/val", epochs=40, bs=4, lr=1e-3,
        channels=32, weight_decay=1e-4, bce_weight=0.3, dice_weight=0.7,
        edge_weight=0.2, in_channels=3,
        crop_size=256, amp=True, use_onecycle=True, patience=10, seed=42,
        num_workers=None):
    set_seeds(seed)
    device = (
        "cuda" if torch.cuda.is_available() else
        ("mps" if hasattr(torch.backends, "mps")
         and torch.backends.mps.is_available() else "cpu")
    )
    net = UNet(c=channels, in_channels=in_channels).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    if num_workers is None:
        try:
            num_workers = max(1, os.cpu_count() or 1)
        except Exception:
            num_workers = 1
    pin = device == "cuda"
    train_loader = DataLoader(EMSet(train_root, aug=True, crop_size=crop_size), batch_size=bs,
                              shuffle=True, num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(EMSet(val_root, aug=False, crop_size=crop_size),
                            batch_size=1, num_workers=num_workers, pin_memory=pin)

    if use_onecycle:
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=amp and device == "cuda")

    best_dice = -1.0
    epochs_no_improve = 0
    for ep in range(1, epochs+1):
        net.train()
        loss_sum = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}"):
            x, y = x.to(device, non_blocking=True), y.to(
                device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                p = net(x)
                loss = bce_weight * \
                    bce(p, y) + dice_weight*dice_loss(p, y) + \
                    edge_weight*boundary_dice_loss(p, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if use_onecycle:
                scheduler.step()
            loss_sum += loss.detach().item()

        net.eval()
        val_loss = 0.0
        val_dice = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(
                    device, non_blocking=True)
                p = net(x)
                val_loss += (bce_weight*bce(p, y)+dice_weight*dice_loss(p,
                             y)+edge_weight*boundary_dice_loss(p, y)).item()
                val_dice += dice_coeff_from_logits(p, y).item()
                n_val += 1
        val_loss /= max(1, n_val)
        val_dice /= max(1, n_val)
        train_loss = loss_sum/max(1, len(train_loader))
        print(
            f"train {train_loss:.4f}  val {val_loss:.4f}  valDice {val_dice:.4f}  lr {opt.param_groups[0]['lr']:.2e}")
        if not use_onecycle:
            scheduler.step()
        improved = val_dice > best_dice + 1e-6
        if improved:
            best_dice = val_dice
            epochs_no_improve = 0
            torch.save(net.state_dict(), "unet_best.pt")
            print("saved unet_best.pt (best by Dice)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {ep} (no improvement for {patience} epochs)")
                break


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", type=str, default="data/train")
    p.add_argument("--val_root", type=str, default="data/val")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--bce_weight", type=float, default=0.3)
    p.add_argument("--dice_weight", type=float, default=0.7)
    p.add_argument("--edge_weight", type=float, default=0.2)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no_onecycle", action="store_true")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=None)
    args = p.parse_args()
    run(train_root=args.train_root, val_root=args.val_root, epochs=args.epochs, bs=args.bs, lr=args.lr,
        channels=args.channels, weight_decay=args.weight_decay, bce_weight=args.bce_weight, dice_weight=args.dice_weight,
        edge_weight=args.edge_weight, in_channels=args.in_channels,
        crop_size=args.crop_size, amp=args.amp, use_onecycle=(not args.no_onecycle), patience=args.patience,
        seed=args.seed, num_workers=args.num_workers)
