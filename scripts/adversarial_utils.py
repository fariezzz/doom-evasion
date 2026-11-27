"""Shared utilities for adversarial attacks, defenses, Grad-CAM XAI, and safe checkpoint loading."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEFENSE_CHOICES = (
    "none",
    "bit_depth_4",
    "bit_depth_3",
    "median_3x3",
    "gaussian_3x3",
    "bit4_median",
)

DEFENSE_LABELS = {
    "none": "Tanpa Preprocessing",
    "bit_depth_4": "Bit-Depth 4-bit",
    "bit_depth_3": "Bit-Depth 3-bit",
    "median_3x3": "Median Filter 3x3",
    "gaussian_3x3": "Gaussian Blur 3x3",
    "bit4_median": "Bit-Depth 4-bit + Median 3x3",
}


def load_checkpoint_safe(path, device):
    """Load a PyTorch checkpoint safely with weights_only=True."""
    checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Model checkpoint does not contain a state_dict: " + str(path))
    return checkpoint


def apply_defense(x, defense="none"):
    """Apply an inference-time input preprocessing defense on a (N, C, H, W) tensor in [0, 1]."""
    if defense == "none" or defense is None:
        return x
    if defense == "bit_depth_4":
        return torch.clamp(torch.round(x * 15.0) / 15.0, 0.0, 1.0)
    if defense == "bit_depth_3":
        return torch.clamp(torch.round(x * 7.0) / 7.0, 0.0, 1.0)
    if defense == "median_3x3":
        padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
        patches = padded.unfold(2, 3, 1).unfold(3, 3, 1)
        patches = patches.contiguous().view(*x.shape, 9)
        return patches.median(dim=-1).values
    if defense == "gaussian_3x3":
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=x.dtype,
            device=x.device,
        ) / 16.0
        channels = x.shape[1]
        weight = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return torch.clamp(F.conv2d(padded, weight, groups=channels), 0.0, 1.0)
    if defense == "bit4_median":
        return apply_defense(apply_defense(x, "bit_depth_4"), "median_3x3")
    raise ValueError("Unknown defense: " + str(defense))


def fgsm_attack(model, x, y, epsilon, criterion=F.cross_entropy, eval_mode=True, random_start=False, generator=None):
    """Generate L_inf FGSM (or FGSM-RS when random_start=True) adversarial examples."""
    if epsilon <= 0.0:
        return x.clone().detach()
    was_training = model.training
    if eval_mode:
        model.eval()
    try:
        if random_start:
            if generator is not None:
                noise = (torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=generator) * 2.0 - 1.0) * epsilon
            else:
                noise = torch.empty_like(x).uniform_(-epsilon, epsilon)
            x_start = torch.clamp(x + noise, 0.0, 1.0).detach()
        else:
            x_start = x.detach()
        x_req = x_start.clone().requires_grad_(True)
        logits = model(x_req)
        loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, x_req, only_inputs=True)[0]
        step = epsilon * torch.sign(grad)
        if random_start:
            x_adv = x_start + step
            delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x + delta, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x + step, 0.0, 1.0)
        return x_adv.detach()
    finally:
        if eval_mode and was_training:
            model.train()


def pgd_attack(
    model,
    x,
    y,
    epsilon,
    steps=10,
    alpha=None,
    random_start=True,
    criterion=F.cross_entropy,
    eval_mode=True,
    generator=None,
):
    """Generate L_inf Projected Gradient Descent (PGD-k) adversarial examples."""
    if epsilon <= 0.0 or steps <= 0:
        return x.clone().detach()
    step_size = max(epsilon / 4.0, 1e-4) if alpha is None else float(alpha)
    was_training = model.training
    if eval_mode:
        model.eval()
    try:
        x_orig = x.detach()
        if random_start:
            if generator is not None:
                noise = (torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=generator) * 2.0 - 1.0) * epsilon
            else:
                noise = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
            x_adv = torch.clamp(x_orig + noise, 0.0, 1.0)
        else:
            x_adv = x_orig.clone()

        for _ in range(steps):
            x_req = x_adv.detach().requires_grad_(True)
            logits = model(x_req)
            loss = criterion(logits, y)
            grad = torch.autograd.grad(loss, x_req, only_inputs=True)[0]
            x_next = x_req.detach() + step_size * torch.sign(grad)
            delta = torch.clamp(x_next - x_orig, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
        return x_adv.detach()
    finally:
        if eval_mode and was_training:
            model.train()


def compute_gradcam(model, x, target_class=None):
    """Compute a normalized [0, 1] Grad-CAM heatmap for SmallCNN on a single input (1, 1, H, W).

    Uses the final Conv2d + ReLU activation map (features[:8], 21x21 spatial grid)
    before the last MaxPool2d for sharper localization.
    Returns (cam_np, logits_detached).
    """
    was_training = model.training
    model.eval()
    try:
        x_in = x.detach()
        with torch.enable_grad():
            acts = model.features[:8](x_in)
            acts = acts.detach().requires_grad_(True)
            pooled = model.features[8:](acts)
            logits = model.classifier(pooled)
            if target_class is None:
                cls_idx = int(logits.argmax(dim=1).item())
            else:
                cls_idx = int(target_class)
            score = logits[0, cls_idx]
            grads = torch.autograd.grad(score, acts, only_inputs=True)[0]
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(x_in.shape[2], x_in.shape[3]), mode="bilinear", align_corners=False)
            cam_np = cam[0, 0].detach().cpu().numpy()
            max_val = float(cam_np.max())
            if max_val > 1e-8:
                cam_np = cam_np / max_val
            else:
                cam_np = np.zeros_like(cam_np, dtype=np.float32)
            return cam_np.astype(np.float32), logits.detach()
    finally:
        if was_training:
            model.train()


def perturbation_vis(clean_np, adv_np, gain=20.0):
    """Amplify signed pixel perturbation around 0.5 gray for visual inspection."""
    return np.clip(0.5 + (adv_np - clean_np) * gain, 0.0, 1.0).astype(np.float32)


def render_gradcam_overlay(gray_frame, cam_01, alpha=0.45):
    """Overlay a [0, 1] Grad-CAM map onto a grayscale frame, returning BGR uint8."""
    if gray_frame.dtype != np.uint8:
        gray_u8 = np.rint(np.clip(gray_frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        gray_u8 = gray_frame
    h, w = gray_u8.shape[:2]
    cam_resized = cv2.resize(cam_01, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_u8 = np.rint(np.clip(cam_resized, 0.0, 1.0) * 255.0).astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap(cam_u8, colormap)
    base_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap, alpha, 0)
