"""
ComfyUI 自定义节点包：二次元风格化后处理 + 批量审美筛选
========================================================

两个节点：
1. AnimeStylePost —— 赛璐璐/二次元风格化后处理（色调分离、饱和度、对比度、
   线稿增强、柔光光晕），用于统一 AI 成片的色彩与光影表现。
2. BatchCurate   —— 批量审美评分与筛选（清晰度/色彩/曝光/构图四项指标），
   从批量出图中自动挑选最优成片，替代人工逐张挑选。

仅依赖 numpy / Pillow / torch，均为 ComfyUI 自带依赖，无需额外安装。
"""

import numpy as np
import torch
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# 工具函数：ComfyUI IMAGE 张量 <-> numpy 图像
# ComfyUI 的 IMAGE 是 float32 张量，shape = (batch, H, W, C)，RGB，范围 [0,1]
# ---------------------------------------------------------------------------

def comfy_image_to_np(image_tensor):
    """(B,H,W,C) float [0,1] 张量 -> list of uint8 RGB ndarray [0,255]"""
    if image_tensor is None:
        return []
    arr = image_tensor.detach().cpu().numpy().astype(np.float32)
    if arr.max() > 1.5:          # 兼容 [0,255] 输入
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return [(arr[i] * 255.0).astype(np.uint8) for i in range(arr.shape[0])]


def np_to_comfy_image(np_images):
    """list of uint8 RGB ndarray -> (B,H,W,C) float 张量"""
    arrs = [img.astype(np.float32) / 255.0 for img in np_images]
    return torch.from_numpy(np.stack(arrs, axis=0))


# ---------------------------------------------------------------------------
# 色彩空间工具（纯 numpy 实现，避免额外依赖）
# ---------------------------------------------------------------------------

def rgb_to_hsv(rgb):
    """rgb: (...,3) float [0,1] -> hsv (...,3)，h/s/v 均 [0,1]"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn

    h = np.zeros_like(mx)
    mask = d > 1e-6
    rc = (mx - r)[mask] / d[mask]
    gc = (mx - g)[mask] / d[mask]
    bc = (mx - b)[mask] / d[mask]
    hh = np.where(mx[mask] == r[mask], bc - gc,
                  np.where(mx[mask] == g[mask], 2.0 + rc - bc, 4.0 + gc - rc))
    h[mask] = (hh / 6.0) % 1.0

    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    v = mx
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv):
    """hsv: (...,3) float [0,1] -> rgb (...,3) float [0,1]"""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h6 = (h * 6.0) % 6.0
    hi = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    out = np.empty_like(hsv)
    for i in range(6):
        mask = hi == i
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        out[..., 0][mask] = r[mask]
        out[..., 1][mask] = g[mask]
        out[..., 2][mask] = b[mask]
    return out


# ---------------------------------------------------------------------------
# 图像处理基础算子
# ---------------------------------------------------------------------------

def edge_map(gray):
    """gray: (H,W) float [0,1] -> 梯度幅值图（Sobel 近似）"""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2.0
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2.0
    return np.sqrt(gx ** 2 + gy ** 2)


def add_glow(img, strength):
    """柔光/光晕：高斯模糊后 screen 混合"""
    blur = np.asarray(Image.fromarray(img).filter(
        ImageFilter.GaussianBlur(radius=8))).astype(np.float32) / 255.0
    f = img.astype(np.float32) / 255.0
    out = 1.0 - (1.0 - f) * (1.0 - blur * strength)
    return np.clip(out, 0.0, 1.0) * 255.0


# ---------------------------------------------------------------------------
# 节点 1：赛璐璐/二次元风格化后处理
# ---------------------------------------------------------------------------

class AnimeStylePost:
    """对 AI 成片做赛璐璐风格化调色与光影统一"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "posterize_levels": ("INT", {"default": 8, "min": 2, "max": 32, "step": 1}),
                "saturation": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 3.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 3.0, "step": 0.05}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "lineart_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "glow": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "image/style"

    def process(self, image, posterize_levels=8, saturation=1.25, contrast=1.1,
                brightness=0.0, lineart_strength=0.4, glow=0.0):
        out = []
        for im in comfy_image_to_np(image):
            f = im.astype(np.float32) / 255.0

            # 1) 亮度与对比度
            f = (f - 0.5) * contrast + 0.5 + brightness
            f = np.clip(f, 0.0, 1.0)

            # 2) 饱和度（HSV 空间）
            hsv = rgb_to_hsv(f)
            hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 1.0)
            f = hsv_to_rgb(hsv)

            # 3) 色调分离（平涂/赛璐璐感）
            f = np.clip(np.round(f * (posterize_levels - 1)) / (posterize_levels - 1), 0.0, 1.0)
            im2 = (f * 255.0).astype(np.uint8)

            # 4) 线稿增强
            if lineart_strength > 0:
                e = edge_map(f.mean(axis=-1))
                e = e / (e.max() + 1e-6)
                dark = np.clip(e * lineart_strength, 0.0, 0.9)[..., None]
                im2 = np.clip(im2.astype(np.float32) * (1.0 - dark), 0, 255).astype(np.uint8)

            # 5) 柔光光晕
            if glow > 0:
                im2 = add_glow(im2, glow).astype(np.uint8)

            out.append(im2)

        return (np_to_comfy_image(out),)


# ---------------------------------------------------------------------------
# 节点 2：批量审美评分与筛选
# ---------------------------------------------------------------------------

def score_sharpness(gray):
    """清晰度：拉普拉斯方差（越清晰越高）"""
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] +
           gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1])
    return float(lap.var())


def score_color(rgb):
    """色彩：平均饱和度 + 色相多样性"""
    hsv = rgb_to_hsv(rgb.astype(np.float32) / 255.0)
    sat = float(hsv[..., 1].mean())
    hue = (hsv[..., 0] * 180.0).astype(np.int32)
    diversity = len(np.unique(hue)) / 180.0
    return sat * 0.7 + diversity * 0.3


def score_exposure(rgb):
    """曝光：惩罚过暗/过曝/削波，奖励均衡亮度"""
    gray = rgb.astype(np.float32).mean(axis=-1)
    clipped = float(((gray < 15) | (gray > 240)).mean())
    mean_penalty = abs(float(gray.mean()) - 127.0) / 127.0
    return max(0.0, 1.0 - clipped - mean_penalty * 0.5)


def score_composition(rgb):
    """构图：三分法交点处的边缘能量占比（>1 说明主体落在三分点）"""
    gray = rgb.astype(np.float32).mean(axis=-1) / 255.0
    e = edge_map(gray)
    h, w = e.shape
    ys = [h // 3, 2 * h // 3]
    xs = [w // 3, 2 * w // 3]
    r = max(h, w) // 12
    pts = 0.0
    for y in ys:
        for x in xs:
            pts += e[max(0, y - r):y + r, max(0, x - r):x + r].mean()
    return float(min(1.5, pts / (e.mean() + 1e-6)))


class BatchCurate:
    """对一批出图做审美评分，按总分排序并返回最优 top_n 张"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "top_n": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1}),
                "w_sharpness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "w_color": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "w_exposure": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "w_composition": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("best_images", "report")
    FUNCTION = "curate"
    CATEGORY = "image/batch"

    def curate(self, image, top_n=4, w_sharpness=1.0, w_color=1.0,
               w_exposure=1.0, w_composition=1.0):
        imgs = comfy_image_to_np(image)
        n = len(imgs)

        raw = {"sharpness": [], "color": [], "exposure": [], "composition": []}
        for im in imgs:
            gray = im.astype(np.float32).mean(axis=-1) / 255.0
            raw["sharpness"].append(score_sharpness(gray))
            raw["color"].append(score_color(im))
            raw["exposure"].append(score_exposure(im))
            raw["composition"].append(score_composition(im))

        weights = {
            "sharpness": w_sharpness,
            "color": w_color,
            "exposure": w_exposure,
            "composition": w_composition,
        }
        total_w = sum(weights.values()) or 1.0

        # 各指标在批内 min-max 归一化到 [0,1] 后加权求和
        scores = np.zeros(n)
        for k in raw:
            arr = np.array(raw[k])
            rng = arr.max() - arr.min()
            norm = (arr - arr.min()) / rng if rng > 1e-6 else np.full_like(arr, 0.5)
            scores += weights[k] * norm
        scores /= total_w

        order = np.argsort(-scores)
        best = [imgs[i] for i in order[:min(top_n, n)]]

        lines = ["BatchCurate 审美评分（由高到低）："]
        for j in range(n):
            idx = order[j]
            lines.append(f"  rank {j + 1}: img#{idx}  score={scores[idx]:.3f}")
        report = "\n".join(lines)

        return (np_to_comfy_image(best), report)


# ---------------------------------------------------------------------------
# 注册节点
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "AnimeStylePost": AnimeStylePost,
    "BatchCurate": BatchCurate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimeStylePost": "Anime Style Post-Process (Cel Shade)",
    "BatchCurate": "Batch Curate (Aesthetic Score)",
}
