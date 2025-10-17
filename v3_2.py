import gradio as gr
import torch
import cv2
import numpy as np
import os
from PIL import Image
from datetime import datetime
from typing import Optional, Tuple
import tempfile
import shutil

# =============================
# 1) Global & Model Loading
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

midas = None
transform = None
model_type = "DPT_Large"

try:
    # Load MiDaS model
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    if DEVICE == "cuda":
        midas = midas.half()  # FP16 on GPU for speed/memory
    midas.to(DEVICE)
    midas.eval()

    # Load proper transforms (robust to model type)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if "DPT" in model_type:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    print("MiDaS model loaded successfully.")
except Exception as e:
    print(f"Error loading MiDaS model: {e}")
    midas = None
    transform = None

# =============================
# 2) Path helpers (WSL-aware)
# =============================

def _is_wsl() -> bool:
    try:
        with open('/proc/version', 'r') as f:
            s = f.read().lower()
        return ('microsoft' in s) or ('wsl' in s)
    except Exception:
        return False

def get_default_download_path() -> str:
    """Auto-select a sensible default download path (WSL-friendly)."""
    if _is_wsl():
        base = "/mnt/c/Users"
        if os.path.isdir(base):
            # Prefer the first user that has a Downloads folder
            for u in os.listdir(base):
                d = os.path.join(base, u, "Downloads")
                if os.path.isdir(d):
                    return os.path.join(d, "normal_maps")
    return "outputs"

def ensure_path_exists(path: str) -> Tuple[str, bool]:
    """Ensure the path exists; if fail, fallback to ./outputs.
    Returns (actual_path, created_ok)
    """
    try:
        os.makedirs(path, exist_ok=True)
        return path, True
    except Exception as e:
        print(f"Path create failed: {e}")
        fallback = "outputs"
        os.makedirs(fallback, exist_ok=True)
        return fallback, False

# =============================
# 3) Image processing primitives
# =============================

def apply_blur_sharp(image: np.ndarray, value: int) -> np.ndarray:
    """Negative = blur, Positive = sharpen, 0 = pass-through."""
    if value == 0:
        return image
    if value < 0:  # blur
        kernel_size = int(abs(value) * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    # sharpen
    alpha = value * 0.1
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.0 + alpha, blurred, -alpha, 0)
    return sharpened


def preprocess_for_normal(image: np.ndarray, method: str = "median_blur") -> np.ndarray:
    img8 = image.astype(np.uint8)
    if method == "median_blur":
        return cv2.medianBlur(img8, 5)
    elif method == "bilateral":
        return cv2.bilateralFilter(img8, 9, 75, 75)
    elif method == "gaussian_smooth":
        return cv2.GaussianBlur(img8, (5, 5), 1.0)
    return img8


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma correction (display gamma). Using pow(x, 1/gamma) for intuitive control."""
    if gamma == 1.0 or image is None:
        return image
    img = image.astype(np.float32) / 255.0
    corrected = np.power(np.clip(img, 0, 1), 1.0 / float(gamma))
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def rgb_to_height_map(image: np.ndarray, method: str = "luminance") -> np.ndarray:
    """Convert RGB(A) to single-channel height map (float64). Assumes RGB order."""
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    img = image.astype(np.float32)
    if method == "luminance":
        # sRGB coefficients
        h = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    elif method == "average":
        h = img.mean(axis=2)
    elif method == "max":
        h = img.max(axis=2)
    elif method in ("red", "green", "blue"):
        ch = {"red": 0, "green": 1, "blue": 2}[method]
        h = img[..., ch]
    else:
        h = img.mean(axis=2)
    return h.astype(np.float64)


def _compute_gradients(h: np.ndarray, algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
    h32 = h.astype(np.float32)
    if algorithm == "sobel":
        gx = cv2.Sobel(h32, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(h32, cv2.CV_32F, 0, 1, ksize=3)
    elif algorithm == "sobel_5":
        gx = cv2.Sobel(h32, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(h32, cv2.CV_32F, 0, 1, ksize=5)
    elif algorithm == "scharr":
        gx = cv2.Scharr(h32, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(h32, cv2.CV_32F, 0, 1)
    elif algorithm == "smooth_sobel":
        s = cv2.GaussianBlur(h32, (3, 3), 0.8)
        gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
    elif algorithm == "prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)
        gx = cv2.filter2D(h32, cv2.CV_32F, kx)
        gy = cv2.filter2D(h32, cv2.CV_32F, ky)
    else:
        gx = cv2.Sobel(h32, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(h32, cv2.CV_32F, 0, 1, ksize=3)
    return gx.astype(np.float64), gy.astype(np.float64)


def finalize_normal_map(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray, green_up: bool) -> Image.Image:
    """Normalize and pack to 8-bit normal map. green_up=True => OpenGL (G up); False => DirectX/Unity (G down)."""
    if not green_up:
        ny = -ny
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    out = np.stack([(nx * 0.5 + 0.5) * 255.0, (ny * 0.5 + 0.5) * 255.0, (nz * 0.5 + 0.5) * 255.0], axis=-1)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def generate_normal_from_image(input_image: np.ndarray, intensity: float = 1.0, algorithm: str = "sobel",
                               height_method: str = "luminance", noise_reduction: bool = True,
                               green_up: bool = False) -> Image.Image:
    """Generate a normal map directly from the RGB image."""
    h = rgb_to_height_map(input_image, height_method)
    h = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if noise_reduction:
        h = preprocess_for_normal(h, "bilateral")
    h = h.astype(np.float64)
    gx, gy = _compute_gradients(h, algorithm)
    scale = float(intensity)
    nx = -gx * scale
    ny = gy * scale
    nz = np.ones_like(nx)
    return finalize_normal_map(nx, ny, nz, green_up=green_up)


def depth_to_normal(depth_map: np.ndarray, intensity: float = 1.0, algorithm: str = "sobel",
                    green_up: bool = False) -> Image.Image:
    """Convert a depth map to a normal map with consistent intensity semantics."""
    d = depth_map.astype(np.float64)
    gx, gy = _compute_gradients(d, algorithm)
    nx = -gx * float(intensity)
    ny = gy * float(intensity)
    nz = np.ones_like(nx)
    return finalize_normal_map(nx, ny, nz, green_up=green_up)

# =============================
# 4) Save / Compress helpers
# =============================

def compress_image(image, compression_level: str = "medium", fmt: str = "PNG"):
    """Lossless by default for WEBP (suitable for normals/depth)."""
    if compression_level == "none":
        return image if isinstance(image, Image.Image) else Image.fromarray(image)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    from io import BytesIO
    buffer = BytesIO()
    f = (fmt or "PNG").upper()

    if f == "PNG":
        compress_level = 6
        if compression_level == "low":
            compress_level = 1
        elif compression_level == "high":
            compress_level = 9
        image.save(buffer, format="PNG", compress_level=compress_level, optimize=True)

    elif f == "WEBP":
        # Use lossless to preserve normals/depth integrity
        image.save(buffer, format="WEBP", lossless=True, quality=100, method=6)

    elif f == "JPEG":
        # Not recommended for normal/depth, but supported
        quality = 85
        if compression_level == "low":
            quality = 95
        elif compression_level == "high":
            quality = 75
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format="JPEG", quality=quality, optimize=True)

    buffer.seek(0)
    return Image.open(buffer)


def save_image_with_compression(image, path: str, prefix: str, compression_level: str = "medium",
                                fmt: str = "PNG", filename_base: Optional[str] = None) -> str:
    if image is None:
        return "沒有可下載的圖片"
    try:
        actual_path, _ = ensure_path_exists(path)
        compressed = compress_image(image, compression_level, fmt)
        ext = (fmt or 'PNG').lower()
        if filename_base:
            file_path = os.path.join(actual_path, f"{filename_base}.{ext}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(actual_path, f"{prefix}_{timestamp}.{ext}")

        if ext == 'jpeg' and compressed.mode == 'RGBA':
            compressed = compressed.convert('RGB')
        compressed.save(file_path)

        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if actual_path.startswith('/mnt/c/'):
            windows_path = actual_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
            return (f"✅ 成功儲存至 Windows 路徑:\n{windows_path}\\{os.path.basename(file_path)}\n"
                    f"📦 檔案大小: {size_mb:.2f} MB\n🗜️ 壓縮等級: {compression_level.upper()}")
        else:
            return (f"✅ 成功儲存至: {file_path}\n"
                    f"📦 檔案大小: {size_mb:.2f} MB\n🗜️ 壓縮等級: {compression_level.upper()}")
    except Exception as e:
        return f"❌ 儲存失敗: {e}"

# =============================
# 5) Core pipeline (single image)
# =============================

def process_image(input_image: np.ndarray, strength: float, level: float, blur_sharp: int,
                  algorithm: str, normal_source: str, height_method: str,
                  normal_level: float, normal_blur_sharp: int,
                  green_up: bool) -> Tuple[Image.Image, Image.Image]:
    """Process image and return (normal_map, depth_preview)."""
    if input_image is None:
        raise gr.Error("請先上傳一張圖片！")

    print(f"Processing with: Strength={strength}, Algorithm={algorithm}, Source={normal_source}")
    print(f"Depth Params: Level={level}, Blur/Sharp={blur_sharp}")
    print(f"Normal Params: Level={normal_level}, Blur/Sharp={normal_blur_sharp}, GreenUp={green_up}")

    depth_pil = None
    processed_depth = None

    # Always attempt MiDaS preview if available
    if midas is not None and transform is not None:
        try:
            # input_image is RGB numpy (H,W,3)
            img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
            inp = transform(img_bgr).to(DEVICE)
            if DEVICE == 'cuda':
                inp = inp.half()
            with torch.no_grad():
                pred = midas(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=img_bgr.shape[:2], mode="bicubic", align_corners=False
                ).squeeze(1)
            depth_map = pred.squeeze().detach().float().cpu().numpy()
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            processed_depth = apply_blur_sharp(depth_vis, blur_sharp)
            processed_depth = apply_gamma(processed_depth, level)
            depth_pil = Image.fromarray(cv2.cvtColor(processed_depth, cv2.COLOR_GRAY2RGB))
        except Exception as e:
            print(f"MiDaS processing failed: {e}")
            depth_pil = Image.fromarray(np.zeros((input_image.shape[0], input_image.shape[1], 3), dtype=np.uint8))

    # Normal map
    if normal_source == "direct":
        proc = input_image.copy()
        proc = apply_gamma(proc, normal_level)
        if normal_blur_sharp != 0:
            ch = cv2.split(proc)
            ch = [apply_blur_sharp(c, normal_blur_sharp) for c in ch]
            proc = cv2.merge(ch)
        normal_map_image = generate_normal_from_image(
            proc, intensity=strength, algorithm=algorithm,
            height_method=height_method, green_up=green_up
        )
    elif normal_source == "midas" and processed_depth is not None:
        normal_map_image = depth_to_normal(processed_depth, intensity=strength,
                                           algorithm=algorithm, green_up=green_up)
    else:
        # Fallback to direct
        normal_map_image = generate_normal_from_image(
            input_image, intensity=strength, algorithm=algorithm,
            height_method=height_method, green_up=green_up
        )

    print("Processing complete.")
    return normal_map_image, depth_pil

# =============================
# 6) Batch processing
# =============================

def process_batch(files, strength, level, blur_sharp, algorithm, normal_source,
                  height_method, normal_level, normal_blur_sharp,
                  compression_level, file_format, output_path,
                  green_up):
    if not files:
        return "沒有選擇檔案"

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    actual_path, _ = ensure_path_exists(output_path)
    batch_folder = os.path.join(actual_path, f"batch_{timestamp}")
    os.makedirs(batch_folder, exist_ok=True)

    for i, file in enumerate(files):
        try:
            with Image.open(file.name) as im:
                img_array = np.array(im.convert("RGB"))
            normal_map, depth_map = process_image(
                img_array, strength, level, blur_sharp, algorithm, normal_source,
                height_method, normal_level, normal_blur_sharp, green_up
            )

            original_filename = os.path.basename(file.name)
            base_filename = os.path.splitext(original_filename)[0]

            # Save normal
            ext = (file_format or 'PNG').lower()
            normal_path = os.path.join(batch_folder, f"{base_filename}_normal.{ext}")
            normal_compressed = compress_image(normal_map, compression_level, file_format or 'PNG')
            if ext == 'jpeg' and normal_compressed.mode == 'RGBA':
                normal_compressed = normal_compressed.convert('RGB')
            normal_compressed.save(normal_path)
            results.append(f"✓ {base_filename}_normal.{ext}")

            # Save depth
            if depth_map is not None:
                ext = (file_format or 'PNG').lower()
                depth_for_compress = depth_map if isinstance(depth_map, Image.Image) else Image.fromarray(depth_map)
                depth_compressed = compress_image(depth_for_compress, compression_level, file_format or 'PNG')
                if ext == 'jpeg' and depth_compressed.mode == 'RGBA':
                    depth_compressed = depth_compressed.convert('RGB')
                depth_path = os.path.join(batch_folder, f"{base_filename}_depth.{ext}")
                depth_compressed.save(depth_path)
                results.append(f"✓ {base_filename}_depth.{ext}")

        except Exception as e:
            results.append(f"✗ 檔案 {i+1} 處理失敗: {str(e)}")

    if batch_folder.startswith('/mnt/c/'):
        windows_path = batch_folder.replace('/mnt/c/', 'C:\\').replace('/', '\\')
        feedback = f"✅ 批次處理完成！\n📁 Windows 路徑: {windows_path}\n\n"
    else:
        feedback = f"✅ 批次處理完成！\n📁 儲存位置: {batch_folder}\n\n"
    return feedback + "\n".join(results)

# =============================
# 7) Presets
# =============================
PRESETS = {
    "標準 (低雜訊)": {"strength": 0.01, "level": 1.0, "blur_sharp": 0, "algorithm": "smooth_sobel", "normal_source": "direct", "height_method": "luminance", "normal_level": 1.0, "normal_blur_sharp": -2},
    "平滑": {"strength": 0.008, "level": 1.0, "blur_sharp": -5, "algorithm": "sobel_5", "normal_source": "direct", "height_method": "luminance", "normal_level": 0.8, "normal_blur_sharp": -5},
    "銳利 (低雜訊)": {"strength": 0.015, "level": 1.0, "blur_sharp": 0, "algorithm": "scharr", "normal_source": "direct", "height_method": "luminance", "normal_level": 1.2, "normal_blur_sharp": -1},
    "極致細節": {"strength": 0.02, "level": 1.0, "blur_sharp": 0, "algorithm": "prewitt", "normal_source": "direct", "height_method": "max", "normal_level": 1.5, "normal_blur_sharp": 0},
    "超級平滑": {"strength": 0.005, "level": 1.0, "blur_sharp": -8, "algorithm": "smooth_sobel", "normal_source": "direct", "height_method": "average", "normal_level": 0.7, "normal_blur_sharp": -8},
    "石材專用": {"strength": 0.012, "level": 1.0, "blur_sharp": -3, "algorithm": "sobel_5", "normal_source": "direct", "height_method": "luminance", "normal_level": 1.1, "normal_blur_sharp": -3}
}

# Default platform: DirectX/Unity (Green Down)
DEFAULT_GREEN_UP = False

# =============================
# 8) Gradio UI
# =============================
with gr.Blocks(title="Advanced Normal Map Generator v3.2", theme=gr.themes.Soft()) as app:

    normal_map_state = gr.State()
    depth_map_state = gr.State()
    input_filename_state = gr.State(value="image")

    gr.Markdown("""
    # 🎨 進階法線貼圖生成器 v3.2
    ### 綠通道方向切換｜Gamma 語意修正｜MiDaS 容錯與 FP16｜更穩定的 I/O
    - **平台方向**：OpenGL (Green Up) 與 DirectX/Unity (Green Down) 可切換。
    - **壓縮建議**：Normal/Depth 建議 PNG；WebP 採 **lossless**，不建議 JPEG。
    """)

    with gr.Tabs():
        with gr.TabItem("單張處理"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_file = gr.File(label="上傳圖片", file_types=["image"], type="filepath")
                    input_image = gr.Image(type="numpy", label="圖片預覽", interactive=False)
                    uploaded_filename_display = gr.Textbox(
                        label="輸出檔案名稱（不含副檔名）",
                        value="image",
                        placeholder="自動填入上傳檔案名稱",
                        info="輸出會自動加上 _normal 或 _depth 後綴"
                    )

                    with gr.Accordion("快速預設", open=True):
                        preset_dropdown = gr.Dropdown(choices=list(PRESETS.keys()), label="選擇預設參數", value="標準 (低雜訊)")
                        apply_preset_btn = gr.Button("套用預設", size="sm")

                    with gr.Accordion("法線生成方式", open=True):
                        normal_source_radio = gr.Radio(
                            choices=[("直接從圖片", "direct"), ("從 MiDaS 深度圖", "midas")],
                            value="direct",
                            label="法線貼圖來源"
                        )
                        height_method_dropdown = gr.Dropdown(
                            choices=["luminance", "average", "max", "red", "green", "blue"],
                            value="luminance",
                            label="高度圖生成方式（僅直接方式）"
                        )
                        green_up_checkbox = gr.Checkbox(value=DEFAULT_GREEN_UP, label="OpenGL 綠向上 (Green Up) ；關閉=DirectX/Unity")

                    with gr.Accordion("詳細參數", open=True):
                        strength_slider = gr.Slider(0.0, 0.2, 0.01, step=0.001, label="Normal Strength (強度)")
                        algorithm_dropdown = gr.Dropdown(
                            choices=["smooth_sobel", "sobel", "sobel_5", "scharr", "prewitt"],
                            value="smooth_sobel",
                            label="邊緣檢測算法"
                        )
                        gr.Markdown("---")
                        gr.Markdown("`僅用於「直接從圖片」`")
                        normal_level_slider = gr.Slider(0.1, 3.0, 1.0, step=0.05, label="Normal Pre-Gamma")
                        normal_blur_sharp_slider = gr.Slider(-10, 10, -2, step=1, label="Normal Pre-Blur/Sharp")
                        gr.Markdown("---")
                        gr.Markdown("`僅用於「深度圖預覽」`")
                        level_slider = gr.Slider(0.1, 3.0, 1.0, step=0.05, label="Depth Map Gamma")
                        blur_sharp_slider = gr.Slider(-10, 10, 0, step=1, label="Depth Map Blur/Sharp")

                with gr.Column(scale=2):
                    with gr.Row():
                        output_depth_map = gr.Image(type="pil", label="深度圖預覽 (MiDaS)")
                        output_normal_map = gr.Image(type="pil", label="法線貼圖")

                with gr.Accordion("下載生成選項", open=True):
                    process_button = gr.Button("生成貼圖", variant="primary")
                    with gr.Row():
                        compression_level = gr.Dropdown(
                            choices=["none", "low", "medium", "high"],
                            value="medium",
                            label="壓縮等級",
                            info="normal/depth 不建議 lossy；WebP 會使用 lossless"
                        )
                        file_format = gr.Dropdown(
                            choices=[
                                "PNG",
                                "WEBP",
                                "JPEG"
                            ],
                            value="PNG",
                            label="檔案格式",
                            info="建議使用 PNG 保留品質"
                        )

                    default_path = get_default_download_path()
                    quick_paths_choices = [("本地 outputs 資料夾", "outputs")]
                    if "/mnt/c/Users/" in default_path:
                        quick_paths_choices.append(("Windows 下載資料夾", default_path))

                    download_path_input = gr.Textbox(
                        label="下載資料夾路徑",
                        value=default_path,
                        info="WSL 提示：使用 /mnt/c/Users/您的用戶名/Downloads 可存取 Windows 下載資料夾"
                    )

                    with gr.Row():
                        quick_paths = gr.Dropdown(choices=quick_paths_choices, label="快速路徑", value=None)
                        set_path_btn = gr.Button("設定路徑", size="sm")
                        browse_folder_btn = gr.Button("🗂️ 瀏覽資料夾", size="sm")

                    with gr.Row():
                        download_depth_btn = gr.Button("下載深度圖", variant="secondary")
                        download_normal_btn = gr.Button("下載法線貼圖", variant="secondary")

                    download_feedback = gr.Textbox(label="狀態", interactive=False, lines=6)

        # ---- 批次處理 ----
        with gr.TabItem("批次處理"):
            with gr.Row():
                with gr.Column():
                    batch_files = gr.File(file_count="multiple", file_types=["image"], label="選擇多張圖片")
                    with gr.Accordion("批次處理設定", open=True):
                        with gr.Row():
                            batch_normal_source = gr.Radio(
                                choices=[("直接從圖片", "direct"), ("從 MiDaS 深度圖", "midas")],
                                value="direct",
                                label="法線貼圖來源"
                            )
                            batch_height_method = gr.Dropdown(
                                choices=["luminance", "average", "max", "red", "green", "blue"],
                                value="luminance",
                                label="高度圖方式"
                            )
                        batch_green_up_checkbox = gr.Checkbox(value=DEFAULT_GREEN_UP, label="OpenGL 綠向上 (Green Up)")

                        with gr.Row():
                            batch_strength = gr.Slider(0.0, 0.2, 0.01, step=0.001, label="Normal Strength")
                            batch_algorithm = gr.Dropdown(
                                choices=["smooth_sobel", "sobel", "sobel_5", "scharr", "prewitt"],
                                value="smooth_sobel",
                                label="邊緣檢測算法"
                            )
                        gr.Markdown("`僅用於「直接從圖片」`")
                        with gr.Row():
                            batch_normal_level = gr.Slider(0.1, 3.0, 1.0, step=0.05, label="Normal Pre-Gamma")
                            batch_normal_blur_sharp = gr.Slider(-10, 10, -2, step=1, label="Normal Pre-Blur/Sharp")
                        gr.Markdown("`僅用於「深度圖預覽」`")
                        with gr.Row():
                            batch_level = gr.Slider(0.1, 3.0, 1.0, step=0.05, label="Depth Map Gamma")
                            batch_blur_sharp = gr.Slider(-10, 10, 0, step=1, label="Depth Map Blur/Sharp")

                    with gr.Accordion("輸出格式設定", open=True):
                        with gr.Row():
                            batch_compression_level = gr.Dropdown(
                                choices=["none", "low", "medium", "high"],
                                value="medium",
                                label="壓縮等級"
                            )
                            batch_file_format = gr.Dropdown(
                                choices=[
                                    "PNG",
                                    "WEBP",
                                    "JPEG"
                                ],
                                value="PNG",
                                label="檔案格式",
                                info="建議使用 PNG 保留品質"
                            )
                        batch_output_path = gr.Textbox(
                            label="批次輸出資料夾",
                            value=get_default_download_path(),
                            info="建議使用 Windows 路徑以便在檔案總管中開啟"
                        )
                        batch_quick_paths = gr.Dropdown(choices=[("本地 outputs 資料夾", "outputs")], label="快速路徑", value=None)
                        batch_browse_folder_btn = gr.Button("🗂️ 瀏覽資料夾", size="sm")
                        batch_set_path_btn = gr.Button("設定路徑", size="sm")

                    batch_process_btn = gr.Button("開始批次處理", variant="primary", size="lg")
                with gr.Column():
                    batch_result = gr.Textbox(label="批次處理結果", lines=20, interactive=False)

    # =============================
    # 9) Event handlers
    # =============================

    def apply_preset(preset_name):
        p = PRESETS[preset_name]
        return (
            p["strength"], p["level"], p["blur_sharp"], p["algorithm"],
            p["normal_source"], p["height_method"], p["normal_level"], p["normal_blur_sharp"]
        )

    def extract_filename_from_upload(file_path):
        try:
            if file_path is None:
                return "image", None
            filename_with_ext = os.path.basename(file_path)
            filename_base = os.path.splitext(filename_with_ext)[0]
            with Image.open(file_path) as pil:
                rgb = np.array(pil.convert("RGB"))
            return filename_base, rgb
        except Exception as e:
            print(f"Error extracting filename: {e}")
            return "image", None

    def on_process_click(image, strength, level, blur_sharp, algorithm, normal_source,
                         height_method, normal_level, normal_blur_sharp, filename, green_up):
        if image is None:
            raise gr.Error("尚未載入圖片")
        normal_map, depth_map = process_image(
            image, strength, level, blur_sharp, algorithm, normal_source,
            height_method, normal_level, normal_blur_sharp, green_up
        )
        filename_base = filename.strip() if filename and filename.strip() else "image"
        return normal_map, depth_map, normal_map, depth_map, filename_base

    apply_preset_btn.click(
        apply_preset,
        inputs=[preset_dropdown],
        outputs=[
            strength_slider, level_slider, blur_sharp_slider, algorithm_dropdown,
            normal_source_radio, height_method_dropdown,
            normal_level_slider, normal_blur_sharp_slider
        ]
    )

    input_file.change(
        extract_filename_from_upload,
        inputs=[input_file],
        outputs=[uploaded_filename_display, input_image]
    )

    process_button.click(
        on_process_click,
        inputs=[
            input_image, strength_slider, level_slider, blur_sharp_slider,
            algorithm_dropdown, normal_source_radio, height_method_dropdown,
            normal_level_slider, normal_blur_sharp_slider, uploaded_filename_display,
            green_up_checkbox
        ],
        outputs=[output_normal_map, output_depth_map, normal_map_state, depth_map_state, input_filename_state]
    )

    batch_process_btn.click(
        process_batch,
        inputs=[
            batch_files, batch_strength, batch_level, batch_blur_sharp, batch_algorithm,
            batch_normal_source, batch_height_method, batch_normal_level, batch_normal_blur_sharp,
            batch_compression_level, batch_file_format, batch_output_path,
            batch_green_up_checkbox
        ],
        outputs=[batch_result]
    )

    def set_quick_path(selected_path):
        return selected_path if selected_path else get_default_download_path()

    def open_folder_dialog():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            folder_path = filedialog.askdirectory(
                title="選擇下載資料夾",
                initialdir=get_default_download_path() if os.path.isdir(get_default_download_path()) else os.path.expanduser("~")
            )
            root.destroy()
            return folder_path if folder_path else get_default_download_path()
        except Exception as e:
            return f"❌ 選擇資料夾失敗: {e}。請手動輸入路徑。"

    set_path_btn.click(set_quick_path, inputs=[quick_paths], outputs=[download_path_input])
    browse_folder_btn.click(open_folder_dialog, inputs=[], outputs=[download_path_input])

    download_depth_btn.click(
        lambda img, path, comp_level, fmt, filename_base: save_image_with_compression(
            img, path, "depth_map", comp_level, fmt,
            f"{filename_base}_depth" if filename_base else None
        ) if img is not None else "❌ 尚未生成深度圖",
        inputs=[depth_map_state, download_path_input, compression_level, file_format, input_filename_state],
        outputs=[download_feedback]
    )

    download_normal_btn.click(
        lambda img, path, comp_level, fmt, filename_base: save_image_with_compression(
            img, path, "normal_map", comp_level, fmt,
            f"{filename_base}_normal" if filename_base else None
        ) if img is not None else "❌ 尚未生成法線貼圖",
        inputs=[normal_map_state, download_path_input, compression_level, file_format, input_filename_state],
        outputs=[download_feedback]
    )

    batch_set_path_btn.click(set_quick_path, inputs=[batch_quick_paths], outputs=[batch_output_path])
    batch_browse_folder_btn.click(open_folder_dialog, inputs=[], outputs=[batch_output_path])

if __name__ == "__main__":
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)
