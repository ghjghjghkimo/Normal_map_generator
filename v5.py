import gradio as gr
import torch
import cv2
import numpy as np
import os
from PIL import Image
from datetime import datetime

# --- 1. 全域變數和模型載入 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

try:
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(DEVICE)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    print("MiDaS DPT_Large model loaded successfully.")
except Exception as e:
    print(f"Error loading MiDaS model: {e}")
    midas = None

# --- 2. 核心功能 ---

import os
import platform

# 在全域變數區域加入 WSL 路徑檢測
def get_default_download_path():
    """根據環境自動設定預設下載路徑"""
    # 檢查是否在 WSL 環境
    try:
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower() or 'wsl' in f.read().lower():
                # WSL 環境，設定 Windows 下載資料夾
                windows_user = os.environ.get('USER', 'user')
                return f"/mnt/c/Users/{windows_user}/Downloads/normal_maps"
    except:
        pass
    
    # 非 WSL 環境或無法檢測，使用本地路徑
    return "outputs"

def ensure_path_exists(path):
    """確保路徑存在，並處理 WSL 路徑問題"""
    try:
        # 如果是 Windows 路徑（通過 /mnt/c 存取）
        if path.startswith('/mnt/c/'):
            # 確保 Windows 路徑存在
            os.makedirs(path, exist_ok=True)
            return path, True
        else:
            # 一般 Linux 路徑
            os.makedirs(path, exist_ok=True)
            return path, True
    except Exception as e:
        print(f"路徑創建失敗: {e}")
        # 回退到本地 outputs 資料夾
        fallback_path = "outputs"
        os.makedirs(fallback_path, exist_ok=True)
        return fallback_path, False

def save_image(image, path, prefix):
    """改進的儲存函式，支援 WSL"""
    if image is None:
        return "沒有可下載的圖片"
    
    try:
        # 處理路徑
        actual_path, success = ensure_path_exists(path)
        
        # 生成檔案名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(actual_path, f"{prefix}_{timestamp}.png")
        
        # 儲存圖片
        image.save(file_path)
        
        # 根據環境提供不同的反饋
        if actual_path.startswith('/mnt/c/'):
            # WSL 環境，轉換為 Windows 路徑顯示
            windows_path = actual_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
            return f"✅ 成功儲存至 Windows 路徑:\n{windows_path}\\{prefix}_{timestamp}.png"
        else:
            return f"✅ 成功儲存至: {file_path}"
            
    except Exception as e:
        return f"❌ 儲存失敗: {e}"


def apply_blur_sharp(image, value):
    """對圖片應用模糊或銳化"""
    if value == 0:
        return image
    
    if value < 0:
        kernel_size = int(abs(value) * 2) + 1
        if kernel_size % 2 == 0: kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    else:
        alpha = value * 0.1
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 1.0 + alpha, blurred, -alpha, 0)
        return sharpened

def preprocess_for_normal(image, method="median_blur"):
    """預處理圖片以減少雜訊"""
    if method == "median_blur":
        return cv2.medianBlur(image.astype(np.uint8), 5)
    elif method == "bilateral":
        return cv2.bilateralFilter(image.astype(np.uint8), 9, 75, 75)
    elif method == "gaussian_smooth":
        return cv2.GaussianBlur(image.astype(np.uint8), (5, 5), 1.0)
    return image

def apply_gamma(image, gamma):
    """應用 Gamma 曲線校正"""
    if gamma == 1.0:
        return image
    norm_image = image / 255.0
    gamma_corrected = np.power(norm_image, gamma)
    return (gamma_corrected * 255).astype(np.uint8)

def rgb_to_height_map(image, method="luminance"):
    """將 RGB 圖片轉換為高度圖（深度圖）"""
    if len(image.shape) == 3:
        if method == "luminance":
            height_map = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        elif method == "average":
            height_map = np.mean(image, axis=2)
        elif method == "max":
            height_map = np.max(image, axis=2)
        elif method == "red":
            height_map = image[:,:,0]
        elif method == "green":
            height_map = image[:,:,1]
        elif method == "blue":
            height_map = image[:,:,2]
    else:
        height_map = image
    
    return height_map.astype(np.float64)

def generate_normal_from_image(input_image, intensity=1.0, algorithm="sobel", height_method="luminance", noise_reduction=True):
    """直接從 RGB 圖片生成法線貼圖，減少雜訊"""
    height_map = rgb_to_height_map(input_image, height_method)
    height_map = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if noise_reduction:
        height_map = preprocess_for_normal(height_map, "bilateral")
    
    height_map = height_map.astype(np.float64)
    
    if algorithm == "sobel":
        grad_x = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=3)
    elif algorithm == "sobel_5":
        grad_x = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=5)
    elif algorithm == "scharr":
        grad_x = cv2.Scharr(height_map, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(height_map, cv2.CV_64F, 0, 1)
    elif algorithm == "smooth_sobel":
        smoothed = cv2.GaussianBlur(height_map, (3, 3), 0.8)
        grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    elif algorithm == "prewitt":
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
        grad_x = cv2.filter2D(height_map, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(height_map, cv2.CV_64F, kernel_y)

    scale_factor = intensity

    normal_map = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.float64)
    # --- MODIFIED (1/2): 修正 Y 軸方向 ---
    normal_map[..., 0] = -grad_x * scale_factor
    normal_map[..., 1] = grad_y * scale_factor  # Y (Green) - 反轉符號以匹配 3D 座標系
    normal_map[..., 2] = 1.0

    norm = np.sqrt(np.square(normal_map).sum(axis=2, keepdims=True))
    norm[norm == 0] = 1e-9
    normal_map /= norm

    normal_map_uint8 = ((normal_map * 0.5 + 0.5) * 255).astype(np.uint8)
    return Image.fromarray(normal_map_uint8)


def depth_to_normal(depth_map, intensity=1.0, algorithm="sobel"):
    """將深度圖轉換為法線貼圖"""
    depth_map = depth_map.astype(np.float64)

    # 修正：增加所有算法支援，並提供預設值
    if algorithm == "sobel" or algorithm == "smooth_sobel":  # 添加 smooth_sobel 支援
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    elif algorithm == "sobel_5":  # 添加 sobel_5 支援
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
    elif algorithm == "scharr":
        grad_x = cv2.Scharr(depth_map, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(depth_map, cv2.CV_64F, 0, 1)
    elif algorithm == "prewitt":  # 添加 prewitt 支援
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
        grad_x = cv2.filter2D(depth_map, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(depth_map, cv2.CV_64F, kernel_y)
    elif algorithm == "laplacian":
        laplacian = cv2.Laplacian(depth_map, cv2.CV_64F)
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3) + laplacian * 0.1
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3) + laplacian * 0.1
    else:
        # 預設回退到 sobel
        print(f"Warning: Unknown algorithm '{algorithm}', falling back to sobel")
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)

    normal_map = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.float64)
    # --- MODIFIED (2/2): 修正 Y 軸方向 ---
    normal_map[..., 0] = -grad_x
    normal_map[..., 1] = grad_y # Y (Green) - 反轉符號
    normal_map[..., 2] = 1.0 / intensity

    norm = np.sqrt(np.square(normal_map).sum(axis=2, keepdims=True))
    norm[norm == 0] = 1e-9
    normal_map /= norm

    normal_map_uint8 = ((normal_map * 0.5 + 0.5) * 255).astype(np.uint8)
    return Image.fromarray(normal_map_uint8)


def compress_image(image, compression_level="medium", format="PNG"):
    """壓縮圖片"""
    if compression_level == "none":
        return image
    
    # 轉換為 PIL Image（如果不是的話）
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # 建立記憶體緩衝區
    from io import BytesIO
    buffer = BytesIO()
    
    if format.upper() == "PNG":
        # PNG 壓縮設定
        if compression_level == "low":
            compress_level = 1  # 最小壓縮
        elif compression_level == "medium":
            compress_level = 6  # 平衡
        elif compression_level == "high":
            compress_level = 9  # 最大壓縮
        
        image.save(buffer, format="PNG", compress_level=compress_level, optimize=True)
        
    elif format.upper() == "WEBP":
        # WebP 壓縮（更好的壓縮比）
        if compression_level == "low":
            quality = 95
        elif compression_level == "medium":
            quality = 85
        elif compression_level == "high":
            quality = 75
        
        image.save(buffer, format="WEBP", quality=quality, optimize=True)
        
    elif format.upper() == "JPEG":
        # JPEG 壓縮（法線貼圖不推薦，但檔案最小）
        if compression_level == "low":
            quality = 95
        elif compression_level == "medium":
            quality = 85
        elif compression_level == "high":
            quality = 75
        
        # 法線貼圖轉 JPEG 需要先轉 RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    return compressed_image

def save_image_with_compression(image, path, prefix, compression_level="medium", format="PNG"):
    """改進的儲存函式，支援壓縮"""
    if image is None:
        return "沒有可下載的圖片"
    
    try:
        # 處理路徑
        actual_path, success = ensure_path_exists(path)
        
        # 壓縮圖片
        compressed_image = compress_image(image, compression_level, format)
        
        # 生成檔案名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = format.lower()
        file_path = os.path.join(actual_path, f"{prefix}_{timestamp}.{file_extension}")
        
        # 儲存圖片
        original_size = len(image.tobytes()) if hasattr(image, 'tobytes') else 0
        
        if format.upper() == "PNG":
            compressed_image.save(file_path, format="PNG", compress_level=(1 if compression_level=="low" else 6 if compression_level=="medium" else 9), optimize=True)
        elif format.upper() == "WEBP":
            quality = 95 if compression_level=="low" else 85 if compression_level=="medium" else 75
            compressed_image.save(file_path, format="WEBP", quality=quality, optimize=True)
        elif format.upper() == "JPEG":
            quality = 95 if compression_level=="low" else 85 if compression_level=="medium" else 75
            if compressed_image.mode == 'RGBA':
                compressed_image = compressed_image.convert('RGB')
            compressed_image.save(file_path, format="JPEG", quality=quality, optimize=True)
        
        # 計算檔案大小
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # 根據環境提供不同的反饋
        if actual_path.startswith('/mnt/c/'):
            windows_path = actual_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
            return f"✅ 成功儲存至 Windows 路徑:\n{windows_path}\\{prefix}_{timestamp}.{file_extension}\n📦 檔案大小: {file_size_mb:.2f} MB\n🗜️ 壓縮等級: {compression_level.upper()}"
        else:
            return f"✅ 成功儲存至: {file_path}\n📦 檔案大小: {file_size_mb:.2f} MB\n🗜️ 壓縮等級: {compression_level.upper()}"
            
    except Exception as e:
        return f"❌ 儲存失敗: {e}"



# --- MODIFIED: 更新 process_image 函式以接受獨立參數 ---
def process_image(input_image, strength, level, blur_sharp, algorithm, normal_source, height_method, normal_level, normal_blur_sharp):
    """處理圖片並生成法線貼圖"""
    if input_image is None:
        raise gr.Error("請先上傳一張圖片！")

    print(f"Processing with: Strength={strength}, Algorithm={algorithm}, Source={normal_source}")
    print(f"Depth Map Params: Level={level}, Blur/Sharp={blur_sharp}")
    print(f"Normal Map Params: Level={normal_level}, Blur/Sharp={normal_blur_sharp}")

    depth_pil = None
    processed_depth = None

    # Step 1: 總是生成 MiDaS 深度圖用於預覽
    if midas is not None:
        try:
            img = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
            input_batch = transform(img).to(DEVICE)
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            depth_map_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # 使用深度圖專用參數進行處理
            processed_depth = apply_blur_sharp(depth_map_visual, blur_sharp)
            processed_depth = apply_gamma(processed_depth, level)
            depth_pil = Image.fromarray(cv2.cvtColor(processed_depth, cv2.COLOR_GRAY2RGB))
        except Exception as e:
            print(f"MiDaS processing failed: {e}")
            depth_pil = Image.fromarray(np.zeros((input_image.shape[0], input_image.shape[1], 3), dtype=np.uint8))

    # Step 2: 生成法線貼圖
    if normal_source == "direct":
        # 直接從原始圖片生成，使用法線貼圖專用參數
        processed_input = input_image.copy()
        
        # 應用法線貼圖的獨立預處理參數
        processed_input = apply_gamma(processed_input, normal_level)
        if normal_blur_sharp != 0:
            # 對 RGB 每個通道應用模糊/銳化
            channels = cv2.split(processed_input)
            processed_channels = [apply_blur_sharp(ch, normal_blur_sharp) for ch in channels]
            processed_input = cv2.merge(processed_channels)

        normal_map_image = generate_normal_from_image(
            processed_input, intensity=strength, algorithm=algorithm, height_method=height_method
        )
    
    elif normal_source == "midas" and midas is not None and processed_depth is not None:
        # 從已處理的 MiDaS 深度圖生成法線貼圖
        # 注意：這種模式下，法線圖會受到深度圖參數的影響
        normal_map_image = depth_to_normal(processed_depth, intensity=strength, algorithm=algorithm)
    else:
        # 回退到預設方法
        normal_map_image = generate_normal_from_image(
            input_image, intensity=strength, algorithm=algorithm, height_method=height_method
        )
    
    print("Processing complete.")
    return normal_map_image, depth_pil

# 批次處理函式保持不變，但需要更新呼叫 process_image 的參數
def process_batch(files, strength, level, blur_sharp, algorithm, normal_source, height_method, output_path):
    if not files:
        return "沒有選擇檔案"
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 處理批次輸出路徑
    actual_path, success = ensure_path_exists(output_path)
    batch_folder = os.path.join(actual_path, f"batch_{timestamp}")
    os.makedirs(batch_folder, exist_ok=True)
    
    for i, file in enumerate(files):
        try:
            img_array = np.array(Image.open(file.name))
            normal_map, _ = process_image(img_array, strength, level, blur_sharp, algorithm, normal_source, height_method, level, blur_sharp)
            
            filename = f"normal_map_{i+1:03d}.png"
            save_path = os.path.join(batch_folder, filename)
            normal_map.save(save_path)
            
            results.append(f"✓ {filename}")
        except Exception as e:
            results.append(f"✗ 檔案 {i+1} 處理失敗: {str(e)}")
    
    # 根據環境提供不同的反饋
    if batch_folder.startswith('/mnt/c/'):
        windows_path = batch_folder.replace('/mnt/c/', 'C:\\').replace('/', '\\')
        feedback = f"✅ 批次處理完成！\n📁 Windows 路徑: {windows_path}\n\n"
    else:
        feedback = f"✅ 批次處理完成！\n📁 儲存位置: {batch_folder}\n\n"
    
    return feedback + "\n".join(results)


# --- 3. 預設參數組合 ---
PRESETS = {
    "標準 (低雜訊)": {"strength": 0.01, "level": 1.0, "blur_sharp": 0, "algorithm": "smooth_sobel", "normal_source": "direct", "height_method": "luminance", "normal_level": 1.0, "normal_blur_sharp": -2},
    "平滑": {"strength": 0.008, "level": 1.0, "blur_sharp": -5, "algorithm": "sobel_5", "normal_source": "direct", "height_method": "luminance", "normal_level": 0.8, "normal_blur_sharp": -5},
    "銳利 (低雜訊)": {"strength": 0.015, "level": 1.0, "blur_sharp": 0, "algorithm": "scharr", "normal_source": "direct", "height_method": "luminance", "normal_level": 1.2, "normal_blur_sharp": -1},
    "極致細節": {"strength": 0.02, "level": 1.0, "blur_sharp": 0, "algorithm": "prewitt", "normal_source": "direct", "height_method": "max", "normal_level": 1.5, "normal_blur_sharp": 0},
    "超級平滑": {"strength": 0.005, "level": 1.0, "blur_sharp": -8, "algorithm": "smooth_sobel", "normal_source": "direct", "height_method": "average", "normal_level": 0.7, "normal_blur_sharp": -8},
    "石材專用": {"strength": 0.012, "level": 1.0, "blur_sharp": -3, "algorithm": "sobel_5", "normal_source": "direct", "height_method": "luminance", "normal_level": 1.1, "normal_blur_sharp": -3}
}

# --- 4. Gradio UI 介面 ---
with gr.Blocks(title="Advanced Normal Map Generator v3.1", theme=gr.themes.Soft()) as app:
    
    normal_map_state = gr.State()
    depth_map_state = gr.State()

    gr.Markdown(
        """
        # 🎨 進階法線貼圖生成器 v3.1
        ### Y 軸修正 | 參數分離 | 更多預設
        """
    )

    with gr.Tabs():
        with gr.TabItem("單張處理"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="numpy", label="上傳原始圖片")
                    
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
                    
                    with gr.Accordion("詳細參數", open=True):
                        strength_slider = gr.Slider(0.0, 0.2, 0.001, step=0.001, label="Normal Strength (強度)")
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
                    
                    # process_button = gr.Button("生成法線貼圖", variant="primary")
                
                with gr.Column(scale=2):
                    with gr.Row():
                        output_depth_map = gr.Image(type="pil", label="深度圖預覽 (MiDaS)")
                        output_normal_map = gr.Image(type="pil", label="法線貼圖")
                    
                with gr.Accordion("下載生成選項", open=True):
                    process_button = gr.Button("生成貼圖", variant="primary")
                    
                    # 壓縮設定
                    with gr.Row():
                        compression_level = gr.Dropdown(
                            choices=["none", "low", "medium", "high"],
                            value="medium",
                            label="壓縮等級",
                            info="none: 無壓縮, low: 輕微, medium: 平衡, high: 最大"
                        )
                        file_format = gr.Dropdown(
                            choices=["PNG", "WEBP", "JPEG"],
                            value="PNG",
                            label="檔案格式",
                            info="PNG: 無損(推薦), WEBP: 高壓縮比, JPEG: 最小檔案(有損)"
                        )
                    
                    default_path = get_default_download_path()
                    download_path_input = gr.Textbox(
                        label="下載資料夾路徑", 
                        value=default_path,
                        info="WSL 提示：使用 /mnt/c/Users/您的用戶名/Downloads 可存取 Windows 下載資料夾"
                    )
                    
                    with gr.Row():
                        quick_paths = gr.Dropdown(
                            choices=[
                                ("本地 outputs 資料夾", "outputs"),
                                ("Windows 下載資料夾", f"/mnt/c/Users/ghjgh/Downloads/normal_maps"),
                            ],
                            label="快速路徑",
                            value=None
                        )
                        set_path_btn = gr.Button("設定路徑", size="sm")
                    
                    with gr.Row():
                        download_depth_btn = gr.Button("下載深度圖", variant="secondary")
                        download_normal_btn = gr.Button("下載法線貼圖", variant="secondary")

                    download_feedback = gr.Textbox(label="下載狀態", interactive=False, lines=2)
                 
                    download_feedback = gr.Textbox(label="狀態", interactive=False, lines=4)

        # 批次處理 Tab (保持原樣，但注意其參數傳遞)
        with gr.TabItem("批次處理"):
            # ... 此處 UI 未變，您可以根據需要為其也添加獨立參數 ...
            with gr.Row():
                with gr.Column():
                    batch_files = gr.File(file_count="multiple", file_types=["image"], label="選擇多張圖片")
                    with gr.Row():
                        batch_normal_source = gr.Radio(choices=[("直接從圖片", "direct"), ("從 MiDaS 深度圖", "midas")], value="direct", label="法線貼圖來源")
                        batch_height_method = gr.Dropdown(choices=["luminance", "average", "max", "red", "green", "blue"], value="luminance", label="高度圖方式")
                    with gr.Row():
                        batch_strength = gr.Slider(0.1, 10.0, 2.0, step=0.1, label="Strength")
                        batch_level = gr.Slider(0.1, 3.0, 1.0, step=0.05, label="Level")
                    with gr.Row():
                        batch_blur_sharp = gr.Slider(-10, 10, 0, step=1, label="Blur/Sharp")
                        batch_algorithm = gr.Dropdown(choices=["sobel", "scharr", "laplacian", "prewitt"], value="sobel", label="算法")
                    batch_output_path = gr.Textbox(
                        label="批次輸出資料夾", 
                        value=get_default_download_path(),
                        info="建議使用 Windows 路徑以便在檔案總管中開啟"
                    )
                    batch_process_btn = gr.Button("開始批次處理", variant="primary")
                with gr.Column():
                    batch_result = gr.Textbox(label="批次處理結果", lines=15, interactive=False)


    # --- 事件處理 ---
    def apply_preset(preset_name):
        preset = PRESETS[preset_name]
        return (
            preset["strength"],
            preset["level"], 
            preset["blur_sharp"],
            preset["algorithm"],
            preset["normal_source"],
            preset["height_method"],
            preset["normal_level"],
            preset["normal_blur_sharp"]
        )

    def on_process_click(image, strength, level, blur_sharp, algorithm, normal_source, height_method, normal_level, normal_blur_sharp):
        normal_map, depth_map = process_image(image, strength, level, blur_sharp, algorithm, normal_source, height_method, normal_level, normal_blur_sharp)
        return normal_map, depth_map, normal_map, depth_map

    def save_image(image, path, prefix):
        if image is None:
            return "沒有可下載的圖片"
        try:
            os.makedirs(path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(path, f"{prefix}_{timestamp}.png")
            image.save(file_path)
            return f"成功儲存至: {file_path}"
        except Exception as e:
            return f"儲存失敗: {e}"

    apply_preset_btn.click(
        apply_preset,
        inputs=[preset_dropdown],
        outputs=[
            strength_slider, level_slider, blur_sharp_slider, algorithm_dropdown, 
            normal_source_radio, height_method_dropdown, 
            normal_level_slider, normal_blur_sharp_slider # --- MODIFIED: 新增 output
        ]
    )

    process_button.click(
        on_process_click,
        inputs=[
            input_image, strength_slider, level_slider, blur_sharp_slider, 
            algorithm_dropdown, normal_source_radio, height_method_dropdown,
            normal_level_slider, normal_blur_sharp_slider # --- MODIFIED: 新增 input
        ],
        outputs=[output_normal_map, output_depth_map, normal_map_state, depth_map_state]
    )


    batch_process_btn.click(
        process_batch,
        inputs=[batch_files, batch_strength, batch_level, batch_blur_sharp, batch_algorithm, batch_normal_source, batch_height_method, batch_output_path],
        outputs=[batch_result]
    )
    # 新增路徑設定事件
    def set_quick_path(selected_path):
        return selected_path if selected_path else default_path

    set_path_btn.click(
        set_quick_path,
        inputs=[quick_paths],
        outputs=[download_path_input]
    )

    download_depth_btn.click(
    lambda img, path, comp_level, fmt: save_image_with_compression(img, path, "depth_map", comp_level, fmt),
    inputs=[depth_map_state, download_path_input, compression_level, file_format],
    outputs=[download_feedback]
    )

    download_normal_btn.click(
        lambda img, path, comp_level, fmt: save_image_with_compression(img, path, "normal_map", comp_level, fmt),
        inputs=[normal_map_state, download_path_input, compression_level, file_format],
        outputs=[download_feedback]
    )   



if __name__ == "__main__":
    app.launch(share=True)