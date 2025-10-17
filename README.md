# Advanced Normal Map Generator v3.2

A research-grade normal map processing tool integrating **MiDaS depth estimation**, **edge-based normal reconstruction**, and **GPU texture output pipelines (DDS BC5/BC4)**. Designed for consistent normal generation across **vision research**, **graphics experiments**, and **engine-level deployment (Unity/Unreal/DirectX/OpenGL)**.


---
## 🔬 Project Overview
This tool was built for **precise height-to-normal mapping**, **depth-aware material reconstruction**, and **engine-friendly normal output**. It supports both **Computer Vision (CV)** and **Graphics (DCC/CG/Game Dev)** workflows.

Unlike typical normal map generators, this system provides:
- **Full algorithmic control** (Sobel/Scharr/Prewitt edge extraction)
- **Differentiated parameter pipelines** (independent depth/normal preprocessing)
- **MiDaS depth refinement** (depth-to-normal conversion)
- **Green channel orientation control** (OpenGL ⬆️ / DirectX/Unity ⬇️)
- **DDS GPU compression output** (BC5 for normal, BC4 for displacement/height)


---
## ✨ Key Features
| Feature | Description |
|----------|-------------|
| ✅ Direct normal extraction | From RGB luminance height estimation |
| ✅ MiDaS integration | Depth-to-normal from monocular depth |
| ✅ Independent pipelines | Separate gamma/filters for depth & normal |
| ✅ Noise control | Bilateral/median prefilters |
| ✅ Green Channel Orientation | **OpenGL (Green-Up)** / **DirectX (Green-Down)** |
| ✅ GPU DDS output | **BC5 normal** / **BC4 depth-height** via texconv |
| ✅ WebP Lossless | Compact + high fidelity normal encoding |
| ✅ Batch processing | Multi-file automation |
| ✅ WSL-friendly paths | Automatic Windows path translation |


---
## 🔧 Installation
```bash
git clone https://github.com/ghjghjghkimo/Normal_map_generator.git
cd Normal_map_generator
python -m venv nmgen_env
# Windows
env\Scripts\activate
# macOS/WSL/Linux
source nmgen_env/bin/activate
pip install -r requirements.txt
```

✅ **Optional (for DDS output)** – Install `texconv.exe` (DirectXTex)  
https://github.com/microsoft/DirectXTex


---
## 🚀 Launch
```bash
python v3_2.py
```
Open browser: **http://127.0.0.1:7860**

---
## 🧭 Normal Space Control
This tool supports **engine-specific normal orientation**:

| Engine | Green Axis | Setting |
|--------|------------|---------|
| Unity / Unreal / DirectX | Green ↓ Down | ✅ Default |
| OpenGL / Vulkan | Green ↑ Up | enable `OpenGL mode` |

You can also enforce correct normal orientation during **DDS (BC5) export** (`-inverty`).


---
## 🧱 Output Formats
| Format | Purpose | Notes |
|--------|---------|--------|
| PNG | Standard normal output | ✅ Lossless |
| WebP (Lossless) | Fast & small | ⚡ Recommended |
| JPEG | **Not recommended** for normals | ❌ Lossy artifacts |
| DDS (BC5) | GPU-optimized normal | ✅ Best for engines |
| DDS (BC4) | Height/Mask maps | Compact grayscale |


---
## 🎛 Normal Generation Algorithms
| Algorithm | Behavior | Best Use |
|-----------|----------|----------|
| smooth_sobel | Smoothed Sobel edges | General-purpose |
| sobel | Standard edge detection | Hard surfaces |
| sobel_5 | Larger edge sensitivity | Mid-textures |
| scharr | High gradient contrast | Metals, sci-fi |
| prewitt | Soft gradients | Organic surfaces |


---
## 🧪 Parameter Guide
| Parameter | Description | Range |
|-----------|-------------|--------|
| Normal Strength | Z-axis intensity | 0.001–0.2 |
| Normal Gamma | Shape amplification | 0.1–3.0 |
| Normal Blur/Sharp | Bilateral filter control | -10 to 10 |
| Depth Gamma | MiDaS leveling | 0.1–3.0 |
| Depth Blur/Sharp | Refine depth smoothness | -10 to 10 |


---
## 🛠 Presets
✅ Built-in presets include:
- `標準 (低雜訊)` – Balanced
- `石材專用` – High solidity
- `平滑` – Clean edges
- `銳利` – Maximum detail
- `極致細節` – Feature boost


---
## 🗃 Batch Processing
Supports **batch image generation** with **shared parameter control**. Produces structured output including:
```
mytexture_normal.png
mytexture_depth.png
```
Or DDS engine form:
```
mytexture_normal.dds  (BC5)
mytexture_depth.dds   (BC4)
```


---
## 🧩 WSL Path Support
If using **WSL**, output paths are translated automatically:
```
/mnt/c/Users/<username>/Downloads/normal_maps
```
This makes results visible in **Windows File Explorer**.


---
## 🔒 Image Integrity
✅ WebP export here uses **lossless** mode  
✅ Avoid JPEG for normals (breaks tangent space)  
✅ DDS export keeps **engine-ready vector precision**


---
## 📌 Requirements
- Python 3.8+  
- PyTorch (CUDA optional)  
- OpenCV + NumPy + Pillow  
- Gradio UI  
- (Optional) texconv for DDS


---
## 📜 License
MIT License


---
## 🤝 Contribution
PRs welcome — especially improvements for:
- GPU normal refinement
- Depth nonlinearity correction
- Vulkan/Metal oriented normal conventions


---
## ✉️ Contact
For collaboration or engine integration questions, feel free to reach out.

---
> Research utility version — Structured pipeline for depth-assisted normal map synthesis.

