# Normal Map Generator v5

Advanced Normal Map Generator with MiDaS depth estimation and multiple edge detection algorithms.

## 🎯 Features

- **Multiple Edge Detection Algorithms**: Sobel, Scharr, Prewitt, and more
- **MiDaS Integration**: High-quality depth estimation using DPT-Large model
- **Independent Parameter Controls**: Separate settings for depth and normal maps
- **Noise Reduction**: Advanced preprocessing to reduce artifacts
- **Batch Processing**: Process multiple images at once
- **WSL Support**: Windows-friendly download paths
- **Compression Options**: Multiple file formats and compression levels
- **Preset Combinations**: Quick parameter presets for different materials

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/ghjghjghkimo/Normal_map_generator.git
cd Normal_map_generator

# Create virtual environment
python -m venv Normal_map_env
source Normal_map_env/bin/activate  # Linux/WSL
# or
Normal_map_env\Scripts\activate     # Windows

# Install dependencies
pip install torch torchvision gradio opencv-python Pillow numpy
```

## 🚀 Usage

```bash
python v5.py
```

Then open your browser to the displayed URL (usually http://127.0.0.1:7860)

## 📱 Interface

### Single Image Processing
- Upload your image
- Choose preset or adjust parameters manually
- Select normal map generation method (Direct from image or MiDaS depth)
- Download results

### Batch Processing
- Upload multiple images
- Set processing parameters
- Batch generate all normal maps

## ⚙️ Parameters

| Parameter | Description | Range | Recommended |
|-----------|-------------|-------|-------------|
| **Strength** | Normal map intensity | 0.001-0.1 | 0.01-0.015 |
| **Depth Map Gamma** | Depth preview contrast | 0.1-3.0 | 1.0 |
| **Normal Pre-Gamma** | Pre-processing contrast | 0.1-3.0 | 0.8-1.2 |
| **Blur/Sharp** | Image smoothing | -10 to 10 | -3 to -5 |

## 🎨 Algorithms

- **smooth_sobel**: Best for reducing noise ⭐ Recommended
- **sobel_5**: Larger kernel, smoother results
- **scharr**: Enhanced edge detection
- **prewitt**: Alternative edge detection
- **sobel**: Classic Sobel operator

## 📁 Output Formats

- **PNG**: Lossless, best quality (recommended)
- **WEBP**: High compression ratio
- **JPEG**: Smallest files (not recommended for normal maps)

## 🖼️ Presets

- **標準 (低雜訊)**: Balanced settings with noise reduction
- **石材專用**: Optimized for stone/brick textures
- **平滑**: Maximum smoothing for clean results
- **銳利**: Enhanced details for high-resolution textures

## 🔧 WSL Support

For Windows Subsystem for Linux users, the tool automatically detects WSL and provides Windows-accessible download paths:

```
/mnt/c/Users/YourUsername/Downloads/normal_maps
```

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.