# 📚 Documentation & Examples

This folder contains additional resources and examples for the Advanced Normal Map Generator.

## 📁 Folder Structure

```
docs/
├── README.md (this file)
└── examples/
    ├── sample_input.jpg          # Original texture input
    ├── sample_normal.png         # Generated normal map (PNG format)
    ├── sample_normal_webp.webp   # Same normal map (WebP format for comparison)
    └── sample_depth.webp         # Generated depth map
```

## 🎨 Examples

### Sample Processing Results

The `examples/` folder contains a complete processing example showing:

1. **Input**: Original texture (JPG)
2. **Output**: 
   - Normal map in PNG format (best for archives & engines)
   - Normal map in WebP format (compact, web-friendly)
   - Depth map in WebP format (efficient grayscale)

### Why Multiple Formats?

| Format | Best For | File Size | Quality |
|--------|----------|-----------|---------|
| **PNG** | Game engines, archival | Larger | 100% lossless |
| **WebP** | Web distribution, demos | 30-40% smaller | Lossless |
| **JPEG** | ❌ NOT recommended | Smallest | Lossy artifacts |

## 🚀 Quick Reference

### Using Sample Images
You can use the sample images to test the tool's capabilities:

```bash
# Place sample_input.jpg in the app
# Processing with different algorithms will produce similar but slightly varied results

# Algorithm Impact:
# - smooth_sobel: Balanced results (shown in samples)
# - sobel: Sharper edges
# - scharr: High contrast
# - prewitt: Softer gradients
# - sobel_5: Larger edge emphasis
```

### Testing Batch Processing
1. Copy `sample_input.jpg` multiple times with different names
2. Upload all copies to batch processing
3. Observe consistent results with auto-generated output names

## 📖 For More Information

- See main [README.md](../README.md) for full documentation
- Check **🎯 Quick Start Guide** section in main README
- View **🧱 Output Formats** table for format comparison
- Review **🎛 Normal Generation Algorithms** for algorithm details

---
> These examples demonstrate the tool's capability to generate engine-ready normal maps from standard textures.
