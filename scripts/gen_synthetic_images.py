"""Generate synthetic satellite-like imagery for documentation.

This script generates synthetic satellite imagery examples for use in
the documentation. Output is saved to docs/assets/images/.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import io

# Ensure reproducibility
np.random.seed(42)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "assets" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_terrain_noise(shape: tuple, scale: float = 0.1) -> np.ndarray:
    """Generate terrain-like noise using multiple octaves."""
    result = np.zeros(shape)
    amplitude = 1.0
    frequency = scale
    
    for _ in range(4):  # 4 octaves
        x = np.linspace(0, shape[1] * frequency, shape[1])
        y = np.linspace(0, shape[0] * frequency, shape[0])
        xx, yy = np.meshgrid(x, y)
        
        # Simple perlin-like noise approximation
        noise = np.sin(xx) * np.cos(yy) + np.sin(xx * 2.3) * np.cos(yy * 1.8)
        noise += np.sin(xx * 0.5 + 1.2) * np.cos(yy * 0.7 + 0.8)
        result += amplitude * noise
        
        amplitude *= 0.5
        frequency *= 2.0
    
    # Normalize to [0, 1]
    result = (result - result.min()) / (result.max() - result.min())
    return result


def generate_rgb_composite():
    """Generate synthetic RGB satellite composite."""
    size = (512, 512)
    
    # Generate base terrain
    terrain = generate_terrain_noise(size, scale=0.02)
    
    # Create land cover classes based on terrain height
    # Blue for water (low terrain)
    # Green for vegetation (mid terrain)
    # Brown for bare soil (high terrain)
    
    r = np.zeros(size)
    g = np.zeros(size)
    b = np.zeros(size)
    
    # Water (blue)
    water_mask = terrain < 0.3
    b[water_mask] = 0.3 + 0.3 * terrain[water_mask] / 0.3
    g[water_mask] = 0.2 + 0.2 * terrain[water_mask] / 0.3
    r[water_mask] = 0.1 + 0.1 * terrain[water_mask] / 0.3
    
    # Vegetation (green)
    veg_mask = (terrain >= 0.3) & (terrain < 0.7)
    g[veg_mask] = 0.3 + 0.4 * (terrain[veg_mask] - 0.3) / 0.4
    r[veg_mask] = 0.1 + 0.2 * (terrain[veg_mask] - 0.3) / 0.4
    b[veg_mask] = 0.05 + 0.1 * (terrain[veg_mask] - 0.3) / 0.4
    
    # Bare soil / urban (brown/gray)
    urban_mask = terrain >= 0.7
    r[urban_mask] = 0.4 + 0.2 * (terrain[urban_mask] - 0.7) / 0.3
    g[urban_mask] = 0.35 + 0.15 * (terrain[urban_mask] - 0.7) / 0.3
    b[urban_mask] = 0.3 + 0.1 * (terrain[urban_mask] - 0.7) / 0.3
    
    # Add variation noise
    noise = np.random.normal(0, 0.02, size)
    r = np.clip(r + noise, 0, 1)
    g = np.clip(g + noise, 0, 1)
    b = np.clip(b + noise, 0, 1)
    
    # Convert to uint8
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    
    img = Image.fromarray(rgb)
    img.save(OUTPUT_DIR / "synthetic_rgb_composite.png")
    print("Generated: synthetic_rgb_composite.png")


def generate_ndvi_visualization():
    """Generate synthetic NDVI visualization."""
    size = (512, 512)
    
    # Generate vegetation pattern
    terrain = generate_terrain_noise(size, scale=0.015)
    
    # NDVI ranges from -1 to 1, but typically positive for vegetation
    ndvi = terrain * 2 - 1  # Range roughly [-1, 1]
    
    # Add some water (very negative NDVI)
    water_noise = generate_terrain_noise(size, scale=0.03)
    water_mask = water_noise < 0.3
    ndvi[water_mask] = -0.3 + np.random.normal(0, 0.05, np.sum(water_mask))
    
    # Normalize to [0, 1] for colormap
    ndvi_norm = (ndvi + 1) / 2
    ndvi_norm = np.clip(ndvi_norm, 0, 1)
    
    # Apply NDVI colormap (brown -> yellow -> green)
    r = np.zeros(size)
    g = np.zeros(size)
    b = np.zeros(size)
    
    # Low NDVI: brown/tan
    low_mask = ndvi_norm < 0.4
    r[low_mask] = 0.6 + 0.2 * ndvi_norm[low_mask] / 0.4
    g[low_mask] = 0.4 + 0.3 * ndvi_norm[low_mask] / 0.4
    b[low_mask] = 0.2 + 0.1 * ndvi_norm[low_mask] / 0.4
    
    # Medium NDVI: yellow-green
    mid_mask = (ndvi_norm >= 0.4) & (ndvi_norm < 0.7)
    t = (ndvi_norm[mid_mask] - 0.4) / 0.3
    r[mid_mask] = 0.8 - 0.6 * t
    g[mid_mask] = 0.7 + 0.2 * t
    b[mid_mask] = 0.3 - 0.2 * t
    
    # High NDVI: dark green
    high_mask = ndvi_norm >= 0.7
    t = (ndvi_norm[high_mask] - 0.7) / 0.3
    r[high_mask] = 0.2 - 0.1 * t
    g[high_mask] = 0.9 - 0.2 * t
    b[high_mask] = 0.1
    
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    
    img = Image.fromarray(rgb)
    img.save(OUTPUT_DIR / "synthetic_ndvi.png")
    print("Generated: synthetic_ndvi.png")


def generate_change_detection_example():
    """Generate pre/post/change detection example."""
    size = (512, 512)
    
    # Pre-image terrain
    terrain = generate_terrain_noise(size, scale=0.02)
    
    # Pre-image (mostly vegetation)
    r_pre = 0.2 + 0.2 * terrain
    g_pre = 0.4 + 0.3 * terrain
    b_pre = 0.1 + 0.1 * terrain
    
    # Add "deforestation" patches for post-image
    deforest_noise = generate_terrain_noise(size, scale=0.05)
    deforest_mask = (deforest_noise > 0.6) & (terrain > 0.3) & (terrain < 0.7)
    
    # Post-image
    r_post = r_pre.copy()
    g_post = g_pre.copy()
    b_post = b_pre.copy()
    
    # Deforested areas become brown
    r_post[deforest_mask] = 0.5 + np.random.normal(0, 0.03, np.sum(deforest_mask))
    g_post[deforest_mask] = 0.4 + np.random.normal(0, 0.03, np.sum(deforest_mask))
    b_post[deforest_mask] = 0.3 + np.random.normal(0, 0.02, np.sum(deforest_mask))
    
    # Clip and convert
    r_pre, g_pre, b_pre = [np.clip(x, 0, 1) for x in [r_pre, g_pre, b_pre]]
    r_post, g_post, b_post = [np.clip(x, 0, 1) for x in [r_post, g_post, b_post]]
    
    pre_rgb = np.stack([r_pre, g_pre, b_pre], axis=-1)
    post_rgb = np.stack([r_post, g_post, b_post], axis=-1)
    
    # Create change mask (red for deforestation)
    change_rgb = np.stack([
        deforest_mask.astype(float) * 0.8,
        deforest_mask.astype(float) * 0.1,
        deforest_mask.astype(float) * 0.1,
    ], axis=-1)
    
    # Save all three
    for name, arr in [("pre_image", pre_rgb), ("post_image", post_rgb), ("change_mask", change_rgb)]:
        img = Image.fromarray((arr * 255).astype(np.uint8))
        img.save(OUTPUT_DIR / f"synthetic_{name}.png")
        print(f"Generated: synthetic_{name}.png")


def generate_segmentation_example():
    """Generate segmentation input/output example."""
    size = (512, 512)
    
    # Generate terrain for class distribution
    terrain = generate_terrain_noise(size, scale=0.02)
    noise2 = generate_terrain_noise(size, scale=0.04)
    
    # Create categorical segmentation
    # 0: Water, 1: Forest, 2: Agriculture, 3: Urban, 4: Bare
    segmentation = np.zeros(size, dtype=np.uint8)
    
    # Water (low terrain)
    segmentation[terrain < 0.25] = 0
    
    # Forest (mid-low terrain)
    forest_mask = (terrain >= 0.25) & (terrain < 0.5)
    segmentation[forest_mask] = 1
    
    # Agriculture (mid-high terrain with specific pattern)
    agri_mask = (terrain >= 0.5) & (terrain < 0.75) & (noise2 > 0.3)
    segmentation[agri_mask] = 2
    
    # Urban (high terrain clusters)
    urban_mask = (terrain >= 0.65) & (noise2 < 0.4)
    segmentation[urban_mask] = 3
    
    # Bare soil (very high terrain)
    segmentation[terrain >= 0.85] = 4
    
    # Color map for visualization
    colors = np.array([
        [65, 105, 225],    # Water: blue
        [34, 139, 34],     # Forest: green
        [255, 215, 0],     # Agriculture: gold
        [128, 128, 128],   # Urban: gray
        [210, 180, 140],   # Bare: tan
    ], dtype=np.uint8)
    
    seg_rgb = colors[segmentation]
    
    img = Image.fromarray(seg_rgb)
    img.save(OUTPUT_DIR / "synthetic_segmentation.png")
    print("Generated: synthetic_segmentation.png")


def main():
    """Generate all synthetic imagery."""
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 40)
    
    generate_rgb_composite()
    generate_ndvi_visualization()
    generate_change_detection_example()
    generate_segmentation_example()
    
    print("-" * 40)
    print(f"Generated 6 synthetic images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
