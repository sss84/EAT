"""
exdark_brightness_histogram.py
ExDark dataset brightness histogram for all used categories
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'

# ==================== Configuration ====================
EXDARK_IMAGES_DIR = Path(r"D:\ExDark\images")
OUTPUT_DIR = Path(r"D:\EXDARK_YOLO")

# Categories used in the experiment (10 classes)
CATEGORIES = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Dog', 'Motorbike', 'People']

print("=" * 60)
print("ExDark Dataset Brightness Analysis")
print("=" * 60)
print(f"Categories: {CATEGORIES}")
print(f"Image directory: {EXDARK_IMAGES_DIR}")
print("=" * 60)


# ==================== Compute Brightness ====================
def compute_brightness(image_path):
    """Compute average brightness of an image (0-255)"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


# Collect brightness for all images
brightness_values = []
category_counts = {}

print("\nProcessing images...")

for cat in CATEGORIES:
    cat_dir = EXDARK_IMAGES_DIR / cat
    if not cat_dir.exists():
        print(f"  Warning: Directory not found: {cat_dir}")
        continue

    # Get all images in this category
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        image_files.extend(cat_dir.glob(ext))

    cat_count = 0
    for img_path in image_files:
        b = compute_brightness(img_path)
        if b is not None:
            brightness_values.append(b)
            cat_count += 1

    category_counts[cat] = cat_count
    print(f"  {cat}: {cat_count} images")

print(f"\nTotal images processed: {len(brightness_values)}")

# ==================== Statistics ====================
brightness_array = np.array(brightness_values)
mean_brightness = np.mean(brightness_array)
std_brightness = np.std(brightness_array)
min_brightness = np.min(brightness_array)
max_brightness = np.max(brightness_array)
median_brightness = np.median(brightness_array)
q25 = np.percentile(brightness_array, 25)
q75 = np.percentile(brightness_array, 75)

print("\n" + "=" * 60)
print("Brightness Statistics")
print("=" * 60)
print(f"Total images: {len(brightness_array)}")
print(f"Mean brightness: {mean_brightness:.2f} ± {std_brightness:.2f}")
print(f"Median brightness: {median_brightness:.2f}")
print(f"25th percentile: {q25:.2f}")
print(f"75th percentile: {q75:.2f}")
print(f"Min brightness: {min_brightness:.2f}")
print(f"Max brightness: {max_brightness:.2f}")

# ==================== Save Statistics to File ====================
stats_file = OUTPUT_DIR / "exdark_brightness_stats.txt"
with open(stats_file, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("ExDark Dataset Brightness Statistics\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Categories used: {CATEGORIES}\n")
    f.write(f"Total images: {len(brightness_array)}\n\n")
    f.write(f"Mean brightness: {mean_brightness:.2f} ± {std_brightness:.2f}\n")
    f.write(f"Median brightness: {median_brightness:.2f}\n")
    f.write(f"25th percentile: {q25:.2f}\n")
    f.write(f"75th percentile: {q75:.2f}\n")
    f.write(f"Min brightness: {min_brightness:.2f}\n")
    f.write(f"Max brightness: {max_brightness:.2f}\n\n")
    f.write("Category counts:\n")
    for cat, count in category_counts.items():
        f.write(f"  {cat}: {count}\n")

print(f"\n✅ Statistics saved to: {stats_file}")

# ==================== Plot: Histogram ====================
fig, ax = plt.subplots(figsize=(12, 8))

# Histogram
n, bins, patches = ax.hist(brightness_array, bins=60, color='steelblue', edgecolor='white', alpha=0.8)

# Add vertical lines for statistics
ax.axvline(mean_brightness, color='red', linestyle='--', linewidth=2,
           label=f'Mean: {mean_brightness:.1f}')
ax.axvline(median_brightness, color='green', linestyle='--', linewidth=2,
           label=f'Median: {median_brightness:.1f}')
ax.axvline(q25, color='orange', linestyle=':', linewidth=1.5,
           label=f'25th percentile: {q25:.1f}')
ax.axvline(q75, color='orange', linestyle=':', linewidth=1.5,
           label=f'75th percentile: {q75:.1f}')

ax.set_xlabel('Brightness (0-255)', fontsize=14)
ax.set_ylabel('Number of Images', fontsize=14)
ax.set_title(
    f'ExDark Dataset Brightness Distribution\n(n = {len(brightness_array)} images, {len(CATEGORIES)} categories)',
    fontsize=14)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

# Add text box with statistics
textstr = f'Mean: {mean_brightness:.1f} ± {std_brightness:.1f}\nMedian: {median_brightness:.1f}\nRange: [{min_brightness:.0f}, {max_brightness:.0f}]'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exdark_brightness_histogram.png", dpi=150, bbox_inches='tight')
print(f"✅ Figure saved to: {OUTPUT_DIR / 'exdark_brightness_histogram.png'}")

# ==================== Plot: Cumulative Distribution ====================
fig, ax = plt.subplots(figsize=(10, 6))

sorted_brightness = np.sort(brightness_array)
cumulative = np.arange(1, len(sorted_brightness) + 1) / len(sorted_brightness) * 100

ax.plot(sorted_brightness, cumulative, 'b-', linewidth=2, label='Cumulative')
ax.axhline(50, color='gray', linestyle='--', alpha=0.7, label='50% (median)')
ax.axhline(25, color='gray', linestyle='--', alpha=0.5, label='25%')
ax.axhline(75, color='gray', linestyle='--', alpha=0.5, label='75%')

# Mark median point
median_idx = np.searchsorted(sorted_brightness, median_brightness)
ax.plot(median_brightness, cumulative[median_idx], 'ro', markersize=8, label=f'Median: {median_brightness:.1f}')

ax.set_xlabel('Brightness (0-255)', fontsize=14)
ax.set_ylabel('Cumulative Percentage (%)', fontsize=14)
ax.set_title(f'ExDark Dataset Cumulative Brightness Distribution\n(n = {len(brightness_array)} images)', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exdark_brightness_cumulative.png", dpi=150, bbox_inches='tight')
print(f"✅ Figure saved to: {OUTPUT_DIR / 'exdark_brightness_cumulative.png'}")

# ==================== Summary ====================
print("\n" + "=" * 60)
print("Analysis Complete")
print("=" * 60)
print(f"\nOutput files:")
print(f"  - Statistics: {stats_file}")
print(f"  - Histogram: {OUTPUT_DIR / 'exdark_brightness_histogram.png'}")
print(f"  - Cumulative: {OUTPUT_DIR / 'exdark_brightness_cumulative.png'}")

# ==================== Print Summary for Paper ====================
print("\n" + "=" * 60)
print("Summary for Paper")
print("=" * 60)
print(f"ExDark dataset brightness statistics:")
print(f"  - Total images: {len(brightness_array)}")
print(f"  - Mean brightness: {mean_brightness:.1f} ± {std_brightness:.1f}")
print(f"  - Median brightness: {median_brightness:.1f}")
print(f"  - Brightness range: [{min_brightness:.0f}, {max_brightness:.0f}]")