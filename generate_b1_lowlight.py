import os
import shutil
from pathlib import Path

SRC_ROOT = r"D:\苏雅文\Desktop\ultralytics-main\DMZ"
DST_ROOT = r"D:\苏雅文\Desktop\ultralytics-main\DMZ_B1"

SPLITS = ["train", "val"]

for split in SPLITS:
    # 正确的源目录
    src_img_dir = os.path.join(SRC_ROOT, split, "images")
    src_lbl_dir = os.path.join(SRC_ROOT, split, "labels")

    # 目标目录
    dst_img_dir = os.path.join(DST_ROOT, split, "images")
    dst_lbl_dir = os.path.join(DST_ROOT, split, "labels")

    # 创建目标目录
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    # ✅ 复制 labels
    if os.path.exists(src_lbl_dir):
        for f in os.listdir(src_lbl_dir):
            shutil.copy(os.path.join(src_lbl_dir, f), os.path.join(dst_lbl_dir, f))
        print(f"✅ {split} labels 已复制")
    else:
        print(f"❌ 找不到 labels: {src_lbl_dir}")

    # 生成低光图片 (示例)
    for img_name in os.listdir(src_img_dir):
        src_img_path = os.path.join(src_img_dir, img_name)
        # 假设你的 diffusion.sample() 已经返回 tensor
        # 这里只是示意
        # out_img = diffusion.sample(...)
        # 保存到 dst_img_dir
        # out_img.save(os.path.join(dst_img_dir, img_name))

print("✅ B1 低光数据生成完成")
