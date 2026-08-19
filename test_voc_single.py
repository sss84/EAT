import os
import math
import cv2
import numpy as np
import torch

# 1. 完全对齐你给的 VOC yaml 类别
CLASS_NAMES = {
    0: 'bicycle', 1: 'boat', 2: 'bottle', 3: 'bus', 4: 'car',
    5: 'cat', 6: 'chair', 7: 'dog', 8: 'motorbike', 9: 'person'
}

# 设定你论文里受损最严重的两个标志性小目标类别 ID
TARGET_SMALL_CLASSES = [1, 2]  # 1: boat, 2: bottle

# 路径配置
IMG_DIR = r"D:\VOC_YOLO\train\images"
TXT_DIR = r"D:\VOC_YOLO\train\labels"


# =======================================================
# 核心函数：空间尺度感知 Mask 生成
# =======================================================
def generate_scale_aware_mask(img_shape, bboxes, gamma=15.0):
    H, W = img_shape
    img_area = H * W
    mask = torch.ones((1, 1, H, W), dtype=torch.float32)

    if bboxes is not None and len(bboxes) > 0:
        for box in bboxes:
            xmin, ymin, xmax, ymax = map(int, box[:4])
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(W, xmax), min(H, ymax)

            box_w = xmax - xmin
            box_h = ymax - ymin
            box_area = box_w * box_h
            relative_area = box_area / img_area

            # 面积越小，m_val 越接近 0（强保护，完全隔绝噪声）
            m_val = 1.0 - math.exp(-gamma * relative_area)

            mask[:, :, ymin:ymax, xmin:xmax] = torch.minimum(
                mask[:, :, ymin:ymax, xmin:xmax],
                torch.tensor(m_val, dtype=torch.float32)
            )
    return mask


# =======================================================
# 核心函数：解析单个 YOLO 文本标签
# =======================================================
def parse_yolo_txt(txt_path, img_w, img_h):
    bboxes = []
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:])
                # 转换绝对像素坐标
                xmin = int((x_c - w / 2) * img_w)
                ymin = int((y_c - h / 2) * img_h)
                xmax = int((x_c + w / 2) * img_w)
                ymax = int((y_c + h / 2) * img_h)
                # 计算相对面积
                rel_area = w * h
                bboxes.append([xmin, ymin, xmax, ymax, class_id, rel_area])
    return bboxes


# =======================================================
# 核心函数：全自动挖掘满足条件的“微小目标”图片
# =======================================================
def auto_find_small_target_image():
    print("🔍 开始自动扫描标签库，正在为你疯狂寻找论文中的‘小船’或‘瓶子’...")

    best_match = None
    min_area = 1.0  # 用来记录找到的最小目标面积

    # 遍历标签文件夹
    for txt_name in os.listdir(TXT_DIR):
        if not txt_name.endswith('.txt'):
            continue
        txt_path = os.path.join(TXT_DIR, txt_name)

        # 预读看有没有目标
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                rel_area = w * h

                # 优先寻找boat/bottle，且相对面积小于 0.01 (即占全图不足1%)
                if class_id in TARGET_SMALL_CLASSES and rel_area < 0.01:
                    if rel_area < min_area:
                        min_area = rel_area
                        best_match = txt_name

    # 如果没找到boat/bottle，退而求其次寻找全图范围内任意小于 0.005 的极致微小目标（比如远处的person或car）
    if not best_match:
        print("⚠️ 未在阈值内发现小船或瓶子，正在为你检索全图库中的极致微小目标...")
        for txt_name in os.listdir(TXT_DIR):
            if not txt_name.endswith('.txt'):
                continue
            txt_path = os.path.join(TXT_DIR, txt_name)
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        rel_area = float(parts[3]) * float(parts[4])
                        if rel_area < 0.005 and rel_area < min_area:
                            min_area = rel_area
                            best_match = txt_name

    if best_match:
        # 寻找对应的图片文件名（尝试常见的几种后缀）
        base_name = os.path.splitext(best_match)[0]
        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
            img_path = os.path.join(IMG_DIR, base_name + ext)
            if os.path.exists(img_path):
                return img_path, os.path.join(TXT_DIR, best_match)

    return None, None


# =======================================================
# 主流水线
# =======================================================
def main():
    img_path, txt_path = auto_find_small_target_image()

    if not img_path or not txt_path:
        print("❌ 扫描失败：没有在对应目录下找到任何有效的图片和标签对，请检查路径是否正确。")
        return

    print(f"\n🎯 成功抓取到最具代表性的小目标测试对！")
    print(f"   🖼️ 图片路径: {img_path}")
    print(f"   📝 标签路径: {txt_path}")

    # 读取原图
    img_bgr = cv2.imread(img_path)
    H, W, C = img_bgr.shape
    print(f"📷 图像分辨率: {W}x{H}")

    # 解析标签
    bboxes = parse_yolo_txt(txt_path, W, H)

    print("📊 该图包含以下目标：")
    for box in bboxes:
        name = CLASS_NAMES.get(box[4], f"Unknown-{box[4]}")
        print(f"   --> 类别: [{name}], 相对全图面积: {box[5] * 100:.3f}%")

    # 转为 Tensor
    img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # 生成尺度自适应 Mask
    mask = generate_scale_aware_mask((H, W), bboxes, gamma=15.0)

    # 沿用你期刊论文和测试的最优强退化参数 (暗度0.74，噪声强度0.18)
    alpha = 0.74
    sigma = 0.18
    noise = torch.randn_like(img_tensor)

    # 核心前向加噪动力学公式
    degraded_tensor = alpha * img_tensor + (mask * noise * sigma)
    degraded_tensor = torch.clamp(degraded_tensor, 0.0, 1.0)

    # 转回 OpenCV 格式
    result_img = (degraded_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    # 在图上把绿框和类别画出来，方便你肉眼审查边界
    for box in bboxes:
        xmin, ymin, xmax, ymax, cls_id, _ = box
        class_name = CLASS_NAMES.get(cls_id, str(cls_id))

        # 绘制边界框
        cv2.rectangle(result_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        # 绘制文本
        cv2.putText(result_img, class_name, (xmin, max(ymin - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 保存测试结果
    output_name = "voc_low_light_protected_test.jpg"
    cv2.imwrite(output_name, result_img)
    print(f"\n🎉 试验大成功！VOC夜间测试图已保存在当前目录下: {output_name}")
    print("💡 赶快打开这张图，放大看里面的 boat 或 bottle，看看高频噪声是不是在小框内被完美隔离了！")


if __name__ == "__main__":
    main()