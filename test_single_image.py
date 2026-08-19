import os
import math
import cv2
import numpy as np
import torch

# 定义类别映射（完全复制您的 yaml 配置）
CLASS_NAMES = {
    0: "crossing",
    1: "stop",
    2: "countdown_blank",
    3: "countdown_go",
    4: "go",
    5: "blank",
    6: "countdown_stop"
}


# =======================================================
# 1. 核心 Mask 生成函数（支持传入带 class_id 的 5 列数据）
# =======================================================
def generate_scale_aware_mask(img_shape, bboxes, gamma=15.0):
    H, W = img_shape
    img_area = H * W
    mask = torch.ones((1, 1, H, W), dtype=torch.float32)

    if bboxes is not None and len(bboxes) > 0:
        for box in bboxes:
            xmin, ymin, xmax, ymax = map(int, box[:4])  # 只取前4列坐标
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(W, xmax), min(H, ymax)

            box_w = xmax - xmin
            box_h = ymax - ymin
            box_area = box_w * box_h
            relative_area = box_area / img_area

            # 核心映射：面积越小，m_val 越接近 0（噪声越小，保护越强）
            m_val = 1.0 - math.exp(-gamma * relative_area)

            # 局部区域赋予抗噪权重
            mask[:, :, ymin:ymax, xmin:xmax] = torch.minimum(
                mask[:, :, ymin:ymax, xmin:xmax],
                torch.tensor(m_val, dtype=torch.float32)
            )
    return mask


# =======================================================
# 2. 修正后的 YOLO 标签解析器（保留 class_id）
# =======================================================
def parse_yolo_txt(txt_path, img_w, img_h):
    bboxes = []
    if not os.path.exists(txt_path):
        print(f"❌ 找不到标签文件: {txt_path}")
        return None
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:])
                # 转换为像素绝对坐标
                xmin = int((x_c - w / 2) * img_w)
                ymin = int((y_c - h / 2) * img_h)
                xmax = int((x_c + w / 2) * img_w)
                ymax = int((y_c + h / 2) * img_h)
                # 存储坐标和类别 ID
                bboxes.append([xmin, ymin, xmax, ymax, class_id])
    return bboxes


# =======================================================
# 3. 主测试流水线
# =======================================================
def test_pipeline():
    # 填入您提供的绝对路径
    img_path = r"D:\project1\DMZ\train\images\heon_IMG_0529_JPG.rf.cd84870cb8819267963feaa3336af317.jpg"
    txt_path = r"D:\project1\DMZ\train\labels\heon_IMG_0529_JPG.rf.cd84870cb8819267963feaa3336af317.txt"

    if not os.path.exists(img_path):
        print(f"❌ 错误：找不到原图，请检查路径：{img_path}")
        return

    # 读取原图
    img_bgr = cv2.imread(img_path)
    H, W, C = img_bgr.shape
    print(f"📷 成功读取图片，分辨率为: {W}x{H}")

    # 解析真实 YOLO 标签
    bboxes = parse_yolo_txt(txt_path, W, H)
    if bboxes:
        print(f"🎯 成功加载真实标签，共包含 {len(bboxes)} 个标注目标。")
        for box in bboxes:
            name = CLASS_NAMES.get(box[4], f"Unknown-{box[4]}")
            print(f"   --> 发现目标类别: [{name}], 坐标: {box[:4]}")
    else:
        print("❌ 标签解析失败，请检查文本文件。")
        return

    # 转换为 PyTorch Tensor: [1, C, H, W]
    img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # 生成真实的各向异性方差掩码
    mask = generate_scale_aware_mask((H, W), bboxes, gamma=15.0)

    # 物理退化锚点 (暗度 0.74，噪声强度 0.18)
    alpha = 0.74
    sigma = 0.18
    noise = torch.randn_like(img_tensor)

    # 核心退化公式
    degraded_tensor = alpha * img_tensor + (mask * noise * sigma)
    degraded_tensor = torch.clamp(degraded_tensor, 0.0, 1.0)

    # 转换回 OpenCV 图像
    result_img = (degraded_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    # 可视化真实标注：在图像上绘制绿框和类别文字
    for box in bboxes:
        xmin, ymin, xmax, ymax, cls_id = box
        class_name = CLASS_NAMES.get(cls_id, str(cls_id))

        # 绘制边界框
        cv2.rectangle(result_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        # 绘制类别文本标签
        cv2.putText(result_img, class_name, (xmin, max(ymin - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 保存测试结果
    output_name = "real_low_light_protected_test.jpg"
    cv2.imwrite(output_name, result_img)
    print(f"🎉 试验成功！真实的夜间测试图已保存在: {output_name}")


if __name__ == "__main__":
    test_pipeline()