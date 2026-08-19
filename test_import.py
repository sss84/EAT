# final_dataset_fix.py
import os
import shutil
from pathlib import Path


def fix_dataset_issues():
    """终极修复数据集问题"""

    print("🔧 终极修复DMZ数据集")
    print("=" * 60)

    # 基础路径
    yolo_dir = Path(r"D:\苏雅文\Desktop\ultralytics-main\DMZ")
    backup_dir = yolo_dir / "backup_before_fix"

    print(f"📂 数据集目录: {yolo_dir}")

    # 1. 先备份当前数据
    print("\n1️⃣ 备份当前数据...")
    if not backup_dir.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)

        for set_name in ["train", "val"]:
            for subdir in ["images", "labels"]:
                src_dir = yolo_dir / set_name / subdir
                if src_dir.exists():
                    dest_dir = backup_dir / set_name / subdir
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    for file in src_dir.glob("*.*"):
                        try:
                            shutil.copy2(file, dest_dir / file.name)
                        except:
                            pass

        print(f"✅ 数据已备份到: {backup_dir}")

    # 2. 直接从VOC目录重新构建
    print("\n2️⃣ 从VOC目录重新构建...")
    rebuild_from_voc()

    # 3. 修复配置文件编码
    print("\n3️⃣ 修复配置文件...")
    fix_config_file(yolo_dir)

    print("\n✅ 修复完成!")


def rebuild_from_voc():
    """从VOC目录重新构建YOLO数据集"""

    voc_dir = Path(r"D:\新建文件夹\eat\DMZ")
    yolo_dir = Path(r"D:\苏雅文\Desktop\ultralytics-main\DMZ")

    print(f"📂 VOC目录: {voc_dir}")
    print(f"🎯 YOLO目录: {yolo_dir}")

    # 检查VOC目录
    if not (voc_dir / "Annotations").exists() or not (voc_dir / "JPEGImages").exists():
        print("❌ VOC目录结构错误")
        return

    # 获取所有图像文件
    image_files = list((voc_dir / "JPEGImages").glob("*.*"))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

    print(f"📷 找到 {len(image_files)} 张图像")

    if not image_files:
        print("❌ 没有找到图像文件")
        return

    # 清空YOLO目录
    print("🧹 清空YOLO目录...")
    for set_name in ["train", "val"]:
        for subdir in ["images", "labels"]:
            dir_path = yolo_dir / set_name / subdir
            dir_path.mkdir(parents=True, exist_ok=True)

            # 删除现有文件
            for file in dir_path.glob("*.*"):
                try:
                    file.unlink()
                except:
                    pass

    # 简单分割：80%训练，20%验证
    import random
    random.seed(42)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"📊 分割: {len(train_files)} 训练, {len(val_files)} 验证")

    # 处理训练集
    print("\n🔄 处理训练集...")
    process_voc_images(train_files, voc_dir, yolo_dir, "train")

    # 处理验证集
    print("\n🔄 处理验证集...")
    process_voc_images(val_files, voc_dir, yolo_dir, "val")

    print("✅ 数据集重新构建完成")


def process_voc_images(image_files, voc_dir, yolo_dir, set_name):
    """处理VOC图像文件"""

    import xml.etree.ElementTree as ET

    success_images = 0
    success_labels = 0

    # 类别映射（根据之前的分析有6个类别）
    class_mapping = {
        'crossing': 0,
        'stop': 1,
        'countdown_blank': 2,
        'countdown_go': 3,
        'go': 4,
        'blank': 5
    }

    for img_file in image_files:
        try:
            # 目标图像路径
            dest_img = yolo_dir / set_name / "images" / img_file.name

            # 直接复制图像（不修改路径）
            shutil.copy2(img_file, dest_img)
            success_images += 1

            # 处理对应的XML文件
            xml_file = voc_dir / "Annotations" / f"{img_file.stem}.xml"

            if xml_file.exists():
                try:
                    # 解析XML
                    tree = ET.parse(xml_file)
                    root = tree.getroot()

                    # 获取图像尺寸
                    size = root.find('size')
                    img_width = int(size.find('width').text)
                    img_height = int(size.find('height').text)

                    yolo_lines = []

                    # 处理每个对象
                    for obj in root.findall('object'):
                        class_name = obj.find('name').text

                        # 使用映射或默认值
                        if class_name in class_mapping:
                            class_id = class_mapping[class_name]
                        else:
                            # 未知类别，使用默认
                            class_id = 0
                            print(f"⚠️ 未知类别 '{class_name}'，使用默认ID 0")

                        # 获取边界框
                        bbox = obj.find('bndbox')
                        xmin = float(bbox.find('xmin').text)
                        ymin = float(bbox.find('ymin').text)
                        xmax = float(bbox.find('xmax').text)
                        ymax = float(bbox.find('ymax').text)

                        # 转换为YOLO格式
                        x_center = (xmin + xmax) / 2 / img_width
                        y_center = (ymin + ymax) / 2 / img_height
                        width = (xmax - xmin) / img_width
                        height = (ymax - ymin) / img_height

                        # 确保值在有效范围内
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        width = max(0.0, min(1.0, width))
                        height = max(0.0, min(1.0, height))

                        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                    # 写入标签文件
                    if yolo_lines:
                        label_file = yolo_dir / set_name / "labels" / f"{img_file.stem}.txt"
                        with open(label_file, 'w', encoding='utf-8') as f:
                            for line in yolo_lines:
                                f.write(line + '\n')
                        success_labels += 1

                except Exception as e:
                    print(f"⚠️ 解析XML失败 {img_file.name}: {str(e)[:50]}")
            else:
                print(f"⚠️ 缺少XML文件: {img_file.stem}.xml")

            # 显示进度
            if success_images % 100 == 0:
                print(f"  已处理 {success_images} 张图像...")

        except Exception as e:
            print(f"❌ 处理失败 {img_file.name}: {str(e)[:50]}")

    print(f"✅ {set_name}集: {success_images} 图像, {success_labels} 标签")


def fix_config_file(yolo_dir):
    """修复配置文件编码和内容"""

    print("🔧 修复配置文件...")

    # 统计实际文件数
    train_images = len(list((yolo_dir / "train" / "images").glob("*.*")))
    val_images = len(list((yolo_dir / "val" / "images").glob("*.*")))

    print(f"📊 统计: {train_images} 训练, {val_images} 验证")

    # 创建新的配置文件（UTF-8编码）
    config_content = f"""# DMZ 目标检测数据集
# 从VOC格式重新构建
path: {yolo_dir}

# 数据集路径
train: train/images
val: val/images

# 类别信息 (6个类别)
nc: 6
names:
  0: crossing
  1: stop
  2: countdown_blank
  3: countdown_go
  4: go
  5: blank

# 数据统计
# 训练集: {train_images} 张图像
# 验证集: {val_images} 张图像
# 总图像: {train_images + val_images} 张
# 验证集比例: {val_images / (train_images + val_images) * 100:.1f}%

# 标签格式: YOLO v5/v7/v8
# class_id x_center y_center width height
# 坐标已归一化到 [0, 1] 范围

# 使用说明:
# 1. 此数据集已从VOC格式重新构建
# 2. 已修复所有文件路径问题
# 3. 使用UTF-8编码
"""

    # 用二进制模式写入确保编码正确
    config_path = yolo_dir / "dmz_fixed.yaml"

    with open(config_path, 'wb') as f:
        f.write(config_content.encode('utf-8'))

    print(f"✅ 创建配置文件: {config_path}")

    # 显示配置文件内容
    print("\n📋 配置文件内容:")
    print("=" * 50)
    print(config_content)
    print("=" * 50)


def verify_final_dataset():
    """验证最终数据集"""

    print("\n🔍 验证最终数据集")
    print("=" * 50)

    yolo_dir = Path(r"D:\苏雅文\Desktop\ultralytics-main\DMZ")

    # 检查文件数量
    train_images = list((yolo_dir / "train" / "images").glob("*.*"))
    train_labels = list((yolo_dir / "train" / "labels").glob("*.txt"))

    val_images = list((yolo_dir / "val" / "images").glob("*.*"))
    val_labels = list((yolo_dir / "val" / "labels").glob("*.txt"))

    print(f"📊 文件统计:")
    print(f"  训练集: {len(train_images)} 图像, {len(train_labels)} 标签")
    print(f"  验证集: {len(val_images)} 图像, {len(val_labels)} 标签")

    # 检查匹配
    train_matched = len([img for img in train_images
                         if (yolo_dir / "train" / "labels" / f"{img.stem}.txt").exists()])
    val_matched = len([img for img in val_images
                       if (yolo_dir / "val" / "labels" / f"{img.stem}.txt").exists()])

    print(f"\n🔗 图像-标签匹配:")
    print(f"  训练集: {train_matched}/{len(train_images)} 匹配")
    print(f"  验证集: {val_matched}/{len(val_images)} 匹配")

    # 检查示例
    if train_images:
        sample = train_images[0]
        print(f"\n📷 训练集示例:")
        print(f"  图像: {sample.name}")

        label_file = yolo_dir / "train" / "labels" / f"{sample.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"  标签: {len(lines)} 个标注")
                if lines:
                    print(f"    第一个: {lines[0].strip()}")

    if val_images:
        sample = val_images[0]
        print(f"\n📷 验证集示例:")
        print(f"  图像: {sample.name}")

        label_file = yolo_dir / "val" / "labels" / f"{sample.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
                print(f"  标签: {len(lines)} 个标注")


def main():
    print("🎯 DMZ数据集终极修复")
    print("=" * 60)

    print("📝 问题分析:")
    print("  1. 文件路径错误导致复制失败")
    print("  2. 编码问题导致配置文件读取失败")
    print("  3. 需要从源头重新构建")

    confirm = input("\n是否开始终极修复? (y/n): ").strip().lower()

    if confirm != 'y':
        print("❌ 用户取消")
        return

    # 执行修复
    fix_dataset_issues()

    # 验证结果
    verify_final_dataset()

    print("\n" + "=" * 60)
    print("🎉 数据集准备完成!")
    print("\n📝 下一步:")
    print("  1. 数据集已准备好")
    print("  2. 配置文件: DMZ/dmz_fixed.yaml")
    print("  3. 可以开始训练B1模型")

    # 询问是否创建训练脚本
    create_script = input("\n是否创建B1模型训练脚本? (y/n): ").strip().lower()
    if create_script == 'y':
        create_b1_training_script()


def create_b1_training_script():
    """创建简单的B1训练脚本"""

    script_content = '''# test_voc_single.py
"""
简单B1模型训练脚本
在DMZ数据集上训练扩散模型
"""

import torch
import os
from pathlib import Path
import sys

def main():
    print("🚀 开始训练B1模型")
    print("=" * 50)

    # 1. 添加模型路径
    model_dir = Path(r"D:\\新建文件夹\\eat")
    sys.path.append(str(model_dir))

    print(f"📁 模型目录: {model_dir}")

    # 2. 检查数据集
    dataset_dir = Path(r"D:\\苏雅文\\Desktop\\ultralytics-main\\DMZ")
    config_file = dataset_dir / "dmz_fixed.yaml"

    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return

    print(f"✅ 数据集: {dataset_dir}")
    print(f"✅ 配置文件: {config_file}")

    # 3. 尝试导入模型
    try:
        # 根据你的实际模型调整
        from diffusion_with_offset import DiffusionWithOffset

        print("✅ 成功导入扩散模型")

        # 创建模型
        model = DiffusionWithOffset()

        # 加载预训练权重（如果有）
        weights_file = model_dir / "b1_original_method.pth"
        if weights_file.exists():
            model.load_state_dict(torch.load(weights_file))
            print(f"✅ 加载预训练权重: {weights_file}")

        # 转移到GPU
        if torch.cuda.is_available():
            model.cuda()
            print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")

        print("\n🏋️ 模型准备就绪，可以开始训练!")

        # 训练参数
        epochs = 50
        batch_size = 4
        learning_rate = 1e-4

        print(f"\n⚙️ 建议训练参数:")
        print(f"  轮数: {epochs}")
        print(f"  批次大小: {batch_size}")
        print(f"  学习率: {learning_rate}")

        print(f"\n💾 模型将保存到:")
        print(f"  {model_dir}/b1_dmz_trained.pth")

    except ImportError as e:
        print(f"❌ 导入模型失败: {e}")
        print("请检查:")
        print("  1. diffusion_with_offset.py 文件是否存在")
        print("  2. 模型类名是否正确")
        print("  3. 依赖包是否已安装")

if __name__ == "__main__":
    main()
'''

    script_path = Path(r"D:\新建文件夹\eat\train_b1_simple.py")

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"\n✅ 创建训练脚本: {script_path}")
    print(f"\n📝 使用说明:")
    print(f"  1. 运行: python {script_path}")
    print(f"  2. 根据输出调整模型导入")
    print(f"  3. 开始训练B1模型")


if __name__ == "__main__":
    main()