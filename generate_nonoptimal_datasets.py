import os
import shutil
import random
from pathlib import Path

# ==================== 配置参数（根据你的实际情况修改，Windows 路径格式） ====================
# 1. 你的原始数据集根目录（复制 Windows 路径，直接粘贴即可，用 r 前缀避免转义）
RAW_DATASET_DIR = r"D:\新建文件夹\eat\ExDark\__MACOSX\ExDark"
# 2. 整理后的 YOLO 数据集输出目录（Windows 目录，可自定义）
YOLO_DATASET_DIR = r"D:\新建文件夹\eat\ExDark\yolo_exdark"
# 3. 训练集/验证集划分比例（train:val = TRAIN_SPLIT : (1-TRAIN_SPLIT)）
TRAIN_SPLIT = 0.8
# 4. 支持的图像格式（无需修改，覆盖常见格式）
SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# ==========================================================================================

def create_yolo_directory_structure(output_dir):
    """创建 YOLO 数据集所需的目录结构（兼容 Windows）"""
    dirs = [
        os.path.join(output_dir, "images", "train"),
        os.path.join(output_dir, "images", "val"),
        os.path.join(output_dir, "labels", "train"),
        os.path.join(output_dir, "labels", "val")
    ]
    for dir_path in dirs:
        # parents=True：自动创建上级目录；exist_ok=True：目录已存在时不报错
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ 已创建目录：{dir_path}")

def get_class_names(raw_dir):
    """获取所有类别名称（排除 __MACOSX 等隐藏/无效文件夹）"""
    class_names = [
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
           and not d.startswith(".")  # 排除 Mac 隐藏文件夹
           and not d.startswith("__")  # 排除 __MACOSX 这类系统文件夹
    ]
    class_names.sort()  # 排序保证类别编号一致，便于后续复现
    print(f"✅ 已识别有效类别：{class_names}（共 {len(class_names)} 类）")
    return class_names

def split_train_val_files(image_files, train_split):
    """将图像文件随机划分为训练集和验证集（固定随机种子，结果可复现）"""
    random.seed(42)  # 固定随机种子，Windows/Linux 上划分结果一致
    random.shuffle(image_files)
    train_size = int(len(image_files) * train_split)
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]
    print(f"✅ 数据划分完成：训练集 {len(train_files)} 张，验证集 {len(val_files)} 张")
    return train_files, val_files

def copy_image_and_label(image_file, class_idx, target_split, raw_dir, yolo_dir):
    """复制图像文件和对应的标签文件到目标目录（兼容 Windows 路径，处理标签类别编号）"""
    # 1. 处理图像文件（保留原始文件名，避免重名冲突）
    image_basename = os.path.basename(image_file)
    image_target_path = os.path.join(
        yolo_dir, "images", target_split, image_basename
    )
    # copy2：保留文件元信息（创建时间、权限等），Windows/Linux 通用
    shutil.copy2(image_file, image_target_path)

    # 2. 处理标签文件（假设：图像 xxx.jpg 对应标签 xxx.txt，与图像同目录）
    label_basename = os.path.splitext(image_basename)[0] + ".txt"
    label_source_path = os.path.join(
        os.path.dirname(image_file), label_basename
    )
    label_target_path = os.path.join(
        yolo_dir, "labels", target_split, label_basename
    )

    # 3. 若标签文件存在，复制并统一类别编号（全局唯一，避免类别冲突）
    if os.path.exists(label_source_path):
        # 读取原始标签，Windows 下用 utf-8-sig 编码，兼容中文和特殊字符
        with open(label_source_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        # 写入新标签，替换为全局类别编号
        with open(label_target_path, "w", encoding="utf-8") as f:
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:  # 确保是 YOLO 格式：class x y w h
                        parts[0] = str(class_idx)  # 替换为全局类别编号
                        f.write(" ".join(parts) + "\n")
    else:
        # 仅提示真实有效图像的标签缺失，垃圾文件已被过滤，不会出现此提示
        print(f"⚠️  未找到有效图像 {image_basename} 对应的标签文件 {label_basename}，跳过标签复制")

def generate_data_yaml(yolo_dir, class_names):
    """生成 YOLO 训练所需的 data.yaml 配置文件（兼容后续 Linux 环境）"""
    # 转换路径格式：Windows \ 转为 Linux /，方便后续上传云端直接使用
    yolo_dir_abs = os.path.abspath(yolo_dir).replace("\\", "/")
    data_yaml_path = os.path.join(yolo_dir, "data.yaml")

    # 构造 yaml 内容，类别名称用引号包裹，避免特殊字符报错
    class_names_str = ", ".join([f"'{c}'" for c in class_names])
    yaml_content = f"""# YOLO 数据集配置文件（Windows 自动生成，兼容 Linux）
path: {yolo_dir_abs}  # 数据集根目录（已转换为 Linux 路径格式）
train: images/train  # 训练集图像路径（相对 path，跨平台兼容）
val: images/val      # 验证集图像路径（相对 path，跨平台兼容）
test:                # 测试集（可选，留空即可）

# 类别信息
nc: {len(class_names)}  # 类别总数
names: [{class_names_str}]  # 类别名称列表
"""
    # 写入 yaml 文件，utf-8 编码兼容云端 Linux
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"✅ 已生成 data.yaml 配置文件：{data_yaml_path}")
    print(f"📌 该配置文件已转换路径格式，可直接上传 Linux 云端使用")

def main():
    """主流程：整理数据集为 YOLO 格式（Windows 专属，兼容后续 Linux 上传）"""
    # 步骤 1：检查原始数据集目录是否存在
    if not os.path.exists(RAW_DATASET_DIR):
        print(f"❌ 原始数据集目录不存在：{RAW_DATASET_DIR}")
        print(f"📌 请检查路径是否正确，Windows 路径格式示例：r'D:\\文件夹\\子文件夹'")
        return

    # 步骤 2：创建 YOLO 标准目录结构
    create_yolo_directory_structure(YOLO_DATASET_DIR)

    # 步骤 3：获取有效类别名称和编号
    class_names = get_class_names(RAW_DATASET_DIR)
    if not class_names:
        print("❌ 未识别到任何有效类别文件夹，请检查原始数据集目录")
        return

    # 步骤 4：遍历所有类别，处理图像和标签文件
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(RAW_DATASET_DIR, class_name)
        print(f"\n===== 正在处理类别：{class_name}（全局编号：{class_idx}）=====")

        # 收集该类别下所有支持的图像文件（核心优化：过滤 ._ 开头的 Mac 垃圾文件 + 提前检查标签）
        image_files = []
        for file in os.listdir(class_dir):
            # 优化点 1：排除 ._ 开头的 Mac 垃圾文件，避免无效处理
            # 优化点 2：仅保留支持的图像格式
            if not file.startswith("._") and file.lower().endswith(SUPPORTED_IMAGE_FORMATS):
                image_file_path = os.path.join(class_dir, file)
                # 优化点 3：提前检查对应标签是否存在，只收集有标签的图像，减少无效操作
                label_file_name = os.path.splitext(file)[0] + ".txt"
                label_file_path = os.path.join(class_dir, label_file_name)
                if os.path.exists(label_file_path):
                    image_files.append(image_file_path)
                else:
                    # 仅提示真实有效图像的标签缺失，垃圾文件已被过滤，不产生冗余提示
                    print(f"⚠️  跳过无标签图像：{file}（未找到 {label_file_name}）")

        if not image_files:
            print(f"⚠️  类别 {class_name} 下无有效（带标签）图像文件，跳过该类别")
            continue

        # 划分训练集和验证集（固定种子，结果可复现）
        train_files, val_files = split_train_val_files(image_files, TRAIN_SPLIT)

        # 复制训练集文件（图像+标签）
        print(f"正在复制 {class_name} 训练集文件...")
        for image_file in train_files:
            copy_image_and_label(image_file, class_idx, "train", RAW_DATASET_DIR, YOLO_DATASET_DIR)

        # 复制验证集文件（图像+标签）
        print(f"正在复制 {class_name} 验证集文件...")
        for image_file in val_files:
            copy_image_and_label(image_file, class_idx, "val", RAW_DATASET_DIR, YOLO_DATASET_DIR)

    # 步骤 5：生成跨平台兼容的 data.yaml 配置文件
    generate_data_yaml(YOLO_DATASET_DIR, class_names)

    print("\n🎉 数据集整理完成！YOLO 格式数据集已保存至：")
    print(f"📂 {YOLO_DATASET_DIR}")
    print("\n📌 下一步操作指南：")
    print("1. 压缩该目录为 .zip 或 .tar.gz 格式（推荐 .tar.gz，Linux 解压更高效）")
    print("2. 上传压缩包到云端 Linux 服务器（如 /root/ultralytics/datasets/）")
    print("3. 云端解压后，直接使用 data.yaml 进行 YOLO 训练，无需修改路径")

if __name__ == "__main__":
    main()