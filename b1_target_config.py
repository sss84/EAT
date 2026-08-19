# ExDark亮度统计配置
# 用于B1偏移网络训练
# 基于500张低光图像计算

TARGET_MEAN = 0.152910  # ExDark平均亮度
TARGET_STD = 0.086645    # ExDark亮度标准差

# 训练建议
RECOMMENDED_STRENGTH = 0.015  # 建议偏移强度
NUM_TRAINING_STEPS = 800      # 建议训练步数
