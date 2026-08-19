# fix_offset_output.py
import torch


def fix_offset_output_range():
    """修正偏移网络的输出范围"""

    # 加载现有网络
    offset_net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 3, 3, padding=1),
        torch.nn.Tanh()  # 添加Tanh限制输出
    )

    try:
        offset_net.load_state_dict(torch.load('trained_offset_net_b1.pth'))
        print("加载现有网络成功")
    except:
        print("创建新网络")

    # 测试修正效果
    test_input = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        offset = offset_net(test_input)
        print(f"修正后输出范围: [{offset.min():.3f}, {offset.max():.3f}]")
        print(f"修正后输出均值: {offset.mean():.3f}")

    # 保存修正后的网络
    torch.save(offset_net.state_dict(), 'offset_net_fixed.pth')
    print("✅ 修正后的网络已保存: offset_net_fixed.pth")

    return offset_net


# 或者直接在p_sample_loop中添加范围限制
def apply_offset_with_range_limit(img, offset_net, strength=0.1):
    """应用偏移网络，但限制输出范围"""
    with torch.no_grad():
        offset = offset_net(img)

        # 限制偏移量范围
        offset = torch.tanh(offset)  # 强制到[-1, 1]

        # 可选：进一步缩放
        offset = offset * 0.5  # 缩小到[-0.5, 0.5]

        img = img + strength * offset
        img = torch.clamp(img, -1, 1)

    return img