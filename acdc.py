import matplotlib.pyplot as plt
import numpy as np

# 阶段名称
stages = ['婴儿期', '成长期', '成熟期', '衰退期']
x = np.arange(len(stages))  # [0,1,2,3]

# 各指标归一化数值（0-100之间，示意趋势）
performance = [10, 45, 85, 70]          # 性能参数：缓升→快升→趋缓→微降
patent_level = [85, 60, 30, 15]         # 专利等级：初始高，持续下降
patent_count = [5, 65, 80, 40]          # 专利数量：少→激增→峰值→下降
profit = [-60, -20, 80, 40]             # 经济收益：负→负转正→峰值→回落

plt.figure(figsize=(10, 6))

plt.plot(x, performance, 'o-', linewidth=2, markersize=8, label='性能参数')
plt.plot(x, patent_level, 's-', linewidth=2, markersize=8, label='专利等级（创新级别）')
plt.plot(x, patent_count, '^-', linewidth=2, markersize=8, label='专利数量')
plt.plot(x, profit, 'd-', linewidth=2, markersize=8, label='经济收益')

plt.xticks(x, stages, fontsize=12)
plt.xlabel('技术发展阶段', fontsize=14)
plt.ylabel('指标相对值', fontsize=14)
plt.title('导盲辅助技术S曲线（多指标进化趋势）', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best', fontsize=12)

# 标注爱奉者产品当前位置（成熟期）
plt.annotate('爱奉者双超声波盲杖\n（当前产品）', xy=(2, performance[2]), xytext=(2.2, 70),
             arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')

plt.tight_layout()
plt.savefig('s_curve_multi.png', dpi=150)
plt.show()
print("图片已保存为 s_curve_multi.png")