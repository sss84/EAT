t datetime

# 用户出生时间和提问时间
birth_date = datetime(2004, 3, 27)  # 2004年3月27日
query_date = datetime(2026, 3, 21, 4)  # 2026年3月21日 4时

# 计算两者之间的年数，月数，天数
years_diff = query_date.year - birth_date.year
months_diff = query_date.month - birth_date.month
days_diff = query_date.day - birth_date.day

if days_diff < 0:
    months_diff -= 1
    days_diff += 30  # 大致按30天计算

if months_diff < 0:
    years_diff -= 1
    months_diff += 12

print(years_diff, months_diff, days_diff)