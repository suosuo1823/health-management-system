# -*- coding: utf-8 -*-
"""
app/ml/constants.py  -  NHC《成人肥胖食养指南》系数标准库

数据来源：国家卫生健康委《成人肥胖食养指南》（2024年版）
  - 附录5：成人肥胖判定标准（WS/T 428-2013）
  - 附录6：常见身体活动强度系数及基础代谢率公式
  - 表1：中国居民成人膳食能量需要量
"""

from __future__ import annotations

# =====================================================================
# 附录5 - BMI 分类标准（WS/T 428-2013）
# =====================================================================
BMI_THRESHOLDS: dict[str, float] = {
    "obese":     28.0,   # 肥胖
    "overweight": 24.0,   # 超重
    "normal":    18.5,   # 正常
}

# 附录5 - 腰围中心型肥胖标准（WS/T 428-2013）
WAIST_CENTRAL: dict[str, dict[str, float]] = {
    "male":   {"pre": 85, "obese": 90},   # cm
    "female": {"pre": 80, "obese": 85},
}

# =====================================================================
# 附录6 - 代谢当量（MET）系数表
# =====================================================================
MET_TABLE: dict[str, float] = {
    # 静态 / 低强度
    "lying":         1.2,   # 安静躺着
    "sitting":       1.3,   # 安静坐着
    "standing":      1.6,   # 安静站着
    "typing":        1.7,   # 坐姿打字
    "light":         2.5,   # 家务等轻活动（综合）
    "walking_slow":  2.9,   # 步行 3km/h
    "walking_med":   3.8,   # 步行 5km/h
    "walking_fast":  5.2,   # 步行 6km/h
    "cycling":       4.0,   # 骑行 12km/h
    "housework":     2.8,   # 家务清洁
    "aerobic_med":   4.8,   # 跑步 5km/h
    "running":       6.5,   # 跑步 6km/h
    "running_fast":  8.2,   # 跑步 8km/h
    "climbing":      6.1,   # 登山慢速
    "swimming":      6.0,   # 游泳（估计）
}

# MET 强度分类
MET_INTENSITY: dict[str, tuple[float, float]] = {
    "static":  (0,   1.5),   # 静态行为
    "low":     (1.6, 2.9),   # 低强度
    "medium":  (3.0, 5.9),   # 中等强度
    "high":    (6.0, 999),   # 高强度
}


# =====================================================================
# 附录6 - 基础代谢率公式（Harris-Benedict）
# =====================================================================
def calc_bmr(weight_kg: float, height_cm: float, age_yr: float, gender: int) -> float:
    """
    根据体重(kg)、身高(cm)、年龄(岁)、性别计算基础代谢率(kcal/day)。
    来源：NHC指南附录6 - Harris-Benedict 公式
    女性: BMR = 655 + 9.5W + 1.8H - 4.7A
    男性: BMR = 66  + 13.7W + 5.0H - 6.8A
    """
    w = float(weight_kg)
    h = float(height_cm)
    a = float(age_yr)
    if gender == 1:
        return 66  + 13.7 * w + 5.0 * h - 6.8 * a
    else:
        return 655 +  9.5 * w + 1.8 * h - 4.7 * a


# =====================================================================
# 表1 - 中国居民膳食能量需要量（kcal/d）
# =====================================================================
DIETARY_ENERGY: dict[str, dict[str, tuple[int, int]]] = {
    "low":    {"male": (1950, 2150), "female": (1600, 1700)},
    "medium": {"male": (2400, 2550), "female": (1950, 2100)},
    "high":   {"male": (2800, 3000), "female": (2300, 2450)},
}

# FAF 值到身体活动水平的映射（用于计算 TDEE）
# FAF: 0=从不, 1=1-2次/周, 2=3-4次/周, 3=每天
FAF_TO_PAL: dict[int, float] = {
    0: 1.4,   # 几乎不活动 → 久坐
    1: 1.55,  # 偶尔 → 轻微活动
    2: 1.75,  # 经常 → 中等活动
    3: 1.9,   # 每天 → 活跃
}

# =====================================================================
# 指南推荐的运动量标准
# =====================================================================
EXERCISE_GUIDELINE: dict[str, int | str] = {
    "aerobic_min":       150,    # 每周最少中等强度有氧（分钟）
    "aerobic_max":       300,    # 每周最多（分钟）
    "aerobic_days":       "5-7",  # 每周天数
    "resistance_days":    "2-3",  # 抗阻运动每周天数
    "resistance_min":     10,    # 每次最少（分钟）
    "resistance_max":     20,    # 每次最多（分钟）
    "weekly_kcal_target": 2000,   # 每周通过运动消耗的目标能量（kcal）
}

# =====================================================================
# 宏量营养素供能比（NHC 指南建议范围）
# =====================================================================
MACRO_RATIOS: dict[str, tuple[float, float]] = {
    "fat":    (0.20, 0.30),   # 脂肪: 20%-30%
    "protein":(0.15, 0.20),    # 蛋白质: 15%-20%
    "carbs":  (0.50, 0.60),    # 碳水: 50%-60%
}

# 每日营养素摄入上限（NHC 指南）
DAILY_LIMITS: dict[str, int] = {
    "salt":  5,   # 盐 ≤5g/天
    "oil":   25,  # 烹调油 ≤25g/天
    "sugar": 25,  # 添加糖 <25g/天
}

# 每日饮水量建议（NHC 指南）
WATER_INTAKE: dict[str, int] = {
    "min_cups": 7,
    "max_cups": 8,
    "min_ml":   1500,
    "max_ml":   2000,
}

# =====================================================================
# 减重速度标准（NHC 指南）
# =====================================================================
WEIGHT_LOSS: dict[str, float | tuple[float, float]] = {
    "pct_6month":            (0.05, 0.10),  # 6个月内减少体重的5%-10%
    "kg_per_month":           (2, 4),         # 每月减2-4kg
    "kg_per_week_start":      0.5,            # 起始目标：每周0.5kg
    "kcal_deficit_per_kg":    7700,           # 减1kg脂肪 ≈ 消耗7700kcal
}

# 能量缺口建议（NHC 指南）
ENERGY_DEFICIT: dict[str, tuple | tuple[int, int]] = {
    "pct_range":    (0.30, 0.50),    # 每日缺口占维持量的30%-50%
    "kcal_range":    (500, 1000),     # 或每日减少500-1000kcal
    "male_limit":    (1200, 1500),    # 男性下限（kcal）
    "female_limit":  (1000, 1200),   # 女性下限（kcal）
}


# =====================================================================
# 风险等级配置
# =====================================================================
LEVEL_CONFIG: dict[str, dict[str, str]] = {
    "Insufficient_Weight":  {"code": "insufficient", "color": "#17a2b8", "level_cn": "体重偏低"},
    "Normal_Weight":       {"code": "normal",        "color": "#28a745", "level_cn": "体重正常"},
    "Overweight_Level_I":   {"code": "overweight1",   "color": "#f0a500", "level_cn": "轻度超重"},
    "Overweight_Level_II": {"code": "overweight2",   "color": "#fd7e14", "level_cn": "中度超重"},
    "Overweight_Level_III":{"code": "overweight3",   "color": "#dc3545", "level_cn": "重度超重"},
    "Obesity_Type_I":      {"code": "obese1",        "color": "#c82333", "level_cn": "一级肥胖"},
    "Obesity_Type_II":     {"code": "obese2",        "color": "#a71d2a", "level_cn": "二级肥胖"},
    "Obesity_Type_III":    {"code": "obese3",        "color": "#7b0026", "level_cn": "三级肥胖"},
}

# =====================================================================
# 运动类型中文映射（统一入口，避免多处重复定义）
# =====================================================================
EXERCISE_TYPE_NAMES: dict[str, str] = {
    "running":       "跑步",
    "walking":       "步行",
    "cycling":       "骑行",
    "swimming":      "游泳",
    "jumping_rope":  "跳绳",
    "yoga":          "瑜伽",
    "gym":           "健身房训练",
    "basketball":    "篮球",
    "football":      "足球",
    "badminton":     "羽毛球",
    "dancing":       "跳舞",
    "hiking":        "徒步",
    "climbing":      "爬山",
    "other":         "其他",
}

# =====================================================================
# 餐次中文标签（导出 CSV 时使用）
# =====================================================================
MEAL_TYPE_NAMES: dict[str, str] = {
    "breakfast": "早餐",
    "lunch":     "午餐",
    "dinner":    "晚餐",
    "snack":     "零食",
}

# recipe.py 使用的别名（保持向后兼容）
MEAL_LABELS = MEAL_TYPE_NAMES
