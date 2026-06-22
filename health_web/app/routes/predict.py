# -*- coding: utf-8 -*-
"""
app/routes/predict.py  -  肥胖风险预测蓝图
（系数标准库已移至 app/ml/constants.py）
"""

import json
from datetime import datetime, date
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from app import db
from app.models.predict import PredictRecord
from app.ml.predictor import predictor
from app.ml.constants import (
    BMI_THRESHOLDS, WAIST_CENTRAL, MET_TABLE, MET_INTENSITY,
    calc_bmr, DIETARY_ENERGY, FAF_TO_PAL, EXERCISE_GUIDELINE,
    MACRO_RATIOS, DAILY_LIMITS, WATER_INTAKE, WEIGHT_LOSS,
    ENERGY_DEFICIT, LEVEL_CONFIG,
)
from app.utils import safe_float, safe_int, parse_date, db_transaction

predict_bp = Blueprint("predict", __name__, url_prefix="/predict")


# ─────────────────────────────────────────────────────────────────
# 量化建议生成引擎
# 所有系数均来自 app/ml/constants.py（NHC《成人肥胖食养指南》2024年版）
# ─────────────────────────────────────────────────────────────────

def _calc_bmi(weight, height):
    h = float(height) / 100.0
    if h <= 0:
        return 0.0
    return round(float(weight) / (h * h), 1)


def _classify_bmi(bmi: float) -> str:
    """根据NHC指南附录5标准分类BMI"""
    if bmi < BMI_THRESHOLDS["normal"]:
        return "体重偏低"
    elif bmi < BMI_THRESHOLDS["overweight"]:
        return "体重正常"
    elif bmi < BMI_THRESHOLDS["obese"]:
        return "超重"
    else:
        return "肥胖"


def _classify_waist(waist_cm: float, gender: int) -> str:
    """根据NHC指南附录5腰围标准分类中心型肥胖"""
    thresholds = WAIST_CENTRAL["male" if gender == 1 else "female"]
    if waist_cm >= thresholds["obese"]:
        return "中心型肥胖"
    elif waist_cm >= thresholds["pre"]:
        return "中心型肥胖前期"
    return "正常"


def _calc_tdee(bmr: float, faf: int) -> float:
    """根据FAF值估算每日总能量消耗（考虑活动系数）"""
    pal = FAF_TO_PAL.get(int(faf), 1.4)
    return round(bmr * pal, 0)


def _calc_weight_loss_plan(current_weight: float, height_cm: float,
                            bmr: float, tdee: float,
                            gender: int) -> dict:
    """
    根据NHC指南计算个性化减重计划。
    返回：目标体重、每日目标热量、热量缺口、减重速度建议
    """
    ideal_weight = round(22 * (height_cm / 100) ** 2, 1)
    excess_weight = round(current_weight - ideal_weight, 1)

    if excess_weight <= 0:
        return {
            "need_loss": False,
            "ideal_weight": ideal_weight,
            "excess_weight": 0,
            "maintain_cal": round(tdee),
            "deficit_cal": 0,
            "target_cal": round(tdee),
            "monthly_kg": 0,
            "6month_pct": 0,
        }

    # 维持量
    maintain_cal = round(tdee)

    # 指南建议：每日缺口500-1000kcal，或30%-50%
    min_deficit = min(ENERGY_DEFICIT["kcal_range"][0],
                      round(maintain_cal * ENERGY_DEFICIT["pct_range"][0]))
    max_deficit = min(ENERGY_DEFICIT["kcal_range"][1],
                      round(maintain_cal * ENERGY_DEFICIT["pct_range"][1]))

    # 性别下限约束
    if gender == 1:  # 男性
        min_target = max(ENERGY_DEFICIT["male_limit"][0],
                         round(maintain_cal - max_deficit))
        target_cal = max(ENERGY_DEFICIT["male_limit"][0],
                         round(maintain_cal - min_deficit))
    else:            # 女性
        min_target = max(ENERGY_DEFICIT["female_limit"][0],
                         round(maintain_cal - max_deficit))
        target_cal = max(ENERGY_DEFICIT["female_limit"][0],
                         round(maintain_cal - min_deficit))

    actual_deficit = maintain_cal - target_cal

    # 每月预计减重（按7700kcal/kg）
    monthly_kg = round(actual_deficit * 30 / WEIGHT_LOSS["kcal_deficit_per_kg"], 1)

    # 6个月目标（不超过初始体重的10%）
    max_6month_loss = round(current_weight * WEIGHT_LOSS["pct_6month"][1], 1)
    realistic_6month = min(max_6month_loss, excess_weight)

    return {
        "need_loss": True,
        "ideal_weight": ideal_weight,
        "excess_weight": excess_weight,
        "maintain_cal": maintain_cal,
        "deficit_cal": round(actual_deficit),
        "target_cal": round(target_cal),
        "monthly_kg": monthly_kg,
        "min_target_cal": round(min_target),
        "6month_kg": round(realistic_6month, 1),
        "6month_pct": round(realistic_6month / current_weight * 100, 1),
    }


def _calc_exercise_plan(faf: int) -> dict:
    """
    根据NHC指南附录6的MET系数和运动推荐计算运动计划。
    FAF: 0=从不, 1=1-2次/周, 2=3-4次/周, 3=每天
    返回：当前运动量评估、周目标、预计每周消耗
    """
    # 当前每周有氧运动分钟数估算（假设每次30-60分钟）
    faf_minutes = {
        0: 0,
        1: 45,   # 1-2次 × 平均45分钟
        2: 105,  # 3-4次 × 平均35分钟（保守）
        3: 210,  # 每天 × 30分钟
    }
    current_min = faf_minutes.get(int(faf), 0)

    # 指南推荐：每周150-300分钟中等强度有氧
    target_min_min = EXERCISE_GUIDELINE["aerobic_min"]
    target_min_max = EXERCISE_GUIDELINE["aerobic_max"]

    need_more = current_min < target_min_min
    gap = max(0, target_min_min - current_min)

    # 估算每周通过有氧消耗的热量（按体重70kg，中等强度4.8MET）
    # 消耗 = MET × 体重(kg) × 时间(h)
    # 假设体重70kg, MET=4.8, 1小时 = 4.8 × 70 = 336 kcal
    kcal_per_min_estimate = 5.6  # ≈ 336/60 kcal/分钟

    current_weekly_kcal = round(current_min * kcal_per_min_estimate)
    target_weekly_kcal = EXERCISE_GUIDELINE["weekly_kcal_target"]
    kcal_gap = max(0, target_weekly_kcal - current_weekly_kcal)

    # 补充缺口需要的额外分钟数（中等强度）
    extra_min_needed = round(kcal_gap / kcal_per_min_estimate) if kcal_gap > 0 else 0

    return {
        "current_min_per_week": current_min,
        "target_min_min": target_min_min,
        "target_min_max": target_min_max,
        "need_more": need_more,
        "gap_min": gap,
        "current_weekly_kcal": current_weekly_kcal,
        "target_weekly_kcal": target_weekly_kcal,
        "kcal_gap": kcal_gap,
        "extra_min_needed": extra_min_needed,
        "resistance_days": EXERCISE_GUIDELINE["resistance_days"],
        "resistance_min": EXERCISE_GUIDELINE["resistance_min"],
        "resistance_max": EXERCISE_GUIDELINE["resistance_max"],
    }


def _generate_personalized_advice(features: dict, bmi: float,
                                   label: str, proba_dict: dict) -> dict:
    """
    基于NHC《成人肥胖食养指南》2024年版量化建议引擎。
    所有系数均有指南原文依据。
    """
    suggestions = []
    gender = int(features.get("Gender", 1))
    age = float(features.get("Age", features.get("age", 30)))
    weight = float(features.get("Weight", features.get("weight", 70)))
    height = float(features.get("Height", features.get("height", 170)))
    faf = int(features.get("FAF", features.get("physical_activity", 1)))

    # ── 预计算基础数据 ──────────────────────────────────────────
    bmr = calc_bmr(weight, height, age, gender)
    tdee = _calc_tdee(bmr, faf)
    weight_plan = _calc_weight_loss_plan(weight, height, bmr, tdee, gender)
    exercise_plan = _calc_exercise_plan(faf)

    # ── 1. BMI专项（impact最高，根据指南精确阈值）────────────
    if bmi < BMI_THRESHOLDS["normal"]:
        deficit = round(BMI_THRESHOLDS["normal"] - bmi, 1)
        target_w = round(BMI_THRESHOLDS["normal"] * (height / 100) ** 2, 1)
        suggestions.append({
            "icon": "fa-arrow-up", "color": "#17a2b8",
            "label": "体重偏低（BMI不足）",
            "current": f"BMI={bmi:.1f}，低于正常下限",
            "target": f"BMI≥{BMI_THRESHOLDS['normal']}（目标体重≥{target_w}kg）",
            "description": (
                "适当增加营养摄入，优先补充优质蛋白（每天kg体重×1.2-1.5g蛋白质），"
                "配合力量训练增加肌肉量。建议增加坚果、全脂奶制品和复合碳水摄入。"
            ),
            "impact": 8,
        })
    elif bmi >= BMI_THRESHOLDS["obese"]:
        wp = weight_plan
        suggestions.append({
            "icon": "fa-arrow-down", "color": "#dc3545",
            "label": "肥胖（BMI≥28.0）",
            "current": f"BMI={bmi:.1f}，超标{wp['excess_weight']}kg（理想体重{wp['ideal_weight']}kg）",
            "target": (
                f"每日目标热量{wp['target_cal']}kcal，"
                f"每月减{wp['monthly_kg']}kg，"
                f"6个月目标减{wp['6month_kg']}kg"
            ),
            "description": (
                f"建议每日摄入量控制在{wp['target_cal']}kcal（基于Harris-Benedict公式，"
                f"基础代谢率{round(bmr)}kcal/d），相比维持量减少约{wp['deficit_cal']}kcal。"
                f"遵循指南减重目标：6个月内减少5%-10%体重，即每月减2-4kg。"
            ),
            "impact": 10,
        })
    elif bmi >= BMI_THRESHOLDS["overweight"]:
        wp = weight_plan
        suggestions.append({
            "icon": "fa-minus", "color": "#f0a500",
            "label": "超重（24.0≤BMI<28.0）",
            "current": f"BMI={bmi:.1f}，超出理想{wp['excess_weight']}kg",
            "target": f"每日{wp['target_cal']}kcal，每月减{wp['monthly_kg']}kg",
            "description": (
                f"根据基础代谢率（{round(bmr)}kcal/d）和活动水平，"
                f"维持热量约{wp['maintain_cal']}kcal/d。建议每日减少{wp['deficit_cal']}kcal，"
                f"降至{wp['target_cal']}kcal/d。减掉{wp['excess_weight']}kg可回到BMI正常范围。"
            ),
            "impact": 7,
        })
    else:
        suggestions.append({
            "icon": "fa-check-circle", "color": "#28a745",
            "label": "BMI健康（18.5-24.0）",
            "current": f"BMI={bmi:.1f}，处于正常范围",
            "target": "维持现状即可",
            "description": "您的体重在健康范围内。建议维持均衡饮食和规律运动，"
                          "关注体脂率和腰围变化，防范中心型肥胖。",
            "impact": 0,
        })

    # ── 2. 高热量食物（>400kcal/100g，指南第162页定义）────────
    cal_val = int(features.get("high_calorie_food", 1))
    if cal_val >= 2:
        freq_labels = {2: "经常（≥3次/周）", 3: "总是吃"}
        suggestions.append({
            "icon": "fa-fire", "color": "#dc3545",
            "label": "高能量食物摄入",
            "current": freq_labels.get(cal_val, f"频率偏高({cal_val})"),
            "target": "偶尔（≤1次/周）",
            "description": (
                "高能量食物通常指每100g提供≥400kcal的食物（油炸食品、含糖烘焙糕点、肥肉等）。"
                "建议减少此类食物，用低能量密度食物替代（蔬菜、水果、全谷物）。"
            ),
            "impact": cal_val * 2,
        })

    # ── 3. 蔬菜摄入（指南：每日300-500g，深色蔬菜≥1/2）─────────
    veg_val = int(features.get("FCVC", 2))
    veg_labels = {0: "很少吃蔬菜", 1: "偶尔（每餐不足）", 2: "经常但量不足", 3: "每餐蔬菜占一半以上"}
    if veg_val < 3:
        suggestions.append({
            "icon": "fa-leaf", "color": "#28a745",
            "label": "蔬菜水果摄入",
            "current": veg_labels.get(veg_val, "不足"),
            "target": "每日300-500g蔬菜，深色蔬菜占1/2以上",
            "description": (
                "减重期间应增加蔬菜摄入量至每日300-500g（生重），"
                "其中深色蔬菜应占一半以上。深色蔬菜富含β-胡萝卜素和膳食纤维，"
                "有助于增强饱腹感、延缓血糖上升。"
            ),
            "impact": (3 - veg_val) * 2,
        })

    # ── 4. 身体活动（指南附录6 MET系数 + 运动推荐）─────────────
    ep = exercise_plan
    if ep["need_more"]:
        suggestions.append({
            "icon": "fa-bicycle", "color": "#f0a500",
            "label": "身体活动（不足）",
            "current": (
                f"每周约{ep['current_min_per_week']}分钟，"
                f"约消耗{ep['current_weekly_kcal']}kcal"
            ),
            "target": (
                f"每周{ep['target_min_min']}-{ep['target_min_max']}分钟中等强度有氧，"
                f"消耗≥{ep['target_weekly_kcal']}kcal；"
                f"抗阻运动{ep['resistance_days']}天，每次{ep['resistance_min']}-{ep['resistance_max']}分钟"
            ),
            "description": (
                f"指南推荐每周150-300分钟中等强度有氧运动（MEP≥3.0，如快走4-5km/h、"
                f"慢跑5km/h等），每周消耗≥2000kcal。当前缺口约{ep['gap_min']}分钟/周，"
                f"约需额外消耗{ep['kcal_gap']}kcal，可通过每天快走30分钟逐步弥补。"
            ),
            "impact": 9,
        })
    elif faf >= 2:
        suggestions.append({
            "icon": "fa-heartbeat", "color": "#28a745",
            "label": "身体活动（达标）",
            "current": f"每周约{ep['current_min_per_week']}分钟，达标",
            "target": "继续保持，逐步增加",
            "description": (
                f"您的运动量已达到指南推荐的最低标准（150分钟/周）。"
                f"建议在此基础上逐步增加至200-300分钟，并坚持{ep['resistance_days']}天/周的抗阻运动。"
            ),
            "impact": 0,
        })

    # ── 5. 屏幕时间（指南：每日≤2-4小时）────────────────────────
    screen_val = int(features.get("TUE", 1))
    screen_map = {0: "几乎不看", 1: "1-2小时/天", 2: "3-4小时/天", 3: "≥4小时/天"}
    if screen_val >= 2:
        suggestions.append({
            "icon": "fa-desktop", "color": "#6c757d",
            "label": "屏幕时间过长",
            "current": screen_map.get(screen_val, f"约{screen_val}小时/天"),
            "target": "每日≤2小时，久坐者每小时起身活动3-5分钟",
            "description": (
                "长时间静坐和被动视屏是肥胖发生的独立危险因素。"
                "建议每天屏幕时间控制在2小时以内；长期伏案工作者每小时起身活动3-5分钟，"
                "做伸展运动或快走以打破久坐习惯。"
            ),
            "impact": screen_val,
        })

    # ── 6. 零食习惯（指南：避免随意进食零食和夜宵）──────────────
    snack_val = int(features.get("CAEC_val", 1))
    if snack_val >= 2:
        snack_labels = {2: "经常吃零食", 3: "总是吃零食/夜宵"}
        suggestions.append({
            "icon": "fa-cookie", "color": "#fd7e14",
            "label": "零食与夜宵",
            "current": snack_labels.get(snack_val, f"频率较高({snack_val})"),
            "target": "以低能量密度食物替代（黄瓜、番茄、低糖水果）",
            "description": (
                "指南建议：严格控制零食摄入，尤其避免晚间随意进食和夜宵。"
                "如感饥饿可选择低能量高纤维食物（如小番茄、黄瓜、一小把坚果）。"
                "每克酒精提供约7kcal热量，建议减重期间避免饮酒。"
            ),
            "impact": snack_val,
        })

    # ── 7. 饮水量（指南：每日充足饮水）─────────────────────────
    water_val = int(features.get("CH2O", 2))
    water_map = {0: "少于1杯", 1: "1-2杯(约250-500ml)", 2: "2-3杯(约500-750ml)", 3: "≥3杯"}
    if water_val < 3:
        target_cups = f"{WATER_INTAKE['min_cups']}-{WATER_INTAKE['max_cups']}杯"
        suggestions.append({
            "icon": "fa-tint", "color": "#0dcaf0",
            "label": "每日饮水量",
            "current": water_map.get(water_val, "偏少"),
            "target": f"{target_cups}，约{WATER_INTAKE['max_ml']}ml",
            "description": (
                "指南建议每日饮水量充足，具体因人而异。"
                f"建议达到{target_cups}（约{WATER_INTAKE['max_ml']}ml），"
                "优先选择白开水或淡茶，避免含糖饮料。"
                "饭前半小时喝一杯水可增加饱腹感，有助于控制总能量摄入。"
            ),
            "impact": max(0, 3 - water_val),
        })

    # ── 8. 交通方式（增加日常活动消耗）─────────────────────────
    trans_val = int(features.get("MTRANS_val", 2))
    if trans_val >= 2:  # 公共交通/私家车为主
        suggestions.append({
            "icon": "fa-car", "color": "#6c757d",
            "label": "交通方式（活动量低）",
            "current": "以私家车/公共交通为主",
            "target": "每天增加步行3000-5000步（约30分钟）",
            "description": (
                "减少机动车出行，增加步行和骑行时间。"
                "以步速5km/h行走30分钟约消耗150kcal。"
                "日常生活中多走楼梯、提前一站下车，都是增加能量消耗的有效方式。"
            ),
            "impact": max(1, trans_val - 1),
        })

    # ── 9. 家族肥胖史（遗传风险提示）───────────────────────────
    if features.get("family_history") == 1:
        suggestions.append({
            "icon": "fa-users", "color": "#dc3545",
            "label": "家族肥胖史（遗传风险）",
            "current": "存在家族肥胖史",
            "target": "提前干预，坚持定期监测",
            "description": (
                "有家族史的人群代谢功能和体重调节可能存在遗传易感性，"
                "发生肥胖和相关代谢性疾病的风险更高。"
                "指南建议有家族史者更应坚持健康饮食和规律运动，定期监测体重、腰围和血压。"
            ),
            "impact": 6,
        })

    # ── 10. 主餐规律（指南：定时定量，晚餐19:00前）────────────
    meals_val = int(features.get("NCP", 2))
    if meals_val <= 1:
        suggestions.append({
            "icon": "fa-cutlery", "color": "#f0a500",
            "label": "主餐次数不规律",
            "current": "每天≤2餐",
            "target": "规律三餐，早中晚供能比3:4:3",
            "description": (
                "指南建议一日三餐时间相对固定，定时定量。"
                "推荐早中晚三餐供能比为3:4:3，晚餐建议在17:00-19:00完成，"
                "晚餐后不再进食（饮水除外）。避免跳过正餐导致过度饥饿后的暴饮暴食。"
            ),
            "impact": 3,
        })

    # ── 11. 卡路里监控（指南：自我监测是减重成功的关键）───────
    if features.get("SCC") == 0:
        suggestions.append({
            "icon": "fa-area-chart", "color": "#28a745",
            "label": "能量摄入监控",
            "current": "未进行日常热量记录",
            "target": "开始记录每日食物摄入量",
            "description": (
                "指南强调自我监测（包括食物摄入量和身体活动情况）对减重成功至关重要。"
                f"建议女性每日摄入控制在{ENERGY_DEFICIT['female_limit'][0]}-{ENERGY_DEFICIT['female_limit'][1]}kcal，"
                f"男性控制在{ENERGY_DEFICIT['male_limit'][0]}-{ENERGY_DEFICIT['male_limit'][1]}kcal，"
                "同时关注体脂率和肌肉量的变化，而非仅盯着体重数字。"
            ),
            "impact": 4,
        })

    # ── 12. 三大营养素比例建议（新增，指南核心内容）────────────
    if bmi >= BMI_THRESHOLDS["overweight"]:
        suggestions.append({
            "icon": "fa-balance-scale", "color": "#6c757d",
            "label": "三大营养素供能比例",
            "current": "比例未知（需记录）",
            "target": "蛋白质15-20% / 脂肪20-30% / 碳水50-60%",
            "description": (
                "指南建议三大宏量营养素供能比为：蛋白质15%-20%、脂肪20%-30%、"
                "碳水化合物50%-60%。"
                "优先选择全谷物（占主食一半以上）、低脂肪动物蛋白（瘦肉、去皮鸡胸、鱼虾）"
                "和脱脂奶类。每克产能：碳水4kcal、蛋白质4kcal、脂肪9kcal。"
            ),
            "impact": 3,
        })

    # ── 13. 控盐控油控糖建议（新增，指南明确限值）─────────────
    suggestions.append({
        "icon": "fa-exclamation-triangle", "color": "#ffc107",
        "label": "限盐控油控糖",
        "current": "需关注（具体用量未知）",
        "target": f"盐≤{DAILY_LIMITS['salt']}g、油≤{DAILY_LIMITS['oil']}g、添加糖<{DAILY_LIMITS['sugar']}g/天",
        "description": (
            f"指南明确建议：每日食盐不超过{DAILY_LIMITS['salt']}g，"
            f"烹调油不超过{DAILY_LIMITS['oil']}g，"
            f"添加糖最好控制在{DAILY_LIMITS['sugar']}g以下。"
            "优先选择蒸、煮、熘、水滑等烹调方式，减少油煎油炸。"
        ),
        "impact": 2,
    })

    # ── 按影响度排序 ───────────────────────────────────────────
    suggestions.sort(key=lambda x: x["impact"], reverse=True)

    # ── 生成综合建议段落 ───────────────────────────────────────
    cfg = LEVEL_CONFIG.get(label, {})
    level_cn = cfg.get("level_cn", "未知")
    code = cfg.get("code", "normal")

    top_issues = [s for s in suggestions if s["impact"] > 0][:3]
    if top_issues:
        issue_names = "、".join(s["label"] for s in top_issues)

        # 根据BMI水平生成个性化综述
        if bmi < BMI_THRESHOLDS["normal"]:
            summary = (
                f"根据您的健康数据，当前BMI={bmi:.1f}，处于【{level_cn}】范围。"
                f"建议重点关注：{issue_names}。"
                "适当增加优质蛋白摄入，配合力量训练，逐步提升体重至正常范围。"
            )
        elif bmi >= BMI_THRESHOLDS["obese"]:
            wp = weight_plan
            summary = (
                f"您的BMI={bmi:.1f}，属于【{level_cn}】。"
                f"建议重点关注：{issue_names}。"
                f"遵循指南建议：将每日热量控制在{wp['target_cal']}kcal，"
                f"每月减{wp['monthly_kg']}kg，"
                f"6个月内减少当前体重的5%-10%（约{wp['6month_kg']}kg）。"
                f"三大营养素比例建议：蛋白质15%-20%、脂肪20%-30%、碳水50%-60%。"
                f"每日盐≤5g、油≤25g、添加糖<25g。"
            )
        elif bmi >= BMI_THRESHOLDS["overweight"]:
            wp = weight_plan
            summary = (
                f"您的BMI={bmi:.1f}，处于【{level_cn}】范围。"
                f"建议重点关注：{issue_names}。"
                f"根据您的基础代谢率（Harris-Benedict公式：{round(bmr)}kcal/d），"
                f"维持量约{wp['maintain_cal']}kcal/d，建议降至{wp['target_cal']}kcal/d，"
                f"每月健康减重{wp['monthly_kg']}kg。减掉{wp['excess_weight']}kg即可回到正常范围。"
            )
        else:
            summary = (
                f"您的BMI={bmi:.1f}，处于【{level_cn}】范围。"
                f"继续保持！建议关注：{issue_names}，并定期监测腰围（防范中心型肥胖）。"
            )
    else:
        summary = f"您的各项指标良好，处于【{level_cn}】范围，请继续保持健康生活方式。"

    return {
        "advice_text": summary,
        "suggestions": suggestions,
        # 附加数据（供模板使用）
        "calc_data": {
            "bmr": round(bmr),
            "tdee": round(tdee),
            "weight_plan": weight_plan,
            "exercise_plan": exercise_plan,
        },
    }


def _level_description(label):
    desc_map = {
        "Insufficient_Weight": "体重不足",
        "Normal_Weight":       "体重正常",
        "Overweight_Level_I":  "超重一级",
        "Overweight_Level_II": "超重二级",
        "Overweight_Level_III":"超重三级",
        "Obesity_Type_I":      "一级肥胖",
        "Obesity_Type_II":     "二级肥胖",
        "Obesity_Type_III":    "三级肥胖",
    }
    return desc_map.get(label, label)


@predict_bp.route("/")
def index():
    history = []
    result  = None
    prefilled = {}

    if current_user.is_authenticated:
        prefilled = {
            "gender":    1 if current_user.gender == "male" else 0,
            "age":       current_user.age or "",
            "height":    current_user.height or "",
            "weight":    current_user.weight or "",
        }
        history = PredictRecord.query.filter_by(user_id=current_user.id)\
            .order_by(PredictRecord.created_at.desc()).limit(10).all()
        for h in history:
            h.level_color = LEVEL_CONFIG.get(h.predicted_level, {}).get("color", "#6b8a77")

    return render_template("predict/index.html",
        result=result,
        history=history,
        prefilled=prefilled,
    )


@predict_bp.route("/", methods=["POST"])
def run():
    gender      = safe_int(request.form.get("gender"), 0)
    age         = safe_float(request.form.get("age"), 25)
    height      = safe_float(request.form.get("height"), 170)
    weight      = safe_float(request.form.get("weight"), 65)

    family_history    = 1 if request.form.get("family_history") == "1" else 0
    high_calorie_food = safe_int(request.form.get("high_calorie_food"), 1)
    vegetable_freq    = safe_int(request.form.get("vegetable_frequency"), 2)
    main_meals        = safe_int(request.form.get("main_meals"), 2)
    snacking          = safe_int(request.form.get("snacking"), 1)
    water_consumption = safe_int(request.form.get("water_consumption"), 2)
    calorie_monitoring= safe_int(request.form.get("calorie_monitoring"), 0)
    physical_activity = safe_int(request.form.get("physical_activity"), 1)
    screen_time       = safe_int(request.form.get("screen_time"), 1)
    alcohol           = safe_int(request.form.get("alcohol"), 0)
    transportation    = safe_int(request.form.get("transportation"), 2)

    # 构造模型输入（13个核心特征，与 obesity_analysis.py 训练时一致）
    # 注意：MTRANS 已从特征体系中移除，不再参与模型预测
    features = {
        "Gender":                       gender,
        "Age":                          age,
        "Height":                        height,
        "Weight":                        weight,
        "family_history_with_overweight": family_history,
        "high_calorie_food":            high_calorie_food,
        "FCVC":                         vegetable_freq,
        "NCP":                          main_meals,
        "CH2O":                         water_consumption,
        "FAF":                          physical_activity,
        "TUE":                          screen_time,
        "SCC":                          calorie_monitoring,
        "CAEC":                         ["no", "Sometimes", "Frequently", "Always"][min(snacking, 3)],
        "CALC":                         ["no", "Sometimes", "Frequently", "Always"][min(alcohol, 3)],
        "SMOKE":                        0,
    }

    try:
        pred_result = predictor.predict(features)
    except Exception as e:
        flash(f"预测失败: {str(e)}", "danger")
        return redirect(url_for("predict.index"))

    label      = pred_result.get("label", "Normal_Weight")
    proba_dict = pred_result.get("probabilities", {})
    bmi        = _calc_bmi(weight, height)
    # 置信度优先使用原始XGBoost模型概率(融合概率因alpha稀释会偏低，不反映真实置信度)
    hybrid_info = pred_result.get("innovation_hybrid", {})
    ml_proba_raw = hybrid_info.get("ml_proba", proba_dict)
    confidence = max(ml_proba_raw.values()) if ml_proba_raw else (max(proba_dict.values()) if proba_dict else 0.0)

    cfg = LEVEL_CONFIG.get(label, LEVEL_CONFIG["Normal_Weight"])

    # 概率排序
    probabilities = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)

    # ── 个性化量化建议（基于NHC指南系数动态生成）───
    features["CAEC_val"]   = snacking
    features["MTRANS_val"]  = transportation
    personalized = _generate_personalized_advice(features, bmi, label, proba_dict)

    # 创新点信息
    innovation_hybrid = pred_result.get("innovation_hybrid", {})
    innovation_shap = pred_result.get("innovation_shap", {})
    innovation_stacking = pred_result.get("innovation_stacking", {})

    # SHAP归因风险因素摘要
    shap_summary = ""
    shap_factors = innovation_shap.get("top_factors", [])
    if shap_factors:
        factor_texts = []
        for f in shap_factors:
            direction = "增加" if f["shap_value"] > 0 else "降低"
            factor_texts.append(f'{f["feature_cn"]}({direction}风险{abs(f["shap_value"]):.2f})')
        shap_summary = "、".join(factor_texts)

    result = {
        "level":       label,
        "level_code":  cfg["code"],
        "description": _level_description(label),
        "probabilities": probabilities,
        "advice":      personalized["advice_text"],
        "suggestions": personalized["suggestions"],
        "calc_data":   personalized.get("calc_data", {}),
        "predict_date": date.today().strftime("%Y-%m-%d"),
        "bmi":         bmi,
        "confidence":  confidence,
        # 创新点信息
        "innovation_hybrid": innovation_hybrid,
        "innovation_shap": innovation_shap,
        "innovation_stacking": innovation_stacking,
        "shap_summary": shap_summary,
    }

    # 保存记录
    if current_user.is_authenticated:
        record = PredictRecord(
            user_id=current_user.id,
            gender=gender,
            age=age,
            height=height,
            weight=weight,
            bmi=bmi,
            family_history=family_history,
            high_calorie_food=high_calorie_food,
            physical_activity=physical_activity,
            screen_time=screen_time,
            alcohol=alcohol,
            transportation=transportation,
            smoking=0,
            vegetable_frequency=vegetable_freq,
            main_meals=main_meals,
            snacking=snacking,
            water_consumption=water_consumption,
            calorie_monitoring=calorie_monitoring,
            predicted_level=label,
            confidence=confidence,
            result_proba=json.dumps(proba_dict, ensure_ascii=False),
            risk_level=cfg["code"],
        )
        db.session.add(record)
        db.session.commit()

    history = []
    prefilled = {
        "gender":    gender,
        "age":       age,
        "height":    height,
        "weight":    weight,
    }
    if current_user.is_authenticated:
        history = PredictRecord.query.filter_by(user_id=current_user.id)\
            .order_by(PredictRecord.created_at.desc()).limit(10).all()
        for h in history:
            h.level_color = LEVEL_CONFIG.get(h.predicted_level, {}).get("color", "#6b8a77")

    return render_template("predict/index.html",
        result=result,
        history=history,
        prefilled=prefilled,
    )


@predict_bp.route("/history")
@login_required
def history():
    page = request.args.get("page", 1, type=int)
    pagination = PredictRecord.query.filter_by(user_id=current_user.id)\
        .order_by(PredictRecord.created_at.desc())\
        .paginate(page=page, per_page=20, error_out=False)

    for r in pagination.items:
        r.level_color = LEVEL_CONFIG.get(r.predicted_level, {}).get("color", "#6b8a77")

    return render_template("predict/history.html",
        records=pagination.items,
        pagination=pagination,
    )
