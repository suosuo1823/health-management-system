# -*- coding: utf-8 -*-
"""
app/routes/analysis.py  -  数据分析展示蓝图
"""

from datetime import date, timedelta
from flask import Blueprint, render_template, jsonify, Response
from flask_login import current_user
from app import db
from app.models.diet import DietRecord
from app.models.exercise import ExerciseRecord
from app.utils import parse_date
import sqlalchemy as sa
import csv
import io

analysis_bp = Blueprint("analysis", __name__, url_prefix="/analysis")

# 常量定义
DEFAULT_STATS_DAYS = 30      # 默认统计天数
DEFAULT_WEEK_DAYS = 7        # 默认周天数
DEFAULT_PAGE_SIZE = 20       # 默认分页大小
DEMO_DIET_RECORDS = 89       # 演示数据：饮食记录数
DEMO_EXERCISE_RECORDS = 42   # 演示数据：运动记录数
DEMO_AVG_DAILY_CAL = 1980    # 演示数据：平均每日热量
DEMO_AVG_DAILY_BURNED = 420  # 演示数据：平均每日消耗


def _demo_stats():
    """未登录时展示的示例数据"""
    return {
        "diet_records":       DEMO_DIET_RECORDS,
        "exercise_records":   DEMO_EXERCISE_RECORDS,
        "avg_daily_cal":      DEMO_AVG_DAILY_CAL,
        "avg_daily_burned":   DEMO_AVG_DAILY_BURNED,
    }


def _user_stats(user_id):
    days = DEFAULT_STATS_DAYS
    end_date   = date.today()
    start_date = end_date - timedelta(days=days - 1)

    diet_count  = DietRecord.query.filter_by(user_id=user_id).count()
    ex_count    = ExerciseRecord.query.filter_by(user_id=user_id).count()

    total_cal   = db.session.query(sa.func.sum(DietRecord.calories))\
        .filter(DietRecord.user_id == user_id,
                 DietRecord.record_date.between(start_date, end_date)).scalar() or 0
    total_burned = db.session.query(sa.func.sum(ExerciseRecord.calories_burned))\
        .filter(ExerciseRecord.user_id == user_id,
                ExerciseRecord.exercise_date.between(start_date, end_date)).scalar() or 0

    active_days = db.session.query(sa.func.count(sa.func.distinct(DietRecord.record_date)))\
        .filter(DietRecord.user_id == user_id,
                DietRecord.record_date.between(start_date, end_date)).scalar() or 1

    return {
        "diet_records":     diet_count,
        "exercise_records": ex_count,
        "avg_daily_cal":    round(float(total_cal) / active_days, 0),
        "avg_daily_burned": round(float(total_burned) / active_days, 0),
    }


@analysis_bp.route("/")
def index():
    if not current_user.is_authenticated:
        # 未登录：展示示例仪表盘
        demo_data = _get_demo_charts_data()
        return render_template("analysis/index.html",
            demo=True,
            stats=_demo_stats(),
            **demo_data,
        )

    stats = _user_stats(current_user.id)
    charts = _user_charts_data(current_user.id)
    return render_template("analysis/index.html",
        demo=False,
        stats=stats,
        **charts,
    )


def _user_charts_data(user_id):
    days = DEFAULT_STATS_DAYS
    end_date   = date.today()
    start_date = end_date - timedelta(days=days - 1)

    # ── 热量趋势：一次 SQLGROUP BY 取代 30x2 次循环查询 ─────
    trend_map: dict = {d: {"intake": 0.0, "burned": 0.0}
                       for d in (start_date + timedelta(days=i) for i in range(days))}

    intake_rows = db.session.query(
        DietRecord.record_date,
        sa.func.coalesce(sa.func.sum(DietRecord.calories), 0.0)
    ).filter(
        DietRecord.user_id == user_id,
        DietRecord.record_date.between(start_date, end_date)
    ).group_by(DietRecord.record_date).all()

    burned_rows = db.session.query(
        ExerciseRecord.exercise_date,
        sa.func.coalesce(sa.func.sum(ExerciseRecord.calories_burned), 0.0)
    ).filter(
        ExerciseRecord.user_id == user_id,
        ExerciseRecord.exercise_date.between(start_date, end_date)
    ).group_by(ExerciseRecord.exercise_date).all()

    for d, val in intake_rows:
        trend_map[d] = dict(trend_map.get(d, {}), intake=float(val))
    for d, val in burned_rows:
        trend_map[d] = dict(trend_map.get(d, {}), burned=float(val))

    trend_dates    = [d.strftime("%m-%d") for d in sorted(trend_map)]
    intake_values  = [round(trend_map[d]["intake"], 1)  for d in sorted(trend_map)]
    burned_values  = [round(trend_map[d]["burned"], 1)  for d in sorted(trend_map)]

    # 营养素
    diet_rows = DietRecord.query.filter_by(user_id=user_id)\
        .filter(DietRecord.record_date.between(start_date, end_date)).all()
    total_protein = round(sum(r.protein or 0 for r in diet_rows), 1)
    total_carb    = round(sum(r.carbs    or 0 for r in diet_rows), 1)
    total_fat     = round(sum(r.fat     or 0 for r in diet_rows), 1)

    # 运动类型
    ex_rows = ExerciseRecord.query.filter_by(user_id=user_id)\
        .filter(ExerciseRecord.exercise_date.between(start_date, end_date)).all()
    # 从 constants.py 引用统一常量，不再重复定义
    from app.ml.constants import EXERCISE_TYPE_NAMES
    type_counts = {}
    for r in ex_rows:
        name = EXERCISE_TYPE_NAMES.get(r.exercise_type, r.exercise_type)
        type_counts[name] = type_counts.get(name, 0) + 1

    exercise_types  = list(type_counts.keys())
    exercise_counts = list(type_counts.values())

    # 热量收支对比（最近7天）：一次 GROUP BY 取代 7x2 次循环查询
    week_start = end_date - timedelta(days=DEFAULT_WEEK_DAYS - 1)
    compare_intake_rows = db.session.query(
        DietRecord.record_date,
        sa.func.coalesce(sa.func.sum(DietRecord.calories), 0.0)
    ).filter(
        DietRecord.user_id == user_id,
        DietRecord.record_date.between(week_start, end_date)
    ).group_by(DietRecord.record_date).all()

    compare_burned_rows = db.session.query(
        ExerciseRecord.exercise_date,
        sa.func.coalesce(sa.func.sum(ExerciseRecord.calories_burned), 0.0)
    ).filter(
        ExerciseRecord.user_id == user_id,
        ExerciseRecord.exercise_date.between(week_start, end_date)
    ).group_by(ExerciseRecord.exercise_date).all()

    intake_map  = {d: float(v) for d, v in compare_intake_rows}
    burned_map  = {d: float(v) for d, v in compare_burned_rows}
    compare_dates, compare_intake, compare_burned = [], [], []
    for i in range(DEFAULT_WEEK_DAYS):
        d = end_date - timedelta(days=DEFAULT_WEEK_DAYS - 1 - i)
        compare_dates.append(d.strftime("%m-%d"))
        compare_intake.append(round(intake_map.get(d, 0.0), 1))
        compare_burned.append(round(burned_map.get(d, 0.0), 1))

    # 餐次分布
    meal_totals = {"breakfast": 0, "lunch": 0, "dinner": 0, "snack": 0}
    for r in diet_rows:
        if r.meal_type in meal_totals:
            meal_totals[r.meal_type] += r.calories or 0

    # 热量平衡
    total_intake   = sum(intake_values)
    total_burned   = sum(burned_values)
    balance_ratio  = round(total_burned / total_intake, 2) if total_intake > 0 else 1.0

    # ── 运动时长趋势：一次 GROUP BY 取代 30 次循环查询 ─────
    ex_dur_rows = db.session.query(
        ExerciseRecord.exercise_date,
        sa.func.coalesce(sa.func.sum(ExerciseRecord.duration), 0)
    ).filter(
        ExerciseRecord.user_id == user_id,
        ExerciseRecord.exercise_date.between(start_date, end_date)
    ).group_by(ExerciseRecord.exercise_date).all()
    ex_dur_map = {d: v for d, v in ex_dur_rows}

    exercise_trend_dates  = [d.strftime("%m-%d") for d in sorted(ex_dur_map)]
    exercise_trend_values = [round(float(ex_dur_map[d]), 1) for d in sorted(ex_dur_map)]

    return {
        "trend_dates":        trend_dates,
        "intake_values":     intake_values,
        "burned_values":     burned_values,
        "total_protein":     total_protein,
        "total_carb":        total_carb,
        "total_fat":         total_fat,
        "exercise_types":    exercise_types or ["暂无数据"],
        "exercise_counts":   exercise_counts or [0],
        "compare_dates":     compare_dates,
        "compare_intake":    compare_intake,
        "compare_burned":    compare_burned,
        "meal_breakfast":    round(meal_totals["breakfast"], 1),
        "meal_lunch":        round(meal_totals["lunch"],     1),
        "meal_dinner":       round(meal_totals["dinner"],    1),
        "meal_snack":        round(meal_totals["snack"],     1),
        "balance_ratio":     balance_ratio,
        "exercise_trend_dates":  exercise_trend_dates,
        "exercise_trend_values": exercise_trend_values,
    }


def _get_demo_charts_data():
    """未登录示例数据"""
    dates7  = [(date.today() - timedelta(days=6-i)).strftime("%m-%d") for i in range(7)]
    dates30 = [(date.today() - timedelta(days=29-i)).strftime("%m-%d") for i in range(30)]
    return {
        "trend_dates":     dates30,
        "intake_values":   [2100,2250,1980,2350,2150,1890,2200,2100,2300,2050,2180,2350,2100,2250,1900,2150,2300,1980,2200,2100,2250,2150,2300,2050,2180,2100,2350,2200,2150,1980],
        "burned_values":   [380, 450, 320, 280, 520, 410, 360, 380, 470, 340, 510, 390, 370, 430, 300, 480, 350, 420, 390, 360, 440, 380, 460, 330, 500, 370, 410, 350, 390, 340],
        "total_protein":   1850,
        "total_carb":      5200,
        "total_fat":       2900,
        "exercise_types":  ["跑步","骑行","游泳","健身房","瑜伽","篮球","羽毛球"],
        "exercise_counts": [8, 5, 3, 7, 4, 2, 3],
        "compare_dates":   dates7,
        "compare_intake":  [2100,2250,1980,2350,2150,1890,2200],
        "compare_burned":  [380, 450, 320, 280, 520, 410, 360],
        "meal_breakfast":  4520,
        "meal_lunch":      7180,
        "meal_dinner":     6770,
        "meal_snack":      2050,
        "balance_ratio":   0.21,
        "exercise_trend_dates":  dates30,
        "exercise_trend_values": [45, 60, 30, 0, 75, 55, 40, 45, 65, 35, 70, 50, 40, 55, 25, 65, 45, 55, 50, 40, 60, 45, 70, 35, 80, 50, 60, 45, 55, 40],
    }


# ── 导出功能 ────────────────────────────────────────────────────────

MEAL_TYPE_MAP = {
    "breakfast": "早餐", "lunch": "午餐",
    "dinner": "晚餐",   "snack": "零食",
}

EXERCISE_TYPE_MAP = {
    "running": "跑步", "walking": "步行", "cycling": "骑行",
    "swimming": "游泳", "jumping_rope": "跳绳", "yoga": "瑜伽",
    "gym": "健身房", "basketball": "篮球", "football": "足球",
    "badminton": "羽毛球", "dancing": "跳舞", "hiking": "徒步",
    "climbing": "爬山", "other": "其他",
}

INTENSITY_MAP = {
    "low": "低强度", "medium": "中等强度", "high": "高强度",
}


@analysis_bp.route("/export/diet")
def export_diet():
    """导出饮食记录 CSV"""
    if not current_user.is_authenticated:
        return jsonify({"code": 401, "msg": "请先登录"})

    records = DietRecord.query.filter_by(user_id=current_user.id)\
        .order_by(DietRecord.record_date.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["日期", "餐次", "食物名称", "份量(g)", "热量(kcal)",
                     "蛋白质(g)", "碳水(g)", "脂肪(g)", "备注"])

    for r in records:
        writer.writerow([
            r.record_date.strftime("%Y-%m-%d"),
            MEAL_TYPE_MAP.get(r.meal_type, r.meal_type),
            r.food_name or "",
            r.portion or "",
            round(r.calories, 1) if r.calories else "",
            round(r.protein, 1) if r.protein else "",
            round(r.carbs, 1) if r.carbs else "",
            round(r.fat, 1) if r.fat else "",
            r.notes or "",
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv; charset=utf-8-sig",
        headers={
            "Content-Disposition": f"attachment; filename=diet_records_{date.today().strftime('%Y%m%d')}.csv",
            "Content-Type": "text/csv; charset=utf-8-sig",
        },
    )


@analysis_bp.route("/export/exercise")
def export_exercise():
    """导出运动记录 CSV"""
    if not current_user.is_authenticated:
        return jsonify({"code": 401, "msg": "请先登录"})

    records = ExerciseRecord.query.filter_by(user_id=current_user.id)\
        .order_by(ExerciseRecord.exercise_date.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["日期", "运动类型", "强度", "时长(min)", "消耗(kcal)", "备注"])

    for r in records:
        writer.writerow([
            r.exercise_date.strftime("%Y-%m-%d"),
            EXERCISE_TYPE_MAP.get(r.exercise_type, r.exercise_type),
            INTENSITY_MAP.get(r.intensity, r.intensity or ""),
            r.duration or "",
            round(r.calories_burned, 1) if r.calories_burned else "",
            r.notes or "",
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv; charset=utf-8-sig",
        headers={
            "Content-Disposition": f"attachment; filename=exercise_records_{date.today().strftime('%Y%m%d')}.csv",
            "Content-Type": "text/csv; charset=utf-8-sig",
        },
    )
