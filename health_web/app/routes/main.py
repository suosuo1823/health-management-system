# -*- coding: utf-8 -*-
"""
app/routes/main.py  -  主页/仪表盘
"""

from datetime import datetime, timedelta
from flask import Blueprint, render_template, jsonify, Response
from flask_login import current_user
from app.models.diet     import DietRecord
from app.models.exercise import ExerciseRecord
from app.models.health   import HealthRecord
from app.models.predict  import PredictRecord

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return render_template("main/index.html")


# ── SEO / 站点地图 ─────────────────────────────────────────────────────────

@main_bp.route("/robots.txt")
def robots_txt():
    """提供 robots.txt"""
    return Response(
        "User-agent: *\nAllow: /\n"
        "Disallow: /auth/\nDisallow: /diet/\nDisallow: /exercise/\n"
        "Disallow: /analysis/\nDisallow: /predict/\nDisallow: /health/\n"
        "Disallow: /api/\nDisallow: /recipes/\n\nSitemap: /sitemap.xml\n",
        mimetype="text/plain",
    )


@main_bp.route("/sitemap.xml")
def sitemap_xml():
    """提供 sitemap.xml（动态路由，保证 URL 绝对路径正确）"""
    from flask import request
    base = request.url_root.rstrip("/")
    pages = [
        ("",         1.0, "daily",   ""),
        ("/auth/login",    0.8, "monthly", ""),
        ("/auth/register", 0.8, "monthly", ""),
        ("/predict/",      0.9, "weekly",  ""),
        ("/diet/",         0.7, "weekly",  ""),
        ("/exercise/",     0.7, "weekly",  ""),
        ("/analysis/",     0.8, "weekly",  ""),
        ("/recipes/",      0.7, "monthly", ""),
    ]
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    for path, priority, freq, _ in pages:
        xml += f'  <url>\n    <loc>{base}{path}</loc>\n'
        xml += f'    <priority>{priority}</priority>\n'
        xml += f'    <changefreq>{freq}</changefreq>\n  </url>\n'
    xml += '</urlset>'
    return Response(xml, mimetype="application/xml")


@main_bp.route("/dashboard")
def dashboard():
    demo_data = None
    if not current_user.is_authenticated:
        demo_data = _get_demo_data()
        return render_template("main/dashboard.html", demo=True, demo_data=demo_data)

    today = datetime.now().date()
    week_ago = today - timedelta(days=6)

    # 今日汇总
    today_diet = DietRecord.query.filter_by(user_id=current_user.id)\
        .filter(DietRecord.record_date == today).all()
    today_calories_in  = round(sum(r.calories for r in today_diet), 1)
    today_protein      = round(sum(r.protein  for r in today_diet), 1)
    today_fat          = round(sum(r.fat      for r in today_diet), 1)
    today_carbs        = round(sum(r.carbs    for r in today_diet), 1)

    today_exercise = ExerciseRecord.query.filter_by(user_id=current_user.id)\
        .filter(ExerciseRecord.exercise_date == today).all()
    today_calories_out = round(sum(r.calories_burned for r in today_exercise), 1)
    today_duration     = sum(r.duration for r in today_exercise)

    # 最新体重/BMI
    latest_health = HealthRecord.query.filter_by(user_id=current_user.id)\
        .order_by(HealthRecord.record_date.desc()).first()

    # 最新预测结果
    latest_predict = PredictRecord.query.filter_by(user_id=current_user.id)\
        .order_by(PredictRecord.created_at.desc()).first()

    # 近7天热量趋势
    weekly_data = _get_weekly_trend(current_user.id, week_ago, today)

    # 连续打卡 + 徽章
    streak_data = _calc_streak(current_user.id)

    return render_template(
        "main/dashboard.html",
        demo=False,
        today_calories_in=today_calories_in,
        today_calories_out=today_calories_out,
        today_protein=today_protein,
        today_fat=today_fat,
        today_carbs=today_carbs,
        today_duration=today_duration,
        latest_health=latest_health,
        latest_predict=latest_predict,
        weekly_data=weekly_data,
        streak=streak_data["streak"],
        badges=streak_data["badges"],
    )


def _get_weekly_trend(user_id, start_date, end_date):
    from app import db
    import sqlalchemy as sa
    # 一次 GROUP BY 取代 7x2 次循环查询
    diet_rows = db.session.query(
        DietRecord.record_date,
        sa.func.coalesce(sa.func.sum(DietRecord.calories), 0.0)
    ).filter(
        DietRecord.user_id == user_id,
        DietRecord.record_date.between(start_date, end_date)
    ).group_by(DietRecord.record_date).all()

    ex_rows = db.session.query(
        ExerciseRecord.exercise_date,
        sa.func.coalesce(sa.func.sum(ExerciseRecord.calories_burned), 0.0)
    ).filter(
        ExerciseRecord.user_id == user_id,
        ExerciseRecord.exercise_date.between(start_date, end_date)
    ).group_by(ExerciseRecord.exercise_date).all()

    diet_map = {d: v for d, v in diet_rows}
    ex_map   = {d: v for d, v in ex_rows}

    delta = end_date - start_date
    dates, calories_in, calories_out = [], [], []
    for i in range(delta.days + 1):
        d = start_date + timedelta(days=i)
        dates.append(d.strftime("%m-%d"))
        calories_in.append(round(float(diet_map.get(d, 0.0)), 1))
        calories_out.append(round(float(ex_map.get(d, 0.0)), 1))

    return {"dates": dates, "calories_in": calories_in, "calories_out": calories_out}


def _get_demo_data():
    """未登录状态下展示的示例数据（静态，不查数据库）"""
    return {
        "today_calories_in":  2150,
        "today_calories_out": 480,
        "today_protein":      82.5,
        "today_fat":          65.3,
        "today_carbs":        265.0,
        "today_duration":     65,
        "bmi":                22.4,
        "weight":             68.0,
        "risk_label":         "正常体重",
        "risk_level":         "normal",
        "streak_diet":        7,
        "streak_exercise":    3,
        "badges":             [],
        "weekly_data": {
            "dates":        ["04-06", "04-07", "04-08", "04-09", "04-10", "04-11", "04-12"],
            "calories_in":  [2100, 2300, 1950, 2450, 2200, 1850, 2150],
            "calories_out": [300,  520,  420,  180,  650,  380,  480],
        },
        "recent_diet": [
            {"food_name": "糙米饭",  "meal_type": "午餐", "amount_g": 200, "calories": 232},
            {"food_name": "鸡胸肉",  "meal_type": "午餐", "amount_g": 150, "calories": 165},
            {"food_name": "西兰花",  "meal_type": "午餐", "amount_g": 100, "calories": 34},
            {"food_name": "全麦面包","meal_type": "早餐", "amount_g": 80,  "calories": 196},
            {"food_name": "脱脂牛奶","meal_type": "早餐", "amount_g": 250, "calories": 90},
        ],
        "recent_exercise": [
            {"exercise_name": "跑步",    "duration_min": 40, "calories_burned": 380},
            {"exercise_name": "力量训练", "duration_min": 25, "calories_burned": 150},
            {"exercise_name": "游泳",    "duration_min": 30, "calories_burned": 320},
        ],
    }


def _calc_streak(user_id):
    """计算连续打卡天数和已获徽章"""
    from app import db
    import sqlalchemy as sa

    today = datetime.now().date()

    # 获取用户所有有记录的日期（饮食 + 运动合并去重）
    # LIMIT 365: 打卡天数只需近一年，节省查询开销
    diet_dates = db.session.query(sa.func.distinct(DietRecord.record_date))\
        .filter(DietRecord.user_id == user_id)\
        .order_by(DietRecord.record_date.desc()).limit(365).all()
    ex_dates   = db.session.query(sa.func.distinct(ExerciseRecord.exercise_date))\
        .filter(ExerciseRecord.user_id == user_id)\
        .order_by(ExerciseRecord.exercise_date.desc()).limit(365).all()
    all_dates  = set(d.strftime("%Y-%m-%d") for d, in diet_dates) | \
                set(d.strftime("%Y-%m-%d") for d, in ex_dates)

    # 计算从今天往前推的连续天数
    streak = 0
    d = today
    while True:
        if d.strftime("%Y-%m-%d") in all_dates:
            streak += 1
            d -= timedelta(days=1)
        else:
            break

    # 徽章定义
    badges = []
    BADGE_DEFS = [
        ("week_streak",  7,  "连续打卡一周",  "fa-calendar-check-o", "#2d8653"),
        ("month_streak", 30, "连续打卡一月",  "fa-trophy",          "#f0a500"),
        ("perfect_week", 7,  "完美一周",      "fa-star",            "#7b2ff7"),
    ]
    for bid, req, name, icon, color in BADGE_DEFS:
        badges.append({"id": bid, "name": name, "icon": icon,
                       "color": color, "unlocked": streak >= req,
                       "progress": min(100, int(streak / req * 100))})

    return {"streak": streak, "badges": badges}
