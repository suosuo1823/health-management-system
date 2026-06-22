# -*- coding: utf-8 -*-
"""
app/routes/api.py  -  通用 REST API 蓝图（AJAX 接口）
"""

from datetime import date, timedelta, datetime
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from app import db
from app.models.diet     import DietRecord
from app.models.exercise import ExerciseRecord
from app.models.health   import HealthRecord
from app.utils import safe_float, safe_int, parse_date, db_transaction
import sqlalchemy as sa

api_bp = Blueprint("api", __name__)


@api_bp.route("/health/add", methods=["POST"])
@login_required
def add_health():
    data = request.get_json() or request.form.to_dict()
    record_date = parse_date(data.get("record_date"), date.today())

    # 如果当天已有记录则更新
    existing = HealthRecord.query.filter_by(
        user_id=current_user.id, record_date=record_date
    ).first()

    weight  = safe_float(data.get("weight"), current_user.weight)
    height  = safe_float(data.get("height"), current_user.height)
    h_m     = height / 100.0
    bmi     = round(weight / (h_m * h_m), 2) if h_m > 0 else 0

    if existing:
        r = existing
    else:
        r = HealthRecord(user_id=current_user.id, weight=weight, height=height)
        db.session.add(r)

    r.weight       = weight
    r.height       = height
    r.bmi          = bmi
    r.body_fat_pct = safe_float(data.get("body_fat_pct"), r.body_fat_pct or 0)
    r.muscle_mass  = safe_float(data.get("muscle_mass"),  r.muscle_mass  or 0)
    r.waist_cm     = safe_float(data.get("waist_cm"),     r.waist_cm     or 0)
    r.hip_cm       = safe_float(data.get("hip_cm"),       r.hip_cm       or 0)
    r.systolic_bp  = safe_int(data.get("systolic_bp"),    r.systolic_bp  or 0)
    r.diastolic_bp = safe_int(data.get("diastolic_bp"),   r.diastolic_bp or 0)
    r.resting_hr   = safe_int(data.get("resting_hr"),     r.resting_hr   or 0)
    r.blood_glucose = safe_float(data.get("blood_glucose"), r.blood_glucose or 0)
    r.sleep_hours  = safe_float(data.get("sleep_hours"),  r.sleep_hours  or 7.0)
    r.water_ml     = safe_int(data.get("water_ml"),       r.water_ml     or 1500)
    r.notes        = data.get("notes", r.notes or "")
    r.record_date  = record_date

    # 同步更新用户体重身高
    current_user.weight = weight
    current_user.height = height
    db.session.commit()

    return jsonify({"code": 0, "msg": "保存成功", "record": r.to_dict()})


@api_bp.route("/health/trend")
@login_required
def health_trend():
    days = int(request.args.get("days", 30))
    end_date   = date.today()
    start_date = end_date - timedelta(days=days - 1)
    rows = HealthRecord.query.filter_by(user_id=current_user.id)\
        .filter(HealthRecord.record_date.between(start_date, end_date))\
        .order_by(HealthRecord.record_date).all()
    return jsonify({
        "dates":   [r.record_date.strftime("%Y-%m-%d") for r in rows],
        "weights": [r.weight     for r in rows],
        "bmi":     [r.bmi        for r in rows],
        "sleep":   [r.sleep_hours for r in rows],
    })


@api_bp.route("/diet/summary")
@login_required
def diet_summary():
    target_date_str = request.args.get("date", date.today().strftime("%Y-%m-%d"))
    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    except ValueError:
        target_date = date.today()

    rows = DietRecord.query.filter_by(user_id=current_user.id, record_date=target_date).all()
    summary = {
        "calories": round(sum(r.calories for r in rows), 1),
        "protein":  round(sum(r.protein  for r in rows), 1),
        "fat":      round(sum(r.fat      for r in rows), 1),
        "carbs":    round(sum(r.carbs    for r in rows), 1),
        "fiber":    round(sum(r.fiber    for r in rows), 1),
        "count":    len(rows),
        "records":  [r.to_dict() for r in rows],
    }
    return jsonify(summary)


@api_bp.route("/exercise/summary")
@login_required
def exercise_summary():
    target_date_str = request.args.get("date", date.today().strftime("%Y-%m-%d"))
    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    except ValueError:
        target_date = date.today()

    rows = ExerciseRecord.query.filter_by(user_id=current_user.id, exercise_date=target_date).all()
    summary = {
        "calories_burned": round(sum(r.calories_burned for r in rows), 1),
        "duration":    sum(r.duration for r in rows),
        "count":           len(rows),
        "records":         [r.to_dict() for r in rows],
    }
    return jsonify(summary)


@api_bp.route("/user/stats")
@login_required
def user_stats():
    """用户综合统计数据（用于仪表盘小部件）"""
    total_diet     = DietRecord.query.filter_by(user_id=current_user.id).count()
    total_exercise = ExerciseRecord.query.filter_by(user_id=current_user.id).count()
    total_health   = HealthRecord.query.filter_by(user_id=current_user.id).count()

    latest_weight = db.session.query(sa.func.max(HealthRecord.record_date))\
        .filter_by(user_id=current_user.id).scalar()
    weight_record = None
    if latest_weight:
        weight_record = HealthRecord.query.filter_by(
            user_id=current_user.id, record_date=latest_weight
        ).first()

    return jsonify({
        "total_diet":     total_diet,
        "total_exercise": total_exercise,
        "total_health":   total_health,
        "current_weight": weight_record.weight if weight_record else current_user.weight,
        "current_bmi":    weight_record.bmi    if weight_record else current_user.bmi,
    })
