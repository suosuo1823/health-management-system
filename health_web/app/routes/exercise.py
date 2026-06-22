# -*- coding: utf-8 -*-
"""
app/routes/exercise.py  -  运动记录蓝图
"""

from datetime import datetime, date
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from app import db
from app.models.exercise import ExerciseRecord
from app.ml.constants import EXERCISE_TYPE_NAMES
from app.utils import safe_float, safe_int, parse_date, db_transaction

exercise_bp = Blueprint("exercise", __name__, url_prefix="/exercise")


def _type_name(code):
    return EXERCISE_TYPE_NAMES.get(code, code)


@exercise_bp.route("/")
@login_required
def index():
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")

    records = ExerciseRecord.query.filter_by(user_id=current_user.id)\
        .filter(ExerciseRecord.exercise_date == today)\
        .order_by(ExerciseRecord.created_at.desc()).all()

    for r in records:
        r.exercise_type_name = _type_name(r.exercise_type)

    summary = {
        "total_duration":       sum(r.duration or 0 for r in records),
        "total_calories_burned": sum(r.calories_burned or 0 for r in records),
        "total_distance":       sum(r.distance or 0 for r in records),
        "total_steps":          sum(r.steps or 0 for r in records),
    }

    return render_template("exercise/index.html",
        today_str=today_str,
        today_records=records,
        summary=summary,
    )


@exercise_bp.route("/", methods=["POST"])
@login_required
def add():
    ex_date = parse_date(request.form.get("exercise_date"), date.today())

    try:
        with db_transaction():
            record = ExerciseRecord(
                user_id=current_user.id,
                exercise_type=request.form.get("exercise_type", "running"),
                exercise_date=ex_date,
                duration=safe_float(request.form.get("duration"), 0),
                calories_burned=safe_float(request.form.get("calories_burned"), 0),
                distance=safe_float(request.form.get("distance"), 0),
                steps=safe_int(request.form.get("steps"), 0),
                heart_rate=safe_int(request.form.get("heart_rate"), 0) or None,
                intensity=request.form.get("intensity", "medium"),
                notes=request.form.get("notes", ""),
            )
            db.session.add(record)
        flash("运动记录已添加", "success")
    except Exception:
        flash("添加记录失败，请稍后重试", "danger")
    return redirect(url_for("exercise.index"))


@exercise_bp.route("/delete/<int:record_id>", methods=["POST"])
@login_required
def delete(record_id):
    record = ExerciseRecord.query.filter_by(id=record_id, user_id=current_user.id).first_or_404()
    try:
        with db_transaction():
            db.session.delete(record)
        flash("记录已删除", "info")
    except Exception:
        flash("删除失败，请稍后重试", "danger")
    return redirect(url_for("exercise.index"))


@exercise_bp.route("/history")
@login_required
def history():
    page = request.args.get("page", 1, type=int)
    start_date     = request.args.get("start_date", "")
    end_date       = request.args.get("end_date", "")
    exercise_type  = request.args.get("exercise_type", "")

    query = ExerciseRecord.query.filter_by(user_id=current_user.id)
    if start_date:
        query = query.filter(ExerciseRecord.exercise_date >= parse_date(start_date, date.min))
    if end_date:
        query = query.filter(ExerciseRecord.exercise_date <= parse_date(end_date, date.max))
    if exercise_type:
        query = query.filter_by(exercise_type=exercise_type)

    pagination = query.order_by(
        ExerciseRecord.exercise_date.desc(),
        ExerciseRecord.created_at.desc()
    ).paginate(page=page, per_page=20, error_out=False)

    for r in pagination.items:
        r.exercise_type_name = _type_name(r.exercise_type)

    return render_template("exercise/history.html",
        records=pagination.items,
        pagination=pagination,
        start_date=start_date,
        end_date=end_date,
        exercise_type=exercise_type,
        types=list(EXERCISE_TYPE_NAMES.keys()),
    )
