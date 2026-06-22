# -*- coding: utf-8 -*-
"""
app/routes/diet.py  -  饮食记录蓝图
"""

from datetime import datetime, date
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from app import db
from app.models.diet import DietRecord, FoodItem
from app.utils import safe_float, parse_date, db_transaction

diet_bp = Blueprint("diet", __name__, url_prefix="/diet")


@diet_bp.route("/")
@login_required
def index():
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")

    records = DietRecord.query.filter_by(user_id=current_user.id)\
        .filter(DietRecord.record_date == today)\
        .order_by(DietRecord.created_at.desc()).all()

    # 今日汇总
    summary = {
        "total_calories": sum(r.calories or 0 for r in records),
        "total_protein":   sum(r.protein  or 0 for r in records),
        "total_carb":      sum(r.carbs    or 0 for r in records),
        "total_fat":       sum(r.fat      or 0 for r in records),
    }

    return render_template("diet/index.html",
        today_str=today_str,
        today_records=records,
        summary=summary,
    )


@diet_bp.route("/", methods=["POST"])
@login_required
def add():
    record_date = request.form.get("record_date", date.today().strftime("%Y-%m-%d"))
    rdate = parse_date(record_date, date.today())

    try:
        with db_transaction():
            record = DietRecord(
                user_id=current_user.id,
                food_name=request.form.get("food_name", "").strip(),
                meal_type=request.form.get("meal_type", "lunch"),
                portion=request.form.get("portion", "100g"),
                calories=safe_float(request.form.get("calories"), 0),
                protein=safe_float(request.form.get("protein"), 0),
                carbs=safe_float(request.form.get("carbs"), 0),
                fat=safe_float(request.form.get("fat"), 0),
                notes=request.form.get("notes", ""),
                record_date=rdate,
            )
            db.session.add(record)
        flash("饮食记录已添加", "success")
    except Exception:
        flash("添加记录失败，请稍后重试", "danger")
    return redirect(url_for("diet.index"))


@diet_bp.route("/api/food-search")
def api_food_search():
    """
    食物快速搜索 API
    Query params:
      q   - 关键字（模糊匹配）
      cat - 可选：分类筛选（谷薯类/蔬菜/水果/肉类/水产/坚果/乳类/豆制品）
      limit - 返回条数（默认10，最大30）
    """
    q = request.args.get("q", "").strip()
    cat = request.args.get("cat", "").strip()
    limit = min(30, max(5, int(request.args.get("limit", 10) or 10)))

    query = db.session.query(FoodItem)
    if q:
        query = query.filter(FoodItem.name.ilike(f"%{q}%"))
    if cat:
        query = query.filter(FoodItem.category == cat)

    rows = query.order_by(FoodItem.calories_per_100g.asc())\
                .limit(limit).all()

    return jsonify({
        "items": [r.to_dict() for r in rows],
        "count": len(rows),
    })


@diet_bp.route("/api/food-categories")
def api_food_categories():
    """返回所有食物分类选项"""
    cats = sorted(set(
        r[0] for r in db.session.query(FoodItem.category).distinct().all() if r[0]
    ))
    return jsonify({"categories": cats})


@diet_bp.route("/delete/<int:record_id>", methods=["POST"])
@login_required
def delete(record_id):
    record = DietRecord.query.filter_by(id=record_id, user_id=current_user.id).first_or_404()
    try:
        with db_transaction():
            db.session.delete(record)
        flash("记录已删除", "info")
    except Exception:
        flash("删除失败，请稍后重试", "danger")
    return redirect(url_for("diet.index"))


@diet_bp.route("/history")
@login_required
def history():
    page = request.args.get("page", 1, type=int)
    start_date = request.args.get("start_date", "")
    end_date   = request.args.get("end_date", "")
    meal_type  = request.args.get("meal_type", "")

    query = DietRecord.query.filter_by(user_id=current_user.id)
    if start_date:
        query = query.filter(DietRecord.record_date >= parse_date(start_date, date.min))
    if end_date:
        query = query.filter(DietRecord.record_date <= parse_date(end_date, date.max))
    if meal_type:
        query = query.filter_by(meal_type=meal_type)

    pagination = query.order_by(
        DietRecord.record_date.desc(),
        DietRecord.created_at.desc()
    ).paginate(page=page, per_page=20, error_out=False)

    return render_template("diet/history.html",
        records=pagination.items,
        pagination=pagination,
        start_date=start_date,
        end_date=end_date,
        meal_type=meal_type,
    )
