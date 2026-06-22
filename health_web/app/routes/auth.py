# -*- coding: utf-8 -*-
"""
app/routes/auth.py  -  认证蓝图：注册/登录/登出/个人资料
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app import db, bcrypt, limiter
from app.models.user import User
from app.models.diet import DietRecord
from app.models.exercise import ExerciseRecord
from app.models.predict import PredictRecord
from app.utils import validate_password, safe_float, db_transaction

auth_bp = Blueprint("auth", __name__)


# ── 频率限制：每人每分钟最多 10 次尝试（防暴力破解）────────────────────────
_auth_limit = limiter.limit("10 per minute")


@auth_bp.route("/login", methods=["GET", "POST"])
@_auth_limit
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    username = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        remember = bool(request.form.get("remember"))

        user = User.query.filter(
            (User.username == username) | (User.email == username)
        ).first()

        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash("登录成功！欢迎回来", "success")
            return redirect(url_for("main.dashboard"))
        else:
            flash("用户名或密码错误", "danger")

    return render_template("auth/login.html", username=username)


@auth_bp.route("/register", methods=["GET", "POST"])
@_auth_limit
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        username   = request.form.get("username", "").strip()
        email      = request.form.get("email", "").strip()
        password   = request.form.get("password", "")
        password2  = request.form.get("password2", "")
        nickname   = request.form.get("nickname", "").strip() or username
        realname   = request.form.get("realname", "").strip()
        gender     = request.form.get("gender", "male")
        height_val = request.form.get("height", "")
        weight_val = request.form.get("weight", "")

        # 验证
        if not username or not email or not password:
            flash("请填写所有必填项", "danger")
            return render_template("auth/register.html", username=username, email=email, nickname=username,
                                   gender=gender, height=height_val, weight=weight_val)

        if password != password2:
            flash("两次输入的密码不一致", "danger")
            return render_template("auth/register.html", username=username, email=email, nickname=username,
                                   gender=gender, height=height_val, weight=weight_val)

        # 密码复杂度验证
        is_valid, error_msg = validate_password(password)
        if not is_valid:
            flash(f"密码不符合要求：{error_msg}", "danger")
            return render_template("auth/register.html", username=username, email=email, nickname=username,
                                   gender=gender, height=height_val, weight=weight_val)

        if User.query.filter_by(username=username).first():
            flash("用户名已存在", "danger")
            return render_template("auth/register.html", username=username, email=email, nickname=username,
                                   gender=gender, height=height_val, weight=weight_val)

        if User.query.filter_by(email=email).first():
            flash("邮箱已被注册", "danger")
            return render_template("auth/register.html", username=username, email=email, nickname=username,
                                   gender=gender, height=height_val, weight=weight_val)

        # 使用事务管理器确保数据一致性
        try:
            with db_transaction():
                new_user = User(
                    username=username,
                    email=email,
                    nickname=nickname,
                    realname=realname,
                    gender=gender,
                    height=safe_float(height_val, 170.0),
                    weight=safe_float(weight_val, 65.0),
                )
                new_user.set_password(password)
                db.session.add(new_user)
            
            # 事务成功后登录
            login_user(new_user)
            flash("注册成功！欢迎加入健康管家", "success")
            return redirect(url_for("main.dashboard"))
        except Exception as e:
            flash("注册失败，请稍后重试", "danger")
            return render_template("auth/register.html", username=username, email=email, nickname=username,
                                   gender=gender, height=height_val, weight=weight_val)

    return render_template("auth/register.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("已安全退出", "info")
    return redirect(url_for("main.index"))


@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    # 统计数据
    stats = {
        "diet_count":    DietRecord.query.filter_by(user_id=current_user.id).count(),
        "exercise_count": ExerciseRecord.query.filter_by(user_id=current_user.id).count(),
        "predict_count":  PredictRecord.query.filter_by(user_id=current_user.id).count(),
    }

    if request.method == "POST":
        action = request.form.get("action", "update_profile")

        if action == "update_profile":
            try:
                with db_transaction():
                    current_user.nickname = request.form.get("nickname", current_user.nickname)
                    current_user.email    = request.form.get("email", current_user.email)
                    current_user.realname = request.form.get("realname", "")
                    current_user.gender   = request.form.get("gender", current_user.gender)
                    h = request.form.get("height", "")
                    w = request.form.get("weight", "")
                    if h:
                        current_user.height = safe_float(h, current_user.height)
                    if w:
                        current_user.weight = safe_float(w, current_user.weight)
                flash("个人资料已更新", "success")
            except Exception:
                flash("资料更新失败，请稍后重试", "danger")

        elif action == "change_password":
            old_pw = request.form.get("old_password", "")
            new_pw  = request.form.get("new_password", "")
            new_pw2 = request.form.get("new_password2", "")
            if not current_user.check_password(old_pw):
                flash("当前密码错误", "danger")
            elif new_pw != new_pw2:
                flash("两次新密码不一致", "danger")
            else:
                # 密码复杂度验证
                is_valid, error_msg = validate_password(new_pw)
                if not is_valid:
                    flash(f"新密码不符合要求：{error_msg}", "danger")
                else:
                    try:
                        with db_transaction():
                            current_user.set_password(new_pw)
                        flash("密码修改成功", "success")
                    except Exception:
                        flash("密码修改失败，请稍后重试", "danger")

        return redirect(url_for("auth.profile"))

    return render_template("auth/profile.html", stats=stats)
