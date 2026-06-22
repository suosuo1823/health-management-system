# -*- coding: utf-8 -*-
"""
app/models/user.py  -  用户模型
"""

from datetime import datetime
from flask_login import UserMixin
from app import db, bcrypt, login_manager


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id           = db.Column(db.Integer,     primary_key=True)
    username     = db.Column(db.String(64),  unique=True, nullable=False, index=True)
    email        = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    nickname     = db.Column(db.String(64),  default="")
    avatar       = db.Column(db.String(256), default="")
    realname     = db.Column(db.String(64),  default="")
    gender       = db.Column(db.String(10),  default="male")   # male / female
    age          = db.Column(db.Integer,     default=25)
    height       = db.Column(db.Float,       default=170.0)    # cm
    weight       = db.Column(db.Float,       default=65.0)     # kg
    created_at   = db.Column(db.DateTime,    default=datetime.now)
    updated_at   = db.Column(db.DateTime,    default=datetime.now, onupdate=datetime.now)
    is_demo      = db.Column(db.Boolean,     default=False)    # 测试账号标记

    # 关联
    diet_records     = db.relationship("DietRecord",     backref="user", lazy="dynamic", cascade="all, delete-orphan")
    exercise_records = db.relationship("ExerciseRecord", backref="user", lazy="dynamic", cascade="all, delete-orphan")
    health_records   = db.relationship("HealthRecord",   backref="user", lazy="dynamic", cascade="all, delete-orphan")
    predict_records  = db.relationship("PredictRecord",  backref="user", lazy="dynamic", cascade="all, delete-orphan")

    def set_password(self, raw_password):
        self.password_hash = bcrypt.generate_password_hash(raw_password).decode("utf-8")

    def check_password(self, raw_password):
        return bcrypt.check_password_hash(self.password_hash, raw_password)

    @property
    def bmi(self):
        if self.height and self.weight and self.height > 0:
            h_m = self.height / 100.0
            return round(self.weight / (h_m * h_m), 2)
        return 0.0

    def to_dict(self):
        return {
            "id":        self.id,
            "username":  self.username,
            "nickname":  self.nickname or self.username,
            "email":     self.email,
            "gender":    self.gender,
            "age":       self.age,
            "height":    self.height,
            "weight":    self.weight,
            "bmi":       self.bmi,
            "avatar":    self.avatar,
            "is_demo":   self.is_demo,
            "created_at": self.created_at.strftime("%Y-%m-%d") if self.created_at else "",
        }


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
