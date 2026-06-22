# -*- coding: utf-8 -*-
"""
app/models/exercise.py  -  运动记录模型
"""

from datetime import datetime
from app import db


class ExerciseType(db.Model):
    """运动类型库"""
    __tablename__ = "exercise_types"

    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(64), nullable=False, unique=True)
    category = db.Column(db.String(32), default="有氧")
    met_value= db.Column(db.Float, default=4.0)
    icon     = db.Column(db.String(64), default="fa-heartbeat")

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "category": self.category,
            "met_value": self.met_value, "icon": self.icon,
        }


class ExerciseRecord(db.Model):
    """用户运动记录"""
    __tablename__ = "exercise_records"

    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    exercise_type   = db.Column(db.String(64), nullable=False)   # running/walking/cycling...
    duration        = db.Column(db.Integer, default=30)           # 分钟
    calories_burned = db.Column(db.Float, default=0.0)             # kcal
    distance        = db.Column(db.Float, default=0.0)             # km
    steps           = db.Column(db.Integer, default=0)
    heart_rate      = db.Column(db.Integer, nullable=True)         # 平均心率 bpm
    intensity       = db.Column(db.String(20), default="medium")  # low/medium/high
    notes           = db.Column(db.Text, default="")
    exercise_date   = db.Column(db.Date, default=datetime.now, index=True)
    created_at      = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id, "exercise_type": self.exercise_type,
            "duration": self.duration, "calories_burned": round(self.calories_burned, 1),
            "distance": round(self.distance, 2), "steps": self.steps,
            "heart_rate": self.heart_rate, "intensity": self.intensity,
            "notes": self.notes,
            "exercise_date": self.exercise_date.strftime("%Y-%m-%d") if self.exercise_date else "",
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else "",
        }
