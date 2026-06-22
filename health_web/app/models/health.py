# -*- coding: utf-8 -*-
"""
app/models/health.py  -  健康体征记录模型
"""

from datetime import datetime
from app import db


class HealthRecord(db.Model):
    """每日健康体征快照"""
    __tablename__ = "health_records"

    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    weight          = db.Column(db.Float,   nullable=False)         # kg
    height          = db.Column(db.Float,   nullable=False)         # cm
    bmi             = db.Column(db.Float,   default=0.0)
    body_fat_pct    = db.Column(db.Float,   default=0.0)            # 体脂率 %
    muscle_mass     = db.Column(db.Float,   default=0.0)            # 肌肉量 kg
    waist_cm        = db.Column(db.Float,   default=0.0)            # 腰围 cm
    hip_cm          = db.Column(db.Float,   default=0.0)            # 臀围 cm
    systolic_bp     = db.Column(db.Integer, default=0)              # 收缩压 mmHg
    diastolic_bp    = db.Column(db.Integer, default=0)              # 舒张压 mmHg
    resting_hr      = db.Column(db.Integer, default=0)              # 静息心率 bpm
    blood_glucose   = db.Column(db.Float,   default=0.0)            # 血糖 mmol/L
    sleep_hours     = db.Column(db.Float,   default=7.0)            # 睡眠时长 h
    water_ml        = db.Column(db.Integer, default=1500)           # 饮水量 mL
    notes           = db.Column(db.Text,    default="")
    record_date     = db.Column(db.Date,    default=datetime.now, index=True)
    created_at      = db.Column(db.DateTime, default=datetime.now)

    def calc_bmi(self):
        if self.height and self.weight and self.height > 0:
            h_m = self.height / 100.0
            return round(self.weight / (h_m * h_m), 2)
        return 0.0

    def to_dict(self):
        return {
            "id":           self.id,
            "weight":       self.weight,
            "height":       self.height,
            "bmi":          self.bmi,
            "body_fat_pct": self.body_fat_pct,
            "muscle_mass":  self.muscle_mass,
            "waist_cm":     self.waist_cm,
            "hip_cm":       self.hip_cm,
            "systolic_bp":  self.systolic_bp,
            "diastolic_bp": self.diastolic_bp,
            "resting_hr":   self.resting_hr,
            "blood_glucose": self.blood_glucose,
            "sleep_hours":  self.sleep_hours,
            "water_ml":     self.water_ml,
            "notes":        self.notes,
            "record_date":  self.record_date.strftime("%Y-%m-%d") if self.record_date else "",
            "created_at":   self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else "",
        }
