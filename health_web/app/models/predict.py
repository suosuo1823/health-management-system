# -*- coding: utf-8 -*-
"""
app/models/predict.py  -  肥胖风险预测记录
"""

import json
from datetime import datetime
from app import db


class PredictRecord(db.Model):
    """肥胖风险预测存档"""
    __tablename__ = "predict_records"

    id                  = db.Column(db.Integer,   primary_key=True)
    user_id             = db.Column(db.Integer,   db.ForeignKey("users.id"), nullable=True, index=True)
    # 基本信息
    gender              = db.Column(db.Integer,   default=1)    # 0=女, 1=男
    age                 = db.Column(db.Float,      default=25.0)
    height              = db.Column(db.Float,      default=170.0)
    weight              = db.Column(db.Float,      default=65.0)
    bmi                 = db.Column(db.Float,      default=0.0)
    family_history      = db.Column(db.Integer,   default=0)
    # 饮食习惯
    high_calorie_food   = db.Column(db.Integer,   default=1)
    vegetable_frequency = db.Column(db.Integer,   default=2)
    main_meals          = db.Column(db.Integer,   default=2)
    snacking            = db.Column(db.Integer,   default=1)
    water_consumption   = db.Column(db.Integer,   default=2)
    calorie_monitoring   = db.Column(db.Integer,   default=0)
    # 运动行为
    physical_activity   = db.Column(db.Integer,   default=1)
    screen_time         = db.Column(db.Integer,   default=1)
    alcohol             = db.Column(db.Integer,   default=0)
    transportation      = db.Column(db.Integer,   default=2)
    smoking             = db.Column(db.Integer,   default=0)
    # 预测结果
    predicted_level     = db.Column(db.String(64), default="Normal_Weight")
    confidence          = db.Column(db.Float,      default=0.0)
    result_proba        = db.Column(db.Text,      default="{}")
    risk_level          = db.Column(db.String(20), default="normal")
    created_at          = db.Column(db.DateTime,  default=datetime.now)

    def to_dict(self):
        try:
            proba_dict = json.loads(self.result_proba) if self.result_proba else {}
        except Exception:
            proba_dict = {}
        return {
            "id": self.id,
            "gender": self.gender, "age": self.age,
            "height": self.height, "weight": self.weight,
            "bmi": round(self.bmi, 1) if self.bmi else 0,
            "predicted_level": self.predicted_level,
            "proba": proba_dict,
            "confidence": round(self.confidence, 3) if self.confidence else 0,
            "risk_level": self.risk_level,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else "",
        }

    @property
    def result_label(self):
        """返回人类可读的预测等级中文名称"""
        level_map = {
            "Insufficient_Weight":    "体重不足",
            "Normal_Weight":          "正常体重",
            "Overweight_Level_I":     "超重 I 级",
            "Overweight_Level_II":    "超重 II 级",
            "Obesity_Type_I":         "肥胖 I 型",
            "Obesity_Type_II":        "肥胖 II 型",
            "Obesity_Type_III":      "肥胖 III 型",
        }
        return level_map.get(self.predicted_level, self.predicted_level)

    @property
    def risk_label(self):
        """返回中文风险等级（低/中/高）"""
        return self.result_label  # 与 result_label 兼容
