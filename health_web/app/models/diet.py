# -*- coding: utf-8 -*-
"""
app/models/diet.py  -  饮食记录模型
"""

from datetime import datetime
from app import db


class FoodItem(db.Model):
    """食物基础信息库"""
    __tablename__ = "food_items"

    id              = db.Column(db.Integer, primary_key=True)
    name            = db.Column(db.String(128), nullable=False, unique=True)
    category        = db.Column(db.String(64), default="其他")
    calories_per_100g = db.Column(db.Float, default=0.0)
    protein_per_100g  = db.Column(db.Float, default=0.0)
    fat_per_100g      = db.Column(db.Float, default=0.0)
    carbs_per_100g    = db.Column(db.Float, default=0.0)
    fiber_per_100g    = db.Column(db.Float, default=0.0)

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "category": self.category,
            "calories_per_100g": self.calories_per_100g,
            "protein_per_100g": self.protein_per_100g,
            "fat_per_100g": self.fat_per_100g,
            "carbs_per_100g": self.carbs_per_100g,
            "fiber_per_100g": self.fiber_per_100g,
        }


class DietRecord(db.Model):
    """用户饮食记录"""
    __tablename__ = "diet_records"

    id          = db.Column(db.Integer,  primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    food_name   = db.Column(db.String(128), nullable=False)
    food_item_id= db.Column(db.Integer, db.ForeignKey("food_items.id"), nullable=True)
    meal_type   = db.Column(db.String(20), default="lunch")
    portion     = db.Column(db.String(64), default="100g")
    calories    = db.Column(db.Float, default=0.0)
    protein     = db.Column(db.Float, default=0.0)
    fat         = db.Column(db.Float, default=0.0)
    carbs       = db.Column(db.Float, default=0.0)
    fiber       = db.Column(db.Float, default=0.0)
    notes       = db.Column(db.Text, default="")
    record_date = db.Column(db.Date, default=datetime.now, index=True)
    created_at  = db.Column(db.DateTime, default=datetime.now)

    food_ref    = db.relationship("FoodItem", foreign_keys=[food_item_id])

    def to_dict(self):
        return {
            "id": self.id, "food_name": self.food_name,
            "meal_type": self.meal_type, "portion": self.portion,
            "calories": round(self.calories, 1),
            "protein": round(self.protein, 1),
            "fat": round(self.fat, 1),
            "carbs": round(self.carbs, 1),
            "notes": self.notes,
            "record_date": self.record_date.strftime("%Y-%m-%d") if self.record_date else "",
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else "",
        }
