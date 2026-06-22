# -*- coding: utf-8 -*-
"""
app/models/recipe.py  -  菜谱数据模型

改动说明：
  - nutrients 关系改为 lazy="select"（兼容 joinedload，不再用 dynamic）
  - 删除 calories/protein/fat/carbs property（会产生 N+1，改在路由层批量查询）
"""

from datetime import datetime
from app import db


class Recipe(db.Model):
    """菜谱主表"""
    __tablename__ = "recipes"

    id                = db.Column(db.Integer, primary_key=True)
    title             = db.Column(db.String(200), nullable=False, index=True)
    cooking_method    = db.Column(db.String(100))
    taste             = db.Column(db.String(100))
    instructions      = db.Column(db.Text)
    suitable          = db.Column(db.Text)
    mouthfeel         = db.Column(db.Text)
    category          = db.Column(db.Text)
    main_ingredient   = db.Column(db.Text)
    sub_ingredient    = db.Column(db.Text)
    seasoning         = db.Column(db.Text)
    recommend_score   = db.Column(db.Integer, default=0)
    spicy_score       = db.Column(db.Integer, default=0)
    nutrition_score   = db.Column(db.Integer, default=0)
    difficulty_score  = db.Column(db.Integer, default=0)
    time_score        = db.Column(db.Integer, default=0)
    diet_score        = db.Column(db.Integer, default=0)
    seasoning_score   = db.Column(db.Integer, default=0)
    created_at        = db.Column(db.DateTime, default=datetime.now)

    # lazy="select"：正常懒加载，与 joinedload 完全兼容
    nutrients = db.relationship(
        "RecipeNutrition", backref="recipe",
        lazy="select", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Recipe {self.id}: {self.title}>"


class RecipeNutrition(db.Model):
    """每道菜的营养成分明细（每100g）"""
    __tablename__ = "recipe_nutritions"

    id             = db.Column(db.Integer, primary_key=True)
    recipe_id      = db.Column(
        db.Integer, db.ForeignKey("recipes.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    nutrient_key   = db.Column(db.String(50), nullable=False)
    nutrient_value = db.Column(db.String(50), nullable=False)

    __table_args__ = (
        db.UniqueConstraint("recipe_id", "nutrient_key", name="uq_recipe_nutrient"),
    )

    def __repr__(self):
        return f"<RecipeNutrition {self.recipe_id}: {self.nutrient_key}={self.nutrient_value}>"
