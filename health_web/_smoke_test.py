# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import create_app, db
from app.models.recipe import Recipe, RecipeNutrition

app = create_app("development")
with app.app_context():
    total   = db.session.query(Recipe).count()
    nut_tot = db.session.query(RecipeNutrition).count()
    print(f"recipes       : {total}")
    print(f"recipe_nutrit : {nut_tot}")

    first = db.session.query(Recipe).first()
    if not first:
        print("[WARN] recipes 表为空，请先运行: python seed_recipes.py")
        sys.exit(0)

    print(f"第1条: id={first.id}  title={first.title}")

    from app.routes.recipe import _build_nutrition_map
    nm = _build_nutrition_map([first.id])
    nd = nm.get(first.id, {})
    print(f"  能量   = {nd.get('能量')}")
    print(f"  蛋白质 = {nd.get('蛋白质')}")
    print(f"  脂肪   = {nd.get('脂肪')}")

    # 模拟 api/list 的一次查询
    from sqlalchemy import func as sql_func
    ids = [r.id for r in db.session.query(Recipe).limit(12).all()]
    nm2 = _build_nutrition_map(ids)
    print(f"\napi/list 12 条批量营养 map 键数: {sum(len(v) for v in nm2.values())}")
    print("PASS: 所有查询正常")
