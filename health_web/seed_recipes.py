# -*- coding: utf-8 -*-
"""
seed_recipes.py  -  导入菜谱数据

用法: python seed_recipes.py

功能:
  1. 读取 merged_caipu.csv
  2. 解析嵌套 JSON 营养数据
  3. 写入 MySQL recipes + recipe_nutritions 表
  4. 静默跳过已存在的记录（按菜名判断）
"""
import os
import sys
import json
import re
import pandas as pd
import pymysql

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models.recipe import Recipe, RecipeNutrition
from config import config_map


def _ensure_database():
    """确保数据库存在"""
    cfg = config_map["development"]()
    try:
        conn = pymysql.connect(
            host=cfg.MYSQL_HOST, port=cfg.MYSQL_PORT,
            user=cfg.MYSQL_USER, password=cfg.MYSQL_PASSWORD,
            charset="utf8mb4",
        )
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{cfg.MYSQL_DB}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()
        conn.close()
        print(f"  [OK] 数据库 '{cfg.MYSQL_DB}' 已确认")
        return True
    except Exception as e:
        print(f"  [ERROR] MySQL 连接失败: {e}")
        return False


def _safe_val(v):
    """将 NaN/None 转为 None，其余转字符串"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s if s else None


def _parse_nutrition(json_str):
    """
    解析营养成分 JSON，返回 {key: value_str} 字典。
    JSON 形如: {"1": {"yingyang_chengfen.key": "能量", "yingyang_chengfen.value": "268.26千卡"}, ...}
    """
    if not json_str or (isinstance(json_str, float) and pd.isna(json_str)):
        return {}
    try:
        obj = json.loads(json_str)
        result = {}
        for item in obj.values():
            key = item.get("yingyang_chengfen.key", "")
            val = item.get("yingyang_chengfen.value", "")
            if key and val:
                result[key] = val
        return result
    except Exception:
        return {}


def main():
    print("=" * 55)
    print("  菜谱数据导入")
    print("=" * 55)

    app = create_app("development")
    with app.app_context():
        if not _ensure_database():
            return

        print("\n[1/4] 创建表结构...")
        db.create_all()
        print("  [OK] 表结构就绪")

        print("\n[2/4] 读取 CSV 文件...")
        csv_path = os.path.join(os.path.dirname(__file__), "merged_caipu.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "..", "merged_caipu.csv")
        if not os.path.exists(csv_path):
            print(f"  [ERROR] 找不到 merged_caipu.csv")
            return

        df = pd.read_csv(csv_path, encoding="utf-8")
        print(f"  [OK] 读取到 {len(df)} 条记录")

        print("\n[3/4] 导入菜谱数据（静默去重）...")
        imported = skipped = 0

        for _, row in df.iterrows():
            title = _safe_val(row.get("caipu.title"))
            if not title:
                skipped += 1
                continue

            # 跳过已存在的同名菜谱
            if Recipe.query.filter_by(title=title).first():
                skipped += 1
                continue

            recipe = Recipe(
                title=title,
                cooking_method=_safe_val(row.get("caipu.gongyi")),
                taste=_safe_val(row.get("caipu.kouwei")),
                instructions=_safe_val(row.get("caipu.zuofa")),
                suitable=_safe_val(row.get("caipu.shiyong")),
                mouthfeel=_safe_val(row.get("caipu.kougan")),
                category=_safe_val(row.get("caipu.leibie")),
                main_ingredient=_safe_val(row.get("caipu.zhuliao")),
                sub_ingredient=_safe_val(row.get("caipu.fuliao")),
                seasoning=_safe_val(row.get("caipu.tiaoliao")),
                recommend_score=int(row.get("caipu.tuijian_zhishu") or 0),
                spicy_score=int(row.get("caipu.mala_zhishu") or 0),
                nutrition_score=int(row.get("caipu.yingyang_zhishu") or 0),
                difficulty_score=int(row.get("caipu.nanyi_zhishu") or 0),
                time_score=int(row.get("caipu.shijian_zhishu") or 0),
                diet_score=int(row.get("caipu.jianfei_zhishu") or 0),
                seasoning_score=int(row.get("caipu.yangyan_zhishu") or 0),
            )
            db.session.add(recipe)
            db.session.flush()  # 获取 recipe.id

            # 解析营养成分
            nut_json = row.get("caipu.caipu_x_yingyang_chengfen_id", "")
            nutrients = _parse_nutrition(nut_json)
            for n_key, n_val in nutrients.items():
                db.session.add(RecipeNutrition(
                    recipe_id=recipe.id,
                    nutrient_key=n_key,
                    nutrient_value=n_val,
                ))

            imported += 1

        db.session.commit()
        print(f"  [OK] 新增导入: {imported} 条  |  跳过（已存在）: {skipped} 条")

        print("\n[4/4] 汇总统计...")
        total = Recipe.query.count()
        # 统计有能量数据的菜谱数（关联营养表）
        with_cal = db.session.query(Recipe.id).join(
            RecipeNutrition,
            (RecipeNutrition.recipe_id == Recipe.id) &
            (RecipeNutrition.nutrient_key == "能量")
        ).count()
        print(f"  总菜谱: {total} 条  |  含热量数据: {with_cal} 条")

        print("\n" + "=" * 55)
        print("  导入完成！")
        print("=" * 55)


if __name__ == "__main__":
    main()
