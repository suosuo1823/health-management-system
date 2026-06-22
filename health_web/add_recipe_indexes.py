# -*- coding: utf-8 -*-
"""
add_recipe_indexes.py  -  为菜谱表添加索引（解决搜索/筛选慢的问题）

用法: python add_recipe_indexes.py

MySQL 5.x 兼容版：使用 ALTER TABLE DROP INDEX，索引不存在时捕获异常继续执行。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db

def add_indexes():
    app = create_app("development")
    with app.app_context():
        conn = db.engine.raw_connection()
        cursor = conn.cursor()

        indexes = [
            # (索引名, 表名, 列名)
            ("idx_recipes_cooking", "recipes", "cooking_method"),
            ("idx_recipes_taste",   "recipes", "taste"),
            ("idx_recipes_category","recipes", "category"),
            ("idx_recipes_main",    "recipes", "main_ingredient"),
            ("idx_nutritions_key",  "recipe_nutritions", "nutrient_key"),
        ]

        for idx_name, table, col in indexes:
            try:
                # 先尝试删除旧索引（MySQL 8 支持 IF EXISTS，5.x 走异常）
                try:
                    cursor.execute(
                        f"ALTER TABLE `{table}` DROP INDEX `{idx_name}`"
                    )
                    conn.commit()
                    print(f"  [DEL] 旧索引 `{idx_name}` 已删除")
                except Exception:
                    pass  # 索引不存在，直接跳过
                # 创建新索引
                cursor.execute(
                    f"ALTER TABLE `{table}` ADD INDEX `{idx_name}`(`{col}`)"
                )
                conn.commit()
                print(f"  [OK] {idx_name} on {table}.{col}")
            except Exception as e:
                print(f"  [SKIP] {idx_name}: {e}")

        cursor.close()
        conn.close()
        print("\n索引创建完成！菜谱库搜索和筛选速度将显著提升。")

if __name__ == "__main__":
    add_indexes()
