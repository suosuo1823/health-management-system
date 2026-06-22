# -*- coding: utf-8 -*-
"""
migrate_recipe_columns.py
把 recipes 表里可能超长的 VARCHAR 列改成 TEXT，避免导入时 Data too long 错误。
用法: python migrate_recipe_columns.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db

ALTERS = [
    "ALTER TABLE `recipes` MODIFY COLUMN `cooking_method` VARCHAR(100)",
    "ALTER TABLE `recipes` MODIFY COLUMN `taste`          VARCHAR(100)",
    "ALTER TABLE `recipes` MODIFY COLUMN `suitable`       TEXT",
    "ALTER TABLE `recipes` MODIFY COLUMN `mouthfeel`      TEXT",
    "ALTER TABLE `recipes` MODIFY COLUMN `category`       TEXT",
    "ALTER TABLE `recipes` MODIFY COLUMN `main_ingredient` TEXT",
    "ALTER TABLE `recipes` MODIFY COLUMN `sub_ingredient`  TEXT",
    "ALTER TABLE `recipes` MODIFY COLUMN `seasoning`       TEXT",
]

def main():
    app = create_app("development")
    with app.app_context():
        conn = db.engine.raw_connection()
        cursor = conn.cursor()
        for sql in ALTERS:
            try:
                cursor.execute(sql)
                conn.commit()
                col = sql.split("COLUMN")[1].strip().split()[0]
                print(f"  [OK] {col}")
            except Exception as e:
                print(f"  [SKIP] {e}")
        cursor.close()
        conn.close()
        print("\n列类型迁移完成，可以重新运行 seed_recipes.py 了。")

if __name__ == "__main__":
    main()
