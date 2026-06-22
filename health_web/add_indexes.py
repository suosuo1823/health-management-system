# -*- coding: utf-8 -*-
"""
add_indexes.py  -  数据库索引迁移脚本

功能：为高频查询列添加索引，加速仪表盘/分析页面响应。

适用表及列：
  - exercise_records: exercise_type, intensity, created_at
  - diet_records:     meal_type
  - health_records:   user_id + record_date（联合，已隐含）
  - predict_records:  user_id + created_at（联合，已隐含）

用法（确保先设置环境变量）：
    python add_indexes.py

    # 或指定数据库配置：
    HEALTH_DB_USER=root HEALTH_DB_PASSWORD=xxx python add_indexes.py
"""

import os
import sys
import pymysql

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config_map


def _get_db_config():
    """从环境变量读取，失败则从 config.py 回退（仅开发）"""
    db_user = os.environ.get("HEALTH_DB_USER", "root")
    db_pass = os.environ.get("HEALTH_DB_PASSWORD", "")
    db_host = os.environ.get("HEALTH_DB_HOST", "localhost")
    db_port = int(os.environ.get("HEALTH_DB_PORT", "3306"))
    db_name = os.environ.get("HEALTH_DB_NAME", "health_db")

    if not db_pass:
        try:
            cfg = config_map["development"]()
            db_pass = cfg.MYSQL_PASSWORD
        except RuntimeError:
            pass
    return db_host, db_port, db_user, db_pass, db_name


def add_indexes():
    host, port, user, pw, dbname = _get_db_config()

    # 要执行的索引创建语句（IF NOT EXISTS 防重复）
    indexes = [
        # exercise_records 索引
        (
            "exercise_records",
            "ix_exercise_type",
            "exercise_type",
            "CREATE INDEX ix_exercise_type ON exercise_records(exercise_type)",
        ),
        (
            "exercise_records",
            "ix_exercise_intensity",
            "intensity",
            "CREATE INDEX ix_exercise_intensity ON exercise_records(intensity)",
        ),
        (
            "exercise_records",
            "ix_exercise_created_at",
            "created_at",
            "CREATE INDEX ix_exercise_created_at ON exercise_records(created_at)",
        ),
        # diet_records 索引
        (
            "diet_records",
            "ix_meal_type",
            "meal_type",
            "CREATE INDEX ix_meal_type ON diet_records(meal_type)",
        ),
    ]

    try:
        conn = pymysql.connect(
            host=host, port=port, user=user, password=pw,
            database=dbname, charset="utf8mb4",
        )
    except Exception as e:
        print(f"[ERROR] 无法连接数据库: {e}")
        print("  请确认：1) MySQL 运行中  2) 环境变量已设置")
        return

    print("=" * 55)
    print("  数据库索引迁移")
    print("=" * 55)

    with conn.cursor() as cur:
        for table, name, col, sql in indexes:
            try:
                cur.execute(f"CREATE INDEX `{name}` ON {table}({col})")
                conn.commit()
                print(f"  [OK] {table}.{col} 索引已创建")
            except pymysql.err.OperationalError as e:
                if "Duplicate key name" in str(e) or "Duplicate index" in str(e):
                    print(f"  [--] {table}.{col} 索引已存在，跳过")
                else:
                    print(f"  [WARN] {table}.{col}: {e}")
            except Exception as e:
                print(f"  [ERROR] {table}.{col}: {e}")

    conn.close()
    print("\n  迁移完成！")
    print("=" * 55)


if __name__ == "__main__":
    add_indexes()
