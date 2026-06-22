# -*- coding: utf-8 -*-
"""
seed_demo.py  -  数据库初始化脚本

用法: python seed_demo.py

功能:
  1. 创建所有数据库表
  2. 创建测试账号 demo/demo123456（充满数据）
  3. 创建食物库示例数据
  4. 创建运动类型示例数据

环境变量: 同 config.py，需要 HEALTH_DB_USER / HEALTH_DB_PASSWORD 等
"""

import os
import sys
from datetime import date, timedelta, datetime
from random import randint, uniform, choice
import pymysql

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from config import config_map

def _ensure_database_exists():
    """在连接目标数据库之前，先用 root 账号创建数据库（如果不存在）

    优先使用环境变量，否则尝试 config.py 中的值。
    这是唯一允许"回退读取密码"的地方，因为 seed_demo 通常在开发时本地运行。
    """
    # 先尝试直接从环境变量读（推荐方式）
    db_user = os.environ.get("HEALTH_DB_USER", "root")
    db_pass = os.environ.get("HEALTH_DB_PASSWORD", "")
    db_host = os.environ.get("HEALTH_DB_HOST", "localhost")
    db_port = int(os.environ.get("HEALTH_DB_PORT", "3306"))
    db_name = os.environ.get("HEALTH_DB_NAME", "health_db")

    # 如果环境变量没设密码，尝试从已加载的 config 读取（仅开发环境）
    if not db_pass:
        try:
            cfg = config_map["development"]()
            db_pass = cfg.MYSQL_PASSWORD
        except RuntimeError:
            pass

    try:
        conn = pymysql.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_pass,
            charset="utf8mb4",
        )
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()
        conn.close()
        print(f"  [OK] 数据库 '{db_name}' 已确认/创建")
        return True
    except Exception as e:
        print(f"  [ERROR] 无法连接 MySQL 或创建数据库: {e}")
        print("  请确认: 1) MySQL 服务已启动  2) 已设置环境变量 HEALTH_DB_PASSWORD")
        return False
from app.models.user import User
from app.models.diet import DietRecord, FoodItem
from app.models.exercise import ExerciseRecord, ExerciseType
from app.models.predict import PredictRecord


def seed_food_items():
    """预置常见食物数据"""
    foods = [
        ("米饭",      "主食",  116, 2.6,  0.3, 25.6, 0.3),
        ("馒头",      "主食",  209, 7.0,  1.1, 41.5, 1.3),
        ("面条",      "主食",  285, 8.3,  0.8, 59.5, 2.2),
        ("全麦面包",  "主食",  247, 13,   4.2, 41,   7.0),
        ("鸡胸肉",    "肉类",  133, 31,   1.2, 0.0,  0.0),
        ("牛肉",      "肉类",  250, 26,   15,  0.0,  0.0),
        ("猪肉",      "肉类",  395, 14,   37,  0.0,  0.0),
        ("鱼肉",      "肉类",  90,  20,   1.0, 0.0,  0.0),
        ("鸡蛋",      "肉类",  144, 13,   9.5, 1.1,  0.0),
        ("西兰花",    "蔬菜",  34,  2.8,  0.4, 6.6,  2.6),
        ("菠菜",      "蔬菜",  23,  2.9,  0.4, 3.6,  2.2),
        ("番茄",      "蔬菜",  18,  0.9,  0.2, 3.9,  1.2),
        ("胡萝卜",    "蔬菜",  32,  0.9,  0.2, 7.3,  2.8),
        ("苹果",      "水果",  52,  0.3,  0.2, 13.8, 2.4),
        ("香蕉",      "水果",  93,  1.4,  0.2, 22.8, 1.2),
        ("橙子",      "水果",  47,  0.9,  0.1, 11.8, 2.4),
        ("牛奶",      "乳制品",61,  3.0,  3.4, 4.8,  0.0),
        ("酸奶",      "乳制品",72,  3.2,  2.5, 9.5,  0.0),
        ("豆腐",      "豆制品",81,  8.1,  4.8, 1.2,  0.5),
        ("豆浆",      "豆制品",33,  2.9,  1.2, 3.0,  0.5),
        ("薯条",      "零食",  312, 3.4,  15,  41,   3.8),
        ("可乐",      "饮料",  42,  0.0,  0.0, 10.6, 0.0),
        ("橙汁",      "饮料",  45,  0.7,  0.2, 10.4, 0.2),
    ]
    for name, cat, cal, prot, fat, carbs, fiber in foods:
        if not FoodItem.query.filter_by(name=name).first():
            db.session.add(FoodItem(
                name=name, category=cat,
                calories_per_100g=cal, protein_per_100g=prot,
                fat_per_100g=fat, carbs_per_100g=carbs, fiber_per_100g=fiber
            ))
    print(f"  [OK] 食物库 ({len(foods)} 条)")


def seed_exercise_types():
    """预置运动类型"""
    types = [
        ("跑步",    "有氧", 9.8,  "fa-circle"),
        ("步行",    "有氧", 3.8,  "fa-male"),
        ("骑行",    "有氧", 7.5,  "fa-bicycle"),
        ("游泳",    "有氧", 8.0,  "fa-life-ring"),
        ("跳绳",    "有氧", 12.3, "fa-arrow-up"),
        ("瑜伽",    "柔韧", 3.0,  "fa-circle-thin"),
        ("力量训练", "力量", 6.0,  "fa-dumbbell"),
        ("篮球",    "球类", 8.0,  "fa-circle-o"),
        ("足球",    "球类", 8.0,  "fa-soccer-ball-o"),
        ("羽毛球",  "球类", 5.5,  "fa-circle-thin"),
        ("跳舞",    "有氧", 7.0,  "fa-music"),
        ("徒步",    "有氧", 5.3,  "fa-tree"),
        ("爬山",    "有氧", 8.0,  "fa-level-up"),
    ]
    for name, cat, met, icon in types:
        if not ExerciseType.query.filter_by(name=name).first():
            db.session.add(ExerciseType(name=name, category=cat, met_value=met, icon=icon))
    print(f"  [OK] 运动类型 ({len(types)} 条)")


def seed_demo_user_data(user):
    """为 demo 用户生成大量示例数据"""
    today = date.today()

    # 近30天饮食记录
    diet_entries = [
        ("米饭",     "breakfast", 200, 232, 5.2, 0.6, 51.2),
        ("鸡蛋",     "breakfast", 50,  72,  6.5, 4.8,  0.6),
        ("全麦面包", "breakfast", 80,  198, 10.4, 3.4, 32.8),
        ("鸡胸肉",   "lunch",    150, 200, 46.5, 1.8,  0.0),
        ("西兰花",   "lunch",    150,  51,  4.2,  0.6,  9.9),
        ("米饭",     "lunch",    200, 232, 5.2,  0.6, 51.2),
        ("牛肉",     "dinner",   120, 300, 31.2, 18.0,  0.0),
        ("菠菜",     "dinner",   100,  23,  2.9,  0.4,  3.6),
        ("苹果",     "snack",    150,  78,  0.5,  0.3, 20.7),
        ("酸奶",     "snack",    200, 144,  6.4,  5.0, 19.0),
        ("香蕉",     "snack",    100,  93,  1.4,  0.2, 22.8),
        ("豆腐",     "lunch",    200, 162, 16.2,  9.6,  2.4),
    ]

    for day_offset in range(30):
        record_date = today - timedelta(days=29 - day_offset)
        num_meals = randint(3, 5)
        used = []
        for _ in range(num_meals):
            food = choice([e for e in diet_entries if e[0] not in used])
            used.append(food[0])
            amount = uniform(0.8, 1.5)
            db.session.add(DietRecord(
                user_id=user.id,
                food_name=food[0],
                meal_type=food[1],
                portion=f"{int(food[2]*amount)}g",
                calories=food[3]*amount,
                protein=food[4]*amount,
                carbs=food[5]*amount,
                fat=food[6]*amount,
                record_date=record_date,
            ))

    # 近30天运动记录
    exercise_types_list = [
        ("running",     40, 380, 5.0, 8000,  145, "medium"),
        ("cycling",     30, 280, 8.0, 0,    135,  "medium"),
        ("swimming",    45, 360, 1.5, 0,    140,  "high"),
        ("gym",         50, 300, 0,   0,    150,  "high"),
        ("yoga",        30, 90,  0,   0,    100,  "low"),
        ("basketball",  60, 480, 0,   0,    155,  "high"),
        ("walking",     60, 200, 5.0, 9000, 110,  "low"),
        ("jumping_rope",20, 200, 2.0, 0,    160,  "high"),
    ]

    for day_offset in range(30):
        record_date = today - timedelta(days=29 - day_offset)
        if choice([True, True, False]):  # 约66%天数有运动
            ex = choice(exercise_types_list)
            db.session.add(ExerciseRecord(
                user_id=user.id,
                exercise_type=ex[0],
                exercise_date=record_date,
                duration=ex[1],
                calories_burned=ex[2],
                distance=ex[3],
                steps=ex[4],
                heart_rate=ex[5],
                intensity=ex[6],
            ))

    # 最近10次肥胖风险判定
    risk_params = [
        {"gender": 1, "age": 28, "height": 175, "weight": 72,  "fam": 0},
        {"gender": 1, "age": 30, "height": 178, "weight": 80,  "fam": 1},
        {"gender": 0, "age": 25, "height": 162, "weight": 55,  "fam": 0},
        {"gender": 1, "age": 35, "height": 170, "weight": 88,  "fam": 1},
        {"gender": 0, "age": 22, "height": 165, "weight": 60,  "fam": 0},
        {"gender": 1, "age": 40, "height": 172, "weight": 95,  "fam": 1},
        {"gender": 0, "age": 29, "height": 160, "weight": 52,  "fam": 0},
        {"gender": 1, "age": 33, "height": 180, "weight": 82,  "fam": 0},
        {"gender": 0, "age": 27, "height": 163, "weight": 62,  "fam": 1},
        {"gender": 1, "age": 31, "height": 176, "weight": 75,  "fam": 0},
    ]

    labels = ["Normal_Weight", "Normal_Weight", "Insufficient_Weight",
              "Overweight_Level_I", "Normal_Weight", "Obesity_Type_I",
              "Insufficient_Weight", "Overweight_Level_I", "Overweight_Level_I", "Normal_Weight"]

    for i, (rp, label) in enumerate(zip(risk_params, labels)):
        bmi = rp["weight"] / ((rp["height"] / 100) ** 2)
        days_ago = 9 - i
        dt = datetime.combine(today - timedelta(days=days_ago), datetime.min.time())
        db.session.add(PredictRecord(
            user_id=user.id,
            gender=rp["gender"],
            age=rp["age"],
            height=rp["height"],
            weight=rp["weight"],
            bmi=round(bmi, 1),
            family_history=rp["fam"],
            high_calorie_food=randint(0, 3),
            vegetable_frequency=randint(1, 3),
            main_meals=3,
            snacking=randint(0, 2),
            water_consumption=randint(1, 3),
            calorie_monitoring=choice([0, 1]),
            physical_activity=randint(0, 3),
            screen_time=randint(0, 3),
            alcohol=randint(0, 2),
            transportation=randint(0, 4),
            smoking=0,
            predicted_level=label,
            confidence=round(uniform(0.72, 0.96), 3),
            risk_level="normal" if "Normal" in label or "Insufficient" in label else "high",
            created_at=dt,
        ))

    print(f"  [OK] demo 用户数据已生成")


def main():
    print("=" * 55)
    print("  健康管理平台 - 数据库初始化")
    print("=" * 55)

    app = create_app("development")

    with app.app_context():
        print("\n[0/4] 检查并创建数据库...")
        if not _ensure_database_exists():
            return

        print("\n[1/4] 创建数据库表...")
        db.create_all()
        print("  [OK] 表创建完成")

        print("\n[2/4] 创建预置数据（食物库/运动类型）...")
        seed_food_items()
        seed_exercise_types()
        db.session.commit()

        print("\n[3/4] 创建测试账号...")
        demo_user = User.query.filter_by(username="demo").first()
        if demo_user:
            print("  [SKIP] demo 账号已存在，跳过创建")
        else:
            demo_user = User(
                username="demo",
                email="demo@example.com",
                nickname="健康达人",
                realname="张三",
                gender="male",
                age=28,
                height=175.0,
                weight=72.0,
                is_demo=True,
            )
            demo_user.set_password("demo123456")
            db.session.add(demo_user)
            db.session.commit()
            print("  [OK] demo 账号已创建: demo / demo123456")

        print("\n[4/4] 为 demo 用户生成示例数据...")
        if demo_user:
            # 先清除旧数据
            DietRecord.query.filter_by(user_id=demo_user.id).delete()
            ExerciseRecord.query.filter_by(user_id=demo_user.id).delete()
            PredictRecord.query.filter_by(user_id=demo_user.id).delete()
            db.session.commit()
            seed_demo_user_data(demo_user)
            db.session.commit()

        print("\n" + "=" * 55)
        print("  初始化完成！")
        print("  启动服务: python run.py")
        print("  测试账号: demo / demo123456")
        print("=" * 55)


if __name__ == "__main__":
    main()
