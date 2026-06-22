# -*- coding: utf-8 -*-
"""
=============================================================
  config.py  -  健康管理平台配置文件
=============================================================
  重要：密码和密钥必须通过环境变量传入！
  详见根目录 .env.example（复制为 .env 后填写）
=============================================================
"""

import os
from dotenv import load_dotenv

# 优先加载 .env 文件（如果存在），同时保留已设置的环境变量
load_dotenv(override=False)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _get_required(key: str) -> str:
    """读取环境变量，若未设置则退出（防止生产环境意外无密码启动）"""
    val = os.environ.get(key, "").strip()
    if not val:
        raise RuntimeError(
            f"[ERROR] 缺少环境变量 {key}！\n"
            "请复制 .env.example 为 .env 并填写配置，"
            "或运行前先 export KEY=value"
        )
    return val


class Config:
    # ── 数据库（从环境变量读取，永不硬编码）──────────────────────────────
    MYSQL_USER     = _get_required("HEALTH_DB_USER")
    MYSQL_PASSWORD = _get_required("HEALTH_DB_PASSWORD")
    MYSQL_HOST     = os.environ.get("HEALTH_DB_HOST", "localhost")
    MYSQL_PORT     = int(os.environ.get("HEALTH_DB_PORT", "3306"))
    MYSQL_DB       = os.environ.get("HEALTH_DB_NAME", "health_db")

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_recycle": 280,
        "pool_pre_ping": True,
    }

    # ── Session 加密密钥（必须从环境变量读取）────────────────────────────
    SECRET_KEY = _get_required("HEALTH_SECRET_KEY")

    # 分页
    RECORDS_PER_PAGE = 10

    # 上传
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

    # ── 机器学习模型路径 ─────────────────────────────────────────────────
    # 模型文件由 obesity_analysis.py 训练后保存在 analysis_output/ 目录
    # 文件名必须与 obesity_analysis.py 第 1105 行 model_save_path 一致
    ML_MODEL_PATH = os.path.join(
        BASE_DIR, "..", "analysis_output", "best_model_tuned.joblib"
    )
    ML_SCALER_PATH = os.path.join(
        BASE_DIR, "..", "analysis_output", "scaler.joblib"
    )


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    """测试环境配置"""
    DEBUG = True
    TESTING = True
    
    # 使用 SQLite 内存数据库进行测试
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    
    # 测试用的密钥（不需要从环境变量读取）
    SECRET_KEY = "test-secret-key-for-testing-only"
    
    # 禁用 CSRF 保护（方便测试）
    WTF_CSRF_ENABLED = False
    
    # 禁用频率限制
    RATELIMIT_ENABLED = False
    
    # 使用简单的密码用于测试
    MYSQL_USER = "test"
    MYSQL_PASSWORD = "test"


config_map = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
    "testing":     TestingConfig,
    "default":     DevelopmentConfig,
}
