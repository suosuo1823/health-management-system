# -*- coding: utf-8 -*-
"""
app/__init__.py  -  Flask 应用工厂
"""

from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from config import config_map

db      = SQLAlchemy()
bcrypt  = Bcrypt()
migrate = Migrate()
csrf    = CSRFProtect()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "请先登录"
login_manager.login_message_category = "warning"

# 频率限制器（in-memory，存储在内存中）
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",   # 单进程内存存储；生产环境应换 Redis
)


def create_app(env="default"):
    app = Flask(__name__)
    app.config.from_object(config_map[env])

    # 扩展初始化
    limiter.init_app(app)
    db.init_app(app)
    bcrypt.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)

    # 注册蓝图
    from app.routes.auth     import auth_bp
    from app.routes.main      import main_bp
    from app.routes.diet      import diet_bp
    from app.routes.exercise  import exercise_bp
    from app.routes.analysis  import analysis_bp
    from app.routes.predict   import predict_bp
    from app.routes.api       import api_bp
    from app.routes.recipe    import bp as recipe_bp
    from app.routes.research  import research_bp

    app.register_blueprint(auth_bp,      url_prefix="/auth")
    app.register_blueprint(main_bp,      url_prefix="/")
    app.register_blueprint(diet_bp,      url_prefix="/diet")
    app.register_blueprint(exercise_bp, url_prefix="/exercise")
    app.register_blueprint(analysis_bp,  url_prefix="/analysis")
    app.register_blueprint(predict_bp,   url_prefix="/predict")
    app.register_blueprint(api_bp,       url_prefix="/api")
    app.register_blueprint(recipe_bp,    url_prefix="/recipes")
    app.register_blueprint(research_bp,  url_prefix="/research")

    # ── Jinja2 数值过滤器 ──────────────────────────────────────────
    @app.template_filter("round1")
    def round1_filter(v):
        """将数值格式化为小数点后1位，None/空值返回 0.0"""
        try:
            return round(float(v), 1)
        except (TypeError, ValueError):
            return 0.0

    # 错误页面
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def internal_error(e):
        return render_template("errors/500.html"), 500

    # Flask-Limiter 触发频率限制时返回友好提示（而非崩溃）
    @app.errorhandler(429)
    def rate_limit_exceeded(e):
        return render_template("errors/429.html"), 429

    return app
