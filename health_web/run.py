# -*- coding: utf-8 -*-
"""
run.py  -  启动入口
填完 config.py 的密码后，运行此文件即可启动服务
"""

import os
from app import create_app, db

app = create_app(os.getenv("FLASK_ENV", "development"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    print("=" * 55)
    print("  健康管理平台已启动  ->  http://127.0.0.1:5000")
    print("  测试账号: demo / demo123456")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=True)
