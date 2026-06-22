# -*- coding: utf-8 -*-
"""
pytest 配置文件
提供测试 fixtures
"""

import pytest
from app import create_app, db
from app.models.user import User


@pytest.fixture
def app():
    """创建测试应用实例"""
    app = create_app('testing')
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return app.test_client()


@pytest.fixture
def runner(app):
    """创建 CLI 测试运行器"""
    return app.test_cli_runner()


@pytest.fixture
def init_database(app):
    """初始化测试数据库并添加测试数据"""
    with app.app_context():
        # 创建测试用户
        user = User(
            username='testuser',
            email='test@example.com',
            nickname='测试用户'
        )
        user.set_password('Test123456!')
        db.session.add(user)
        db.session.commit()
        
        yield db
        
        db.session.remove()


class AuthActions:
    """认证操作辅助类"""
    
    def __init__(self, client):
        self._client = client
    
    def login(self, username='testuser', password='Test123456!'):
        return self._client.post(
            '/auth/login',
            data={'username': username, 'password': password}
        )
    
    def logout(self):
        return self._client.get('/auth/logout')


@pytest.fixture
def auth(client):
    """认证操作 fixture"""
    return AuthActions(client)
