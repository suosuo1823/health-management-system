# -*- coding: utf-8 -*-
"""
认证模块单元测试
"""

import pytest
from app.models.user import User


class TestAuth:
    """认证相关测试"""
    
    def test_register_page(self, client):
        """测试注册页面可访问"""
        response = client.get('/auth/register')
        assert response.status_code == 200
        assert b'\xe6\xb3\xa8\xe5\x86\x8c' in response.data  # "注册"
    
    def test_login_page(self, client):
        """测试登录页面可访问"""
        response = client.get('/auth/login')
        assert response.status_code == 200
        assert b'\xe7\x99\xbb\xe5\xbd\x95' in response.data  # "登录"
    
    def test_register_success(self, client, app):
        """测试成功注册"""
        response = client.post(
            '/auth/register',
            data={
                'username': 'newuser',
                'email': 'new@example.com',
                'password': 'NewPass123!',
                'password2': 'NewPass123!',
                'nickname': '新用户'
            },
            follow_redirects=True
        )
        assert response.status_code == 200
        
        with app.app_context():
            user = User.query.filter_by(username='newuser').first()
            assert user is not None
            assert user.email == 'new@example.com'
    
    def test_register_password_mismatch(self, client):
        """测试密码不一致时注册失败"""
        response = client.post(
            '/auth/register',
            data={
                'username': 'newuser',
                'email': 'new@example.com',
                'password': 'Pass123!',
                'password2': 'Different123!',
                'nickname': '新用户'
            }
        )
        assert b'\xe4\xb8\xa4\xe6\xac\xa1\xe8\xbe\x93\xe5\x85\xa5\xe7\x9a\x84\xe5\xaf\x86\xe7\xa0\x81\xe4\xb8\x8d\xe4\xb8\x80\xe8\x87\xb4' in response.data
    
    def test_register_weak_password(self, client):
        """测试弱密码注册失败"""
        response = client.post(
            '/auth/register',
            data={
                'username': 'newuser',
                'email': 'new@example.com',
                'password': '123',
                'password2': '123',
                'nickname': '新用户'
            }
        )
        assert response.status_code == 200
        # 应该显示密码不符合要求的提示
    
    def test_register_duplicate_username(self, client, init_database):
        """测试重复用户名注册失败"""
        response = client.post(
            '/auth/register',
            data={
                'username': 'testuser',  # 已存在的用户名
                'email': 'another@example.com',
                'password': 'Pass123!',
                'password2': 'Pass123!',
                'nickname': '另一个用户'
            }
        )
        assert b'\xe7\x94\xa8\xe6\x88\xb7\xe5\x90\x8d\xe5\xb7\xb2\xe5\xad\x98\xe5\x9c\xa8' in response.data
    
    def test_login_success(self, client, init_database, auth):
        """测试成功登录"""
        response = auth.login()
        assert response.status_code == 200
    
    def test_login_wrong_password(self, client, init_database):
        """测试错误密码登录失败"""
        response = client.post(
            '/auth/login',
            data={'username': 'testuser', 'password': 'wrongpassword'}
        )
        assert b'\xe7\x94\xa8\xe6\x88\xb7\xe5\x90\x8d\xe6\x88\x96\xe5\xaf\x86\xe7\xa0\x81\xe9\x94\x99\xe8\xaf\xaf' in response.data
    
    def test_logout(self, client, auth, init_database):
        """测试登出"""
        auth.login()
        response = auth.logout()
        assert response.status_code == 200
    
    def test_profile_access_requires_login(self, client):
        """测试未登录时访问个人资料被重定向"""
        response = client.get('/auth/profile', follow_redirects=True)
        assert response.status_code == 200
        # 应该被重定向到登录页
