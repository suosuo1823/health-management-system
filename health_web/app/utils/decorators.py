# -*- coding: utf-8 -*-
"""
app/utils/decorators.py  -  装饰器工具

包含数据库事务管理、缓存等装饰器。
"""
from functools import wraps
from contextlib import contextmanager
from flask import current_app
from app import db


@contextmanager
def db_transaction():
    """数据库事务上下文管理器。
    
    自动处理 commit 和 rollback，确保事务完整性。
    
    Usage:
        with db_transaction():
            db.session.add(user)
            # 自动 commit，异常时自动 rollback
    """
    try:
        yield
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"数据库事务回滚: {e}")
        raise


def transactional(func):
    """事务装饰器 - 自动包裹函数在数据库事务中。
    
    Usage:
        @transactional
        def create_user(data):
            user = User(**data)
            db.session.add(user)
            return user
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with db_transaction():
            return func(*args, **kwargs)
    return wrapper
