# -*- coding: utf-8 -*-
"""
工具函数单元测试
"""

import pytest
from datetime import date, datetime
from app.utils import (
    safe_float, safe_int, parse_date, 
    validate_password, truncate_string
)


class TestSafeFloat:
    """safe_float 函数测试"""
    
    def test_valid_float_string(self):
        """测试有效的浮点数字符串"""
        assert safe_float("123.45", 0.0) == 123.45
        assert safe_float("0", 0.0) == 0.0
        assert safe_float("-10.5", 0.0) == -10.5
    
    def test_invalid_string_returns_default(self):
        """测试无效字符串返回默认值"""
        assert safe_float("abc", 0.0) == 0.0
        assert safe_float("", 5.0) == 5.0
        assert safe_float(None, 10.0) == 10.0
    
    def test_empty_string_returns_default(self):
        """测试空字符串返回默认值"""
        assert safe_float("", 100.0) == 100.0
    
    def test_whitespace_string(self):
        """测试带空格的字符串"""
        assert safe_float("  123.45  ", 0.0) == 123.45


class TestSafeInt:
    """safe_int 函数测试"""
    
    def test_valid_int_string(self):
        """测试有效的整数字符串"""
        assert safe_int("123", 0) == 123
        assert safe_int("0", 0) == 0
        assert safe_int("-10", 0) == -10
    
    def test_invalid_string_returns_default(self):
        """测试无效字符串返回默认值"""
        assert safe_int("abc", 0) == 0
        assert safe_int("12.34", 0) == 0  # 小数不能转int
        assert safe_int(None, 5) == 5


class TestParseDate:
    """parse_date 函数测试"""
    
    def test_valid_date_string(self):
        """测试有效的日期字符串"""
        result = parse_date("2024-03-15", date.today())
        assert result == date(2024, 3, 15)
    
    def test_invalid_date_returns_default(self):
        """测试无效日期返回默认值"""
        default = date(2024, 1, 1)
        assert parse_date("invalid", default) == default
        assert parse_date("2024-13-45", default) == default  # 无效日期
        assert parse_date("", default) == default
    
    def test_none_returns_default(self):
        """测试 None 返回默认值"""
        default = date.today()
        assert parse_date(None, default) == default


class TestValidatePassword:
    """validate_password 函数测试"""
    
    def test_valid_password(self):
        """测试有效密码"""
        is_valid, msg = validate_password("Test123456!")
        assert is_valid is True
        assert msg == ""
    
    def test_too_short(self):
        """测试密码太短"""
        is_valid, msg = validate_password("Test1!")
        assert is_valid is False
        assert "至少8位" in msg
    
    def test_no_uppercase(self):
        """测试缺少大写字母"""
        is_valid, msg = validate_password("test123456!")
        assert is_valid is False
        assert "大写字母" in msg
    
    def test_no_lowercase(self):
        """测试缺少小写字母"""
        is_valid, msg = validate_password("TEST123456!")
        assert is_valid is False
        assert "小写字母" in msg
    
    def test_no_digit(self):
        """测试缺少数字"""
        is_valid, msg = validate_password("TestPassword!")
        assert is_valid is False
        assert "数字" in msg


class TestTruncateString:
    """truncate_string 函数测试"""
    
    def test_short_string_unchanged(self):
        """测试短字符串保持不变"""
        assert truncate_string("hello", 10) == "hello"
    
    def test_long_string_truncated(self):
        """测试长字符串被截断"""
        assert truncate_string("hello world", 5) == "hello..."
    
    def test_exact_length(self):
        """测试刚好等于长度的字符串"""
        assert truncate_string("hello", 5) == "hello"
