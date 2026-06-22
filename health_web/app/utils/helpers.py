# -*- coding: utf-8 -*-
"""
app/utils/helpers.py  -  通用工具函数

包含安全类型转换、日期解析、验证等常用功能。
"""
from datetime import datetime, date
from typing import Optional, Union
import re


def safe_float(value: Union[str, int, float, None], default: float = 0.0) -> float:
    """安全地将值转换为浮点数。
    
    Args:
        value: 待转换的值
        default: 转换失败时的默认值
        
    Returns:
        转换后的浮点数，失败返回 default
        
    Examples:
        >>> safe_float("123.45")
        123.45
        >>> safe_float("abc", 0.0)
        0.0
        >>> safe_float(None, 10.0)
        10.0
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Union[str, int, float, None], default: int = 0) -> int:
    """安全地将值转换为整数。
    
    Args:
        value: 待转换的值
        default: 转换失败时的默认值
        
    Returns:
        转换后的整数，失败返回 default
    """
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def parse_date(date_str: Optional[str], default: Optional[date] = None) -> date:
    """安全解析日期字符串。
    
    Args:
        date_str: 日期字符串，格式 "%Y-%m-%d"
        default: 解析失败时的默认值，默认为今天
        
    Returns:
        解析后的 date 对象
    """
    if not date_str:
        return default or date.today()
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return default or date.today()


def parse_datetime(datetime_str: Optional[str], 
                   default: Optional[datetime] = None) -> datetime:
    """安全解析日期时间字符串。
    
    Args:
        datetime_str: 日期时间字符串，格式 "%Y-%m-%d %H:%M"
        default: 解析失败时的默认值，默认为当前时间
        
    Returns:
        解析后的 datetime 对象
    """
    if not datetime_str:
        return default or datetime.now()
    try:
        return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return default or datetime.now()


def validate_password(password: str) -> tuple[bool, str]:
    """验证密码复杂度。
    
    检查规则：
    - 至少8位
    - 包含大写字母
    - 包含小写字母
    - 包含数字
    
    Args:
        password: 待验证的密码
        
    Returns:
        (是否通过, 错误信息)
        
    Examples:
        >>> validate_password("Hello123")
        (True, "")
        >>> validate_password("hello")
        (False, "密码至少8位")
    """
    if len(password) < 8:
        return False, "密码至少8位"
    if not re.search(r'[A-Z]', password):
        return False, "需包含大写字母"
    if not re.search(r'[a-z]', password):
        return False, "需包含小写字母"
    if not re.search(r'\d', password):
        return False, "需包含数字"
    return True, ""


def truncate_string(text: Optional[str], max_length: int = 100, 
                    suffix: str = "...") -> str:
    """截断字符串到指定长度。
    
    Args:
        text: 原始字符串
        max_length: 最大长度
        suffix: 截断后添加的后缀
        
    Returns:
        截断后的字符串
    """
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_number(value: Optional[float], decimals: int = 1, 
                  default: str = "-") -> str:
    """格式化数字显示。
    
    Args:
        value: 数值
        decimals: 小数位数
        default: 空值时的默认显示
        
    Returns:
        格式化后的字符串
    """
    if value is None:
        return default
    return f"{value:.{decimals}f}"
