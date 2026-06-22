# 健康管家 - AI 肥胖风险分析平台

基于 Flask + MySQL + XGBoost 的完整健康管理网站。

---

## 功能概览

| 模块 | 说明 |
|------|------|
| 首页 | 未登录展示示例数据，登录跳转全功能 |
| 仪表盘 | 今日热量/运动/BMI 概览，7天趋势图 |
| 饮食记录 | 添加/查看/删除每日饮食（按日期+餐次） |
| 运动记录 | 添加/查看/删除每日运动（14种运动类型） |
| 数据分析 | ECharts 多图表：热量趋势/营养素/运动分布/收支对比 |
| 肥胖风险判定 | 基于 XGBoost 模型（准确率 91.1%）实时预测 7 类肥胖等级 |
| 个人资料 | 修改个人信息 / 修改密码 |
| 测试账号 | demo / demo123456（充满 30 天数据） |

---

## 快速开始

### 第一步：安装 MySQL 并创建数据库

> **如何安装 MySQL？**
> - Windows: 下载 [MySQL Installer](https://dev.mysql.com/downloads/installer/)
> - 或使用 phpStudy / phpEnv 等集成环境
> - 确保 MySQL 服务已启动

创建空数据库（SQLAlchemy 只建表不建库）：
```sql
CREATE DATABASE health_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 第二步：配置环境变量

复制 `.env.example` 为 `.env`，填写以下三个变量：

```bash
HEALTH_DB_USER=root
HEALTH_DB_PASSWORD=你的MySQL密码
HEALTH_SECRET_KEY=任意随机字符串（用于 session 加密）
```

> **重要**：`.env` 文件不会被提交到 Git（已加入 .gitignore），请妥善保管。

### 第三步：初始化数据库

```bash
cd c:\Users\WXS\Desktop\学校\毕设数据分析2\health_web
pip install -r requirements.txt
python seed_demo.py
```

看到以下输出表示成功：

```
=======================================================
  健康管理平台 - 数据库初始化
=======================================================
  初始化完成！
  启动服务: python run.py
  测试账号: demo / demo123456
=======================================================
```

### 第四步：启动服务

```bash
python run.py
```

浏览器打开：**http://127.0.0.1:5000**

---

## 项目结构

```
health_web/
├── config.py           # 从 .env 读取配置（无需手动修改）
├── run.py              # 启动入口
├── seed_demo.py        # 数据库初始化（含测试数据）
├── requirements.txt    # 依赖列表
└── app/
    ├── __init__.py     # Flask 应用工厂
    ├── models/         # 数据库模型
    │   ├── user.py
    │   ├── diet.py     # 饮食记录
    │   ├── exercise.py # 运动记录
    │   ├── predict.py  # 预测存档
    │   └── health.py   # 健康体征
    ├── routes/         # 路由蓝图
    │   ├── main.py     # 首页 / 仪表盘
    │   ├── auth.py     # 登录 / 注册 / 资料
    │   ├── diet.py     # 饮食记录
    │   ├── exercise.py # 运动记录
    │   ├── analysis.py # 数据分析
    │   ├── predict.py  # 风险判定
    │   └── api.py      # AJAX 接口
    ├── ml/
    │   └── predictor.py # XGBoost 预测引擎
    ├── static/
    │   ├── css/style.css  # 全局样式（绿色系 Premium UI）
    │   └── js/app.js      # 全局 JavaScript
    └── templates/      # HTML 模板
        ├── base.html
        ├── main/
        ├── auth/
        ├── diet/
        ├── exercise/
        ├── analysis/
        └── predict/
```

---

## 账号说明

| 账号 | 密码 | 说明 |
|------|------|------|
| demo | demo123456 | 测试账号，含 30 天饮食/运动/判定数据 |
| 注册新账号 | 自设 | 新用户，所有功能为空 |

---

## 技术栈

- **后端**: Flask 3 + SQLAlchemy + Flask-Login + Flask-Bcrypt
- **数据库**: MySQL 8.x
- **前端**: Bootstrap 5.3 + Font Awesome 4.7 + ECharts 5.4
- **ML 模型**: XGBoost（ obesity_analysis.py 训练，AUC 0.991）
- **图表**: ECharts 5.4（热量趋势/营养素饼图/收支对比/仪表盘等 7 种图表）

---

## 首次配置完成后完整操作流程

```
1. 安装 MySQL 并启动服务，创建 health_db 数据库
        ↓
2. 复制 .env.example 为 .env，填写数据库密码和 SECRET_KEY
        ↓
3. 运行 pip install -r requirements.txt
        ↓
4. 运行 python seed_demo.py（创建表+测试账号+示例数据）
        ↓
5. 运行 python add_indexes.py（添加数据库索引，优化查询性能）
        ↓
6. 运行 python run.py 启动服务
        ↓
7. 打开 http://127.0.0.1:5000 查看
```

---

## 常见问题

**Q: 运行时报 `ModuleNotFoundError`？**
```bash
pip install -r requirements.txt
```

**Q: MySQL 连接失败？**
- 确认 MySQL 服务已启动
- 确认 `.env` 文件中的 `HEALTH_DB_PASSWORD` 已正确填写
- 确认数据库 `health_db` 已存在（seed_demo.py 会自动创建表但不会自动建库）
- 检查 `.env` 文件格式（每行 `KEY=value`，无引号，无空格）

**Q: 模型没有加载？**
- 模型文件路径：`../analysis_output/best_model_tuned.joblib`
- 标准化器路径：`../analysis_output/scaler.joblib`
- 如果文件不存在，预测功能会自动切换为规则引擎兜底模式，保证功能可用

**Q: 图表不显示？**
- 确认浏览器网络正常（CDN 加载 ECharts）
- 检查浏览器控制台是否有报错
