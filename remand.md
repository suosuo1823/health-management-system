# 🥗 轻食计划 — 基于机器学习的肥胖风险预测与健康管理系统

> 毕业设计项目 · Flask + MySQL + Bootstrap 5 · XGBoost 91.1% 准确率

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-green?logo=flask)](https://flask.palletsprojects.com)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple?logo=bootstrap)](https://getbootstrap.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## ✨ 功能亮点

| 模块 | 功能 |
|------|------|
| 🔮 **肥胖风险预测** | XGBoost 模型，准确率 91.1%，AUC=0.991，支持 SHAP 可解释分析 |
| 🥦 **饮食记录** | 搜索食物库（150+ 种）、实时营养预览、快速记录模式 |
| 🏃 **运动记录** | MET 强度分级、每日消耗计算、运动历史追踪 |
| 📊 **数据分析** | 热量/运动趋势图、BMI 追踪、导出 CSV |
| 🏅 **连续打卡** | 徽章系统（7天/30天/完美一周），激励坚持 |
| 🌙 **主题切换** | 亮色 / 暗色 / 跟随系统，一键切换 |

---

## 🧠 算法创新

- **NHC 指南约束的混合预测模型**：将 2024 版《成人肥胖食养指南》规则与 XGBoost 输出加权融合（α=0.7），在 BMI 边界区域动态调整权重，增强鲁棒性
- **SHAP 个体化归因**：每次预测输出 Top-3 风险因素与贡献值，结果可解释
- **Stacking 集成学习**：XGBoost + Random Forest + SVM → Logistic Regression，最终准确率 **91.23%**

---

## 📈 模型性能

| 指标 | 数值 |
|------|------|
| 数据集 | 20,758 样本，15 特征，7 类别 |
| 测试集准确率 | **91.09%** |
| AUC | **0.9905** |
| Stacking 集成准确率 | **91.23%** |
| 对比基线算法 | 9 种（LR / DT / RF / ET / GBT / SVM / KNN / NB） |

---

## 🛠️ 技术栈

```
后端：Flask 3 · SQLAlchemy · Flask-Login · Flask-Limiter
数据库：MySQL 8
前端：Bootstrap 5.3 · ECharts 5.4 · Jinja2
ML：XGBoost · scikit-learn · SHAP · joblib
安全：登录频率限制（10次/分钟）· 环境变量隔离 · bcrypt 密码哈希
```

---

## 🚀 快速启动

### 1. 克隆仓库

```bash
git clone https://github.com/Suosuo1823/health-management-system.git
cd health-management-system/health_web
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入数据库密码和 SECRET_KEY
```

### 4. 初始化数据库

```bash
python seed_demo.py        # 创建表 + demo 账号 + 30天示例数据
python add_indexes.py      # 添加索引（首次运行后执行一次）
```

### 5. 启动

```bash
python run.py
```

访问 `http://localhost:5000`，测试账号：`demo / demo123456`

---

## 📁 项目结构

```
health_web/
├── app/
│   ├── ml/              # 机器学习模块（predictor / constants）
│   ├── models/          # 数据库模型
│   ├── routes/          # 路由蓝图
│   ├── static/          # CSS / JS / 图片
│   └── templates/       # Jinja2 模板（13个页面）
├── migrations/          # 数据库迁移
├── tests/               # 单元测试
├── run.py               # 启动入口
├── config.py            # 配置（读取环境变量）
└── .env.example         # 环境变量模板
```

---