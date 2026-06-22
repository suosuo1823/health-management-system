# -*- coding: utf-8 -*-
"""
train_web_model.py - 快速训练Web端用全特征模型

不生成图表，只训练模型并保存，供Web端使用
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib

# 配置
RANDOM_STATE = 42
TEST_SIZE = 0.2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "obesity_level.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  训练Web端全特征模型")
print("=" * 60)

# 加载数据
df_raw = pd.read_csv(DATA_PATH, index_col='id')
print(f"\n数据集形状：{df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")

# 数据预处理
df = df_raw.copy()
df['0be1dad'] = df['0be1dad'].str.strip()
df['0be1dad'] = df['0be1dad'].replace({'0rmal_Weight': 'Normal_Weight',
                                        'Ormal_Weight': 'Normal_Weight'})

# 特征编码
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

caec_map = {'no': 0, '0': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
df['CAEC'] = df['CAEC'].map(caec_map)
df['CAEC'] = df['CAEC'].fillna(df['CAEC'].mode()[0])

calc_map = {'no': 0, '0': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
df['CALC'] = df['CALC'].map(calc_map)
df['CALC'] = df['CALC'].fillna(df['CALC'].mode()[0])

# 删除 MTRANS 列
if 'MTRANS' in df.columns:
    df = df.drop(columns=['MTRANS'])
    print("  MTRANS 列已删除")

# 目标变量编码
le = LabelEncoder()
y = le.fit_transform(df['0be1dad'])
class_names = le.classes_
print(f"\n类别映射：")
for i, cls in enumerate(class_names):
    print(f"    {i} → {cls}")

# 构建特征矩阵
feature_cols = [c for c in df.columns if c != '0be1dad']
X = df[feature_cols].copy()

# 确保所有列都是数值型
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())
    X[col] = X[col].fillna(X[col].median())

print(f"\n特征矩阵形状：{X.shape}")
print(f"特征列表：{list(X.columns)}")

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n训练集：{X_train.shape[0]} 样本  |  测试集：{X_test.shape[0]} 样本")

# 特征标准化
scaler = StandardScaler()
continuous_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
print(f"\n连续型特征（标准化）：{continuous_cols}")

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

# 使用最优超参数训练模型
best_params = {
    'subsample': 0.7,
    'reg_lambda': 10,
    'reg_alpha': 0.01,
    'n_estimators': 100,
    'min_child_weight': 1,
    'max_depth': 6,
    'learning_rate': 0.2,
    'gamma': 0.1,
    'colsample_bytree': 0.7
}

print("\n" + "-" * 60)
print("训练 XGBoost 模型...")
print("-" * 60)

model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# 评估
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
try:
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
except:
    auc = float('nan')

print(f"\n模型性能：")
print(f"  特征数：{X.shape[1]} 个")
print(f"  测试集准确率：{acc:.4f} ({acc*100:.2f}%)")
print(f"  F1(加权)：{f1:.4f}")
print(f"  AUC(OVR)：{auc:.4f}")

# 保存模型
model_path = os.path.join(OUTPUT_DIR, 'best_model_tuned.joblib')
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.joblib')
le_path = os.path.join(OUTPUT_DIR, 'label_encoder.joblib')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(le, le_path)

print(f"\n模型已保存：")
print(f"  → {model_path}")
print(f"  → {scaler_path}")
print(f"  → {le_path}")

# 保存特征信息
feature_info = {
    'all_features': list(X.columns),
    'selected_features': list(X.columns),  # Web端用全特征
    'continuous_cols': continuous_cols,
    'best_model': 'XGBoost (XGB)',
    'best_params': best_params,
    'test_accuracy': float(acc),
    'test_f1': float(f1),
    'test_auc': float(auc) if not np.isnan(auc) else None,
    'class_names': list(class_names)
}

info_path = os.path.join(OUTPUT_DIR, 'feature_info.json')
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(feature_info, f, ensure_ascii=False, indent=2)

print(f"  → {info_path}")

# 保存预处理后数据样本
X_train_scaled.to_csv(os.path.join(OUTPUT_DIR, 'X_train_preprocessed.csv'), index=False)
X_test_scaled.to_csv(os.path.join(OUTPUT_DIR, 'X_test_preprocessed.csv'), index=False)
print(f"  → X_train_preprocessed.csv")
print(f"  → X_test_preprocessed.csv")

print("\n" + "=" * 60)
print("  Web端模型训练完成！")
print("=" * 60)
