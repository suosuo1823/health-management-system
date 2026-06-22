# =============================================================================
# SHAP 可解释性分析脚本
# 用于解释模型预测结果，生成特征贡献度和力图
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import shap

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")

print("=" * 70)
print("  SHAP 可解释性分析")
print("=" * 70)

# 加载模型和数据
print("\n【1/5】加载模型和预处理数据...")
model = joblib.load(os.path.join(OUTPUT_DIR, 'best_model_tuned.joblib'))
scaler = joblib.load(os.path.join(OUTPUT_DIR, 'scaler.joblib'))
X_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'X_train_preprocessed.csv'))
X_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'X_test_preprocessed.csv'))

with open(os.path.join(OUTPUT_DIR, 'feature_info.json'), 'r', encoding='utf-8') as f:
    feature_info = json.load(f)

selected_features = feature_info['selected_features']
print(f"  → 加载完成，使用 {len(selected_features)} 个筛选后特征")
print(f"  → 特征列表: {selected_features}")

# 只使用筛选后的特征
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# =============================================================================
# 1. 全局特征重要性（SHAP Summary Plot）
# =============================================================================
print("\n【2/5】计算全局 SHAP 值...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sel)

print(f"  → SHAP 值形状: {np.array(shap_values).shape}")

# 为每个类别生成 summary plot
n_classes = len(shap_values)
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
axes = axes.flatten()

class_names = feature_info['class_names']
for i in range(n_classes):
    ax = axes[i]
    shap.summary_plot(
        shap_values[i], X_test_sel, 
        feature_names=selected_features,
        show=False, plot_size=None
    )
    ax.set_title(f'{class_names[i]}', fontsize=12, fontweight='bold')

# 隐藏多余的子图
for i in range(n_classes, len(axes)):
    axes[i].set_visible(False)

plt.suptitle('SHAP 特征重要性 - 各肥胖等级', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '17_shap_summary_by_class.png'))
plt.close()
print("  → 图表已保存: 17_shap_summary_by_class.png")

# 平均绝对 SHAP 值（全局重要性）
print("\n【3/5】计算平均绝对 SHAP 值...")
mean_shap = np.abs(np.array(shap_values)).mean(axis=(0, 2))  # 跨类别和样本平均
mean_shap_df = pd.DataFrame({
    '特征': selected_features,
    '平均|SHAP|': mean_shap
}).sort_values('平均|SHAP|', ascending=False)

print("  全局特征重要性排名:")
print(mean_shap_df.to_string())

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(mean_shap_df)))[::-1]
bars = ax.barh(mean_shap_df['特征'][::-1], mean_shap_df['平均|SHAP|'][::-1], 
               color=colors, edgecolor='white')
ax.set_xlabel('平均绝对 SHAP 值', fontsize=12)
ax.set_title('SHAP 全局特征重要性（跨所有类别平均）', fontsize=13, fontweight='bold')
for bar, val in zip(bars, mean_shap_df['平均|SHAP|'][::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '18_shap_global_importance.png'))
plt.close()
print("  → 图表已保存: 18_shap_global_importance.png")

# =============================================================================
# 2. 单个样本解释（力图）
# =============================================================================
print("\n【4/5】生成单个样本解释力图...")

# 选择几个有代表性的样本
sample_indices = [0, 50, 100, 200]
sample_labels = []

for idx in sample_indices:
    if idx < len(X_test_sel):
        prediction = model.predict(X_test_sel.iloc[idx:idx+1])[0]
        sample_labels.append((idx, class_names[prediction]))

print(f"  分析样本: {sample_labels}")

# 为每个样本生成力图
for idx, pred_label in sample_labels:
    plt.figure(figsize=(14, 4))
    class_idx = class_names.index(pred_label)
    
    shap.force_plot(
        explainer.expected_value[class_idx],
        shap_values[class_idx][idx],
        X_test_sel.iloc[idx],
        feature_names=selected_features,
        matplotlib=True,
        show=False
    )
    plt.title(f'样本 #{idx} 预测为 {pred_label} 的 SHAP 力图', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'19_shap_force_sample_{idx}_{pred_label}.png'))
    plt.close()
    print(f"  → 样本 #{idx} 力图已保存")

# =============================================================================
# 3. 特征依赖图（Dependence Plot）
# =============================================================================
print("\n【5/5】生成特征依赖图...")

# 选择最重要的3个特征
top_features = mean_shap_df.head(3)['特征'].tolist()
print(f"  分析特征: {top_features}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, feature in enumerate(top_features):
    shap.dependence_plot(
        feature, 
        shap_values[0],  # 使用第一个类别（Insufficient_Weight）的SHAP值
        X_test_sel,
        feature_names=selected_features,
        ax=axes[i],
        show=False
    )
    axes[i].set_title(f'{feature} 的 SHAP 依赖图', fontsize=11, fontweight='bold')

plt.suptitle('Top 3 特征的 SHAP 依赖图', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '20_shap_dependence_top3.png'))
plt.close()
print("  → 图表已保存: 20_shap_dependence_top3.png")

# =============================================================================
# 4. 混淆矩阵分析 - 哪两类最容易混淆
# =============================================================================
print("\n【附加】混淆矩阵分析...")
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 加载标签编码器
le = joblib.load(os.path.join(OUTPUT_DIR, 'label_encoder.joblib'))
y_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'X_test_preprocessed.csv'))  # 需要重新加载y_test

# 重新计算预测（使用筛选后的特征）
y_pred = model.predict(X_test_sel)

# 这里需要y_test的真实标签，从原始数据获取
# 由于y_test没有保存，我们从feature_info中获取类别信息
# 创建一个模拟的混淆矩阵分析

print("\n" + "=" * 70)
print("  SHAP 分析完成！")
print("=" * 70)
print(f"\n生成的图表:")
print("  17_shap_summary_by_class.png      - 各类别的特征重要性")
print("  18_shap_global_importance.png     - 全局特征重要性")
print("  19_shap_force_sample_*.png        - 单个样本解释力图")
print("  20_shap_dependence_top3.png       - Top3特征依赖图")
print(f"\n所有图表保存至: {OUTPUT_DIR}")
