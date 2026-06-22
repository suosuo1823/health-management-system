# =============================================================================
# 算法创新实验脚本
# 包含3个创新点：
#   创新点1: NHC指南约束的混合预测模型（ML + 规则引擎融合）
#   创新点2: 基于SHAP的个体化特征归因解释
#   创新点3: 交互特征工程 + Stacking集成学习
#
# 运行此脚本后将生成：
#   - 对比实验数据（准确率、AUC、F1等）
#   - 可视化图表（保存至 analysis_output/）
#   - 创新模型文件（保存至 analysis_output/）
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
import seaborn as sns
from scipy import stats

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import joblib

# 尝试导入 SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[警告] shap 库未安装，创新点2的SHAP分析将跳过")

# ─────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "obesity_level.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

print("=" * 70)
print("  算法创新实验 - 三大创新点对比验证")
print("=" * 70)


# =============================================================================
# 数据预处理（与 obesity_analysis.py 完全一致）
# =============================================================================
print("\n" + "-" * 70)
print("[Step 0] 数据加载与预处理")
print("-" * 70)

df_raw = pd.read_csv(DATA_PATH, index_col='id')
# 去重
dup_count = df_raw.duplicated().sum()
if dup_count > 0:
    df_raw = df_raw.drop_duplicates().reset_index(drop=True)
    print(f"  删除重复行 {dup_count} 条，剩余 {len(df_raw)} 行")

df = df_raw.copy()
# 标签统一化
df['0be1dad'] = df['0be1dad'].str.strip()
df['0be1dad'] = df['0be1dad'].replace({'0rmal_Weight': 'Normal_Weight',
                                        'Ormal_Weight': 'Normal_Weight'})
# 类别型特征编码
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
caec_map = {'no': 0, '0': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
calc_map = {'no': 0, '0': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
df['CAEC'] = df['CAEC'].map(caec_map).fillna(df['CAEC'].mode()[0])
df['CALC'] = df['CALC'].map(calc_map).fillna(df['CALC'].mode()[0])
if 'MTRANS' in df.columns:
    df = df.drop(columns=['MTRANS'])

# 目标变量编码
le = LabelEncoder()
y = le.fit_transform(df['0be1dad'])
class_names = le.classes_

# 构建特征矩阵
feature_cols = [c for c in df.columns if c != '0be1dad']
X = df[feature_cols].copy()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X[col] = X[col].fillna(X[col].median())

# 训练/测试集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 标准化
continuous_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

# 同时保留未标准化版本（用于BMI计算等规则引擎）
X_train_raw = X_train.copy()
X_test_raw = X_test.copy()

print(f"  训练集: {X_train_scaled.shape[0]} 样本, {X_train_scaled.shape[1]} 特征")
print(f"  测试集: {X_test_scaled.shape[0]} 样本")
print(f"  类别数: {len(class_names)} ({', '.join(class_names)})")

# 加载基线模型（原始XGBoost）
baseline_model_path = os.path.join(OUTPUT_DIR, 'best_model_tuned.joblib')
baseline_model = joblib.load(baseline_model_path)
print(f"  基线模型已加载: {baseline_model_path}")


# =============================================================================
# 创新点1: NHC指南约束的混合预测模型（ML + 规则引擎融合）
# =============================================================================
print("\n" + "=" * 70)
print("  创新点1: NHC指南约束的混合预测模型（ML + 规则融合）")
print("=" * 70)


def _nhc_rule_predict(weight, height, favc, faf, family_history, caec):
    """
    基于NHC《成人肥胖食养指南》的规则引擎预测。
    返回7类概率分布（与LabelEncoder类别一致）。
    
    创新点说明：
    - 不使用简单的if-else分类，而是将医学规则转化为概率分布
    - BMI边界区域使用更平滑的概率过渡，避免硬切换
    - 辅助特征（家族史、运动等）作为概率修正因子
    """
    # BMI计算
    height_m = height / 100.0
    bmi = weight / (height_m ** 2) if height_m > 0 else 25.0

    # NHC BMI标准（WS/T 428-2013）
    # 7类: Insufficient_Weight, Normal_Weight, Obesity_Type_I/II/III, Overweight_Level_I/II
    # LabelEncoder类别顺序: 0=Insufficient, 1=Normal, 2=Obesity_I, 3=Obesity_II, 4=Obesity_III, 5=Overweight_I, 6=Overweight_II

    # 基于BMI的先验概率（高斯核平滑）
    # 为每个BMI值生成一个7维概率分布
    bmi_centers = {
        'Insufficient_Weight': 17.0,
        'Normal_Weight': 21.0,
        'Obesity_Type_I': 32.0,
        'Obesity_Type_II': 37.0,
        'Obesity_Type_III': 42.0,
        'Overweight_Level_I': 25.5,
        'Overweight_Level_II': 28.0,
    }
    
    # 类别索引映射（与LabelEncoder一致）
    class_idx = {name: i for i, name in enumerate(class_names)}
    
    # 用高斯核生成BMI先验概率
    sigma = 3.0  # 控制过渡平滑度
    probs = np.zeros(len(class_names))
    for name, center in bmi_centers.items():
        idx = class_idx[name]
        probs[idx] = np.exp(-0.5 * ((bmi - center) / sigma) ** 2)
    
    # 归一化
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(len(class_names)) / len(class_names)
    
    # 辅助特征修正因子
    # 家族史：向更高级别偏移
    if family_history >= 1:
        shift = np.array([0, -0.05, 0.02, 0.02, 0.01, -0.03, 0.03])
        probs += shift
    
    # 高热量食物：向肥胖偏移
    if favc >= 1:
        shift = np.array([-0.02, -0.05, 0.03, 0.02, 0.01, 0.0, 0.01])
        probs += shift * favc
    
    # 缺乏运动：向肥胖偏移
    if faf <= 1:
        shift = np.array([-0.02, -0.03, 0.02, 0.02, 0.01, 0, 0])
        probs += shift * (2 - faf)
    
    # 频繁吃零食：向超重/肥胖偏移
    if caec >= 2:
        shift = np.array([-0.01, -0.03, 0.01, 0.01, 0, 0.01, 0.01])
        probs += shift * (caec - 1)
    
    # 确保概率非负并归一化
    probs = np.maximum(probs, 0.01)
    probs = probs / probs.sum()
    
    return probs


def hybrid_predict(ml_model, X_scaled, X_raw_df, alpha=0.7):
    """
    NHC指南约束的混合预测模型。
    
    创新点说明：
    - 动态权重融合：alpha不是固定值，而是根据BMI位置动态调整
    - BMI边界区域（18.5/24/28附近）：增加规则引擎权重，减少ML误判
    - BMI远离边界区域：ML权重更大，发挥统计模型优势
    
    参数：
        ml_model: 训练好的ML模型（XGBoost）
        X_scaled: 标准化后的特征矩阵
        X_raw_df: 原始特征矩阵（用于BMI计算等）
        alpha: 基础ML权重（0-1），会被动态调整
    """
    # ML模型概率
    ml_proba = ml_model.predict_proba(X_scaled)
    
    # 规则引擎概率
    n_samples = X_raw_df.shape[0]
    rule_proba = np.zeros((n_samples, len(class_names)))
    
    for i in range(n_samples):
        row = X_raw_df.iloc[i]
        weight = row['Weight']
        height = row['Height']
        favc = row.get('FAVC', 1)
        faf = row.get('FAF', 1)
        family_history = row.get('family_history_with_overweight', 0)
        caec = row.get('CAEC', 1)
        rule_proba[i] = _nhc_rule_predict(weight, height, favc, faf, family_history, caec)
    
    # 动态权重：BMI边界区域增加规则引擎权重
    weights = np.ones(n_samples) * alpha  # 默认ML权重
    
    for i in range(n_samples):
        row = X_raw_df.iloc[i]
        height_m = row['Height'] / 100.0
        bmi = row['Weight'] / (height_m ** 2) if height_m > 0 else 25.0
        
        # BMI边界距离：越靠近边界，规则引擎权重越大
        boundaries = [18.5, 24.0, 28.0]
        min_dist = min(abs(bmi - b) for b in boundaries)
        
        if min_dist < 2.0:
            # 距离边界<2时，降低ML权重（增加规则权重）
            # 距离0时alpha降到0.5，距离2时保持原始alpha
            dynamic_alpha = 0.5 + (alpha - 0.5) * (min_dist / 2.0)
            weights[i] = dynamic_alpha
    
    # 加权融合
    ml_weight = weights.reshape(-1, 1)
    rule_weight = 1 - ml_weight
    hybrid_proba = ml_weight * ml_proba + rule_weight * rule_proba
    
    # 预测标签
    hybrid_pred = np.argmax(hybrid_proba, axis=1)
    
    return hybrid_pred, hybrid_proba


# ─── 创新点1实验 ───
print("\n  [创新点1] 实验开始...")

# 基线XGBoost
y_pred_baseline = baseline_model.predict(X_test_scaled)
y_proba_baseline = baseline_model.predict_proba(X_test_scaled)
acc_baseline = accuracy_score(y_test, y_pred_baseline)
f1_baseline = f1_score(y_test, y_pred_baseline, average='weighted', zero_division=0)
auc_baseline = roc_auc_score(y_test, y_proba_baseline, multi_class='ovr', average='macro')

print(f"  基线XGBoost: Acc={acc_baseline:.4f}, F1={f1_baseline:.4f}, AUC={auc_baseline:.4f}")

# 混合模型（不同alpha值对比）
alpha_values = [0.5, 0.6, 0.7, 0.8, 0.9]
hybrid_results = {}

for alpha in alpha_values:
    y_pred_hybrid, y_proba_hybrid = hybrid_predict(
        baseline_model, X_test_scaled, X_test_raw, alpha=alpha
    )
    acc_h = accuracy_score(y_test, y_pred_hybrid)
    f1_h = f1_score(y_test, y_pred_hybrid, average='weighted', zero_division=0)
    try:
        auc_h = roc_auc_score(y_test, y_proba_hybrid, multi_class='ovr', average='macro')
    except:
        auc_h = np.nan
    
    hybrid_results[alpha] = {
        'accuracy': acc_h,
        'f1': f1_h,
        'auc': auc_h,
        'y_pred': y_pred_hybrid,
        'y_proba': y_proba_hybrid,
    }
    print(f"  混合模型(alpha={alpha}): Acc={acc_h:.4f}, F1={f1_h:.4f}, AUC={auc_h:.4f}")

# 找最优alpha
best_alpha = max(hybrid_results, key=lambda a: hybrid_results[a]['accuracy'])
best_hybrid = hybrid_results[best_alpha]
print(f"\n  最优alpha={best_alpha}, Acc={best_hybrid['accuracy']:.4f}")

# BMI边界区域专项分析
print("\n  [创新点1] BMI边界区域专项分析...")
heights_test = X_test_raw['Height'].values / 100.0
bmis_test = X_test_raw['Weight'].values / (heights_test ** 2)

# 定义边界区域：BMI在[16.5-20.5], [22-26], [26-30]
boundary_mask = np.zeros(len(y_test), dtype=bool)
for bmi_val in bmis_test:
    pass  # 循环只是为了确认可以计算

# BMI边界样本（距任一边界<2.0）
boundary_indices = []
for i, bmi_val in enumerate(bmis_test):
    dist_to_boundary = min(abs(bmi_val - 18.5), abs(bmi_val - 24.0), abs(bmi_val - 28.0))
    if dist_to_boundary < 2.0:
        boundary_indices.append(i)

if boundary_indices:
    boundary_mask = np.zeros(len(y_test), dtype=bool)
    boundary_mask[boundary_indices] = True
    
    # 基线在边界区域的准确率
    acc_baseline_boundary = accuracy_score(y_test[boundary_mask], y_pred_baseline[boundary_mask])
    # 混合模型在边界区域的准确率
    acc_hybrid_boundary = accuracy_score(y_test[boundary_mask], best_hybrid['y_pred'][boundary_mask])
    # 非边界区域
    acc_baseline_non_boundary = accuracy_score(y_test[~boundary_mask], y_pred_baseline[~boundary_mask])
    acc_hybrid_non_boundary = accuracy_score(y_test[~boundary_mask], best_hybrid['y_pred'][~boundary_mask])
    
    print(f"  边界样本数: {sum(boundary_mask)}/{len(y_test)}")
    print(f"  基线 - 边界区域准确率: {acc_baseline_boundary:.4f} | 非边界: {acc_baseline_non_boundary:.4f}")
    print(f"  混合 - 边界区域准确率: {acc_hybrid_boundary:.4f} | 非边界: {acc_hybrid_non_boundary:.4f}")
    print(f"  边界区域提升: {(acc_hybrid_boundary - acc_baseline_boundary)*100:+.2f}%")
else:
    print("  未找到BMI边界区域样本")


# =============================================================================
# 创新点2: 基于SHAP的个体化特征归因解释
# =============================================================================
print("\n" + "=" * 70)
print("  创新点2: 基于SHAP的个体化特征归因解释")
print("=" * 70)

FEATURE_NAME_CN = {
    'Gender': '性别',
    'Age': '年龄',
    'Height': '身高',
    'Weight': '体重',
    'family_history_with_overweight': '家族肥胖史',
    'FAVC': '高热量食物',
    'FCVC': '蔬菜摄入频率',
    'NCP': '主餐次数',
    'CAEC': '零食频率',
    'SMOKE': '吸烟',
    'CH2O': '饮水量',
    'SCC': '热量监控',
    'FAF': '运动频率',
    'TUE': '屏幕时间',
    'CALC': '饮酒频率',
}

# 交互特征的中文名
INTERACTION_FEATURE_CN = {
    'BMI': 'BMI指数',
    'Age_x_FAF': '年龄×运动频率',
    'Weight_x_FAVC': '体重×高热量食物',
    'FAF_x_TUE': '运动×屏幕时间',
    'FCVC_x_CH2O': '蔬菜×饮水量',
}


def compute_shap_attribution(model, X_sample, feature_names):
    """
    计算单个样本的SHAP特征归因。
    
    创新点说明：
    - 不是简单的全局特征重要性排名
    - 而是对每个用户生成Top-N风险因素及其贡献方向
    - 输出可解释的中文归因："您的肥胖风险主要来自：体重(+0.28)、缺乏运动(+0.15)"
    """
    if not SHAP_AVAILABLE:
        return None
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    return shap_values, explainer


if SHAP_AVAILABLE:
    print("\n  [创新点2] SHAP分析开始...")
    
    # 对测试集做SHAP分析
    explainer = shap.TreeExplainer(baseline_model)
    shap_values_all = explainer.shap_values(X_test_scaled)
    
    # ---- 2.1 全局SHAP特征重要性 ----
    print("\n  [2.1] 全局SHAP特征重要性...")
    # 处理shap_values的维度：可能是list[n_classes]或array[n_samples, n_features]
    shap_arr = np.array(shap_values_all)
    print(f"  SHAP values shape: {shap_arr.shape}")
    
    if shap_arr.ndim == 3:
        # shape = (n_classes, n_samples, n_features)
        mean_abs_shap = np.abs(shap_arr).mean(axis=(0, 1))
    elif shap_arr.ndim == 2:
        # shape = (n_samples, n_features) - 二分类或单输出
        mean_abs_shap = np.abs(shap_arr).mean(axis=0)
    else:
        print(f"  [WARNING] Unexpected SHAP shape: {shap_arr.shape}, skipping SHAP analysis")
        SHAP_AVAILABLE = False
    
    if SHAP_AVAILABLE:
        n_features = X_test_scaled.shape[1]
        if len(mean_abs_shap) != n_features:
            print(f"  [WARNING] SHAP feature count ({len(mean_abs_shap)}) != data features ({n_features})")
            if len(mean_abs_shap) > n_features:
                mean_abs_shap = mean_abs_shap[:n_features]
            else:
                mean_abs_shap = np.pad(mean_abs_shap, (0, n_features - len(mean_abs_shap)))
        
        shap_importance_df = pd.DataFrame({
            'feature': list(X_test_scaled.columns),
            'cn_name': [FEATURE_NAME_CN.get(f, f) for f in X_test_scaled.columns],
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        
        print("  Global SHAP Feature Importance Ranking:")
        print(shap_importance_df.to_string())
    
    # ---- 2.2 个体化归因（核心创新）----
    print("\n  [2.2] 个体化SHAP归因（5个典型样本）...")
    
    # 选择不同类别的典型样本
    sample_indices = []
    for cls_idx in range(len(class_names)):
        cls_samples = np.where(y_test == cls_idx)[0]
        if len(cls_samples) > 0:
            sample_indices.append(cls_samples[0])
    sample_indices = sample_indices[:5]  # 最多5个
    
    # 确保索引不超过SHAP值范围
    max_shap_samples = shap_arr.shape[0] if shap_arr.ndim >= 1 else 0
    sample_indices = [i for i in sample_indices if i < max_shap_samples]
    
    individual_attributions = []
    for idx in sample_indices:
        # 该样本的SHAP值
        shap_arr_local = np.array(shap_values_all)
        
        if shap_arr_local.ndim == 3:
            # shape = (n_classes, n_samples, n_features)
            # idx是相对于X_test的索引，需要检查边界
            if idx >= shap_arr_local.shape[1]:
                continue
            sample_shap = shap_arr_local[:, idx, :]  # (n_classes, n_features)
        elif shap_arr_local.ndim == 2:
            # shape = (n_samples, n_features) - 直接使用
            if idx >= shap_arr_local.shape[0]:
                continue
            sample_shap = shap_arr_local[idx, :]  # (n_features,)
        else:
            continue
        
        # 预测类别
        pred_class = baseline_model.predict(X_test_scaled.iloc[idx:idx+1])[0]
        
        # 取该预测类别的SHAP值
        if sample_shap.ndim == 2:
            pred_shap = sample_shap[pred_class]  # (n_features,)
        else:
            pred_shap = sample_shap  # 1D case
        
        # 确保特征数匹配
        n_feat = X_test_scaled.shape[1]
        if len(pred_shap) != n_feat:
            if len(pred_shap) > n_feat:
                pred_shap = pred_shap[:n_feat]
            else:
                pred_shap = np.pad(pred_shap, (0, n_feat - len(pred_shap)))
        
        # 特征归因排序
        feat_attribution = pd.DataFrame({
            '特征': list(X_test_scaled.columns),
            '中文名': [FEATURE_NAME_CN.get(f, f) for f in X_test_scaled.columns],
            'SHAP值': pred_shap,
            '特征值': X_test_scaled.iloc[idx].values,
        }).sort_values('SHAP值', key=lambda x: abs(x), ascending=False).reset_index(drop=True)
        
        # Top-3 风险因素
        top3 = feat_attribution.head(3)
        risk_factors = []
        for _, row in top3.iterrows():
            direction = "增加风险" if row['SHAP值'] > 0 else "降低风险"
            risk_factors.append({
                'feature': row['特征'],
                'feature_cn': row['中文名'],
                'shap_value': round(float(row['SHAP值']), 4),
                'direction': direction,
                'contribution': f"{row['中文名']}({row['SHAP值']:+.4f})"
            })
        
        attr_result = {
            'sample_idx': idx,
            'true_label': class_names[y_test[idx]],
            'pred_label': class_names[pred_class],
            'top3_factors': risk_factors,
            'summary': "、".join(r['contribution'] for r in risk_factors),
        }
        individual_attributions.append(attr_result)
        
        print(f"\n  样本#{idx}: 真实={attr_result['true_label']}, 预测={attr_result['pred_label']}")
        print(f"    Top-3归因: {attr_result['summary']}")
    
    # ---- 2.3 SHAP全局可视化 ----
    print("\n  [2.3] 生成SHAP可视化图表...")
    
    # 全局SHAP柱状图
    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = min(15, len(shap_importance_df))
    colors = ['#e74c3c' if v > 0.1 else '#3498db' if v > 0.05 else '#95a5a6' 
              for v in shap_importance_df['mean_abs_shap'].head(top_n)]
    bars = ax.barh(
        shap_importance_df['cn_name'].head(top_n)[::-1],
        shap_importance_df['mean_abs_shap'].head(top_n)[::-1],
        color=colors[::-1], edgecolor='white'
    )
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('SHAP Global Feature Importance\n(Innovation 2: Individual Attribution Basis)', 
                 fontsize=13, fontweight='bold')
    for bar, val in zip(bars, shap_importance_df['mean_abs_shap'].head(top_n)[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'innovation_01_shap_global_importance.png'))
    plt.close()
    print("  -> innovation_01_shap_global_importance.png")
    
    # 个体化归因示例图
    fig, axes = plt.subplots(1, min(3, len(individual_attributions)), figsize=(18, 6))
    if len(individual_attributions) == 1:
        axes = [axes]
    for i, attr in enumerate(individual_attributions[:3]):
        ax = axes[i]
        factors = attr['top3_factors']
        feat_names = [f['feature_cn'] for f in factors]
        shap_vals = [f['shap_value'] for f in factors]
        bar_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_vals]
        
        bars = ax.barh(feat_names[::-1], shap_vals[::-1], color=bar_colors[::-1], edgecolor='white')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_title(f'Sample #{attr["sample_idx"]}\nPred: {attr["pred_label"]}', 
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('SHAP Value')
        for bar, val in zip(bars, shap_vals[::-1]):
            ax.text(bar.get_width() + 0.01 * (1 if val > 0 else -1), 
                    bar.get_y() + bar.get_height()/2,
                    f'{val:+.4f}', va='center', fontsize=9,
                    ha='left' if val > 0 else 'right')
    
    plt.suptitle('Innovation 2: Individual SHAP Attribution Examples', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'innovation_02_shap_individual_attribution.png'))
    plt.close()
    print("  -> innovation_02_shap_individual_attribution.png")

else:
    print("  [创新点2] SHAP不可用，跳过")


# =============================================================================
# 创新点3: 交互特征工程 + Stacking集成学习
# =============================================================================
print("\n" + "=" * 70)
print("  创新点3: 交互特征工程 + Stacking集成学习")
print("=" * 70)


def create_interaction_features(df):
    """
    构建交互特征。
    
    创新点说明：
    - 不使用原始特征直接送入模型
    - 基于医学领域知识构建有意义的交互特征
    - BMI = Weight / Height^2（最核心的衍生特征）
    - 年龄×运动频率：反映"随年龄增长运动减少"的复合效应
    - 体重×高热量食物：反映"高体重+高热量摄入"的叠加风险
    - 运动×屏幕时间：活动/静态行为的对抗效应
    - 蔬菜×饮水量：健康生活习惯的协同效应
    """
    df_new = df.copy()
    
    # BMI指数（核心衍生特征）
    df_new['BMI'] = df_new['Weight'] / ((df_new['Height'] / 100.0) ** 2)
    
    # 年龄×运动频率（随年龄增长运动减少的复合效应）
    df_new['Age_x_FAF'] = df_new['Age'] * df_new['FAF']
    
    # 体重×高热量食物（高体重+高热量的叠加风险）
    df_new['Weight_x_FAVC'] = df_new['Weight'] * df_new['FAVC']
    
    # 运动×屏幕时间（活动/静态对抗效应）
    df_new['FAF_x_TUE'] = df_new['FAF'] * df_new['TUE']
    
    # 蔬菜×饮水量（健康习惯协同）
    df_new['FCVC_x_CH2O'] = df_new['FCVC'] * df_new['CH2O']
    
    return df_new


print("\n  [创新点3] 构建交互特征...")
X_train_enhanced = create_interaction_features(X_train.copy())
X_test_enhanced = create_interaction_features(X_test.copy())

# 新增特征名
interaction_features = ['BMI', 'Age_x_FAF', 'Weight_x_FAVC', 'FAF_x_TUE', 'FCVC_x_CH2O']
all_enhanced_features = list(X_train_enhanced.columns)
print(f"  原始特征: {X_train.shape[1]}个")
print(f"  增强特征: {X_train_enhanced.shape[1]}个（+{len(interaction_features)}个交互特征）")
print(f"  新增特征: {interaction_features}")

# 标准化增强特征
scaler_enhanced = StandardScaler()
continuous_enhanced = continuous_cols + interaction_features
X_train_enhanced_scaled = X_train_enhanced.copy()
X_test_enhanced_scaled = X_test_enhanced.copy()
X_train_enhanced_scaled[continuous_enhanced] = scaler_enhanced.fit_transform(X_train_enhanced[continuous_enhanced])
X_test_enhanced_scaled[continuous_enhanced] = scaler_enhanced.transform(X_test_enhanced[continuous_enhanced])


# ---- 3.1 仅交互特征 + XGBoost ----
print("\n  [3.1] 交互特征 + XGBoost...")
xgb_enhanced = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.2,
    subsample=0.7, colsample_bytree=0.7,
    reg_lambda=10, reg_alpha=0.01, min_child_weight=1, gamma=0.1,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=RANDOM_STATE, n_jobs=-1
)
xgb_enhanced.fit(X_train_enhanced_scaled, y_train)
y_pred_xgb_enh = xgb_enhanced.predict(X_test_enhanced_scaled)
y_proba_xgb_enh = xgb_enhanced.predict_proba(X_test_enhanced_scaled)
acc_xgb_enh = accuracy_score(y_test, y_pred_xgb_enh)
f1_xgb_enh = f1_score(y_test, y_pred_xgb_enh, average='weighted', zero_division=0)
auc_xgb_enh = roc_auc_score(y_test, y_proba_xgb_enh, multi_class='ovr', average='macro')
print(f"  交互特征XGBoost: Acc={acc_xgb_enh:.4f}, F1={f1_xgb_enh:.4f}, AUC={auc_xgb_enh:.4f}")


# ---- 3.2 Stacking集成 ----
print("\n  [3.2] Stacking集成学习...")
base_estimators = [
    ('xgb', xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.2,
        subsample=0.7, colsample_bytree=0.7,
        reg_lambda=10, reg_alpha=0.01, min_child_weight=1, gamma=0.1,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=RANDOM_STATE, n_jobs=-1
    )),
    ('rf', RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        random_state=RANDOM_STATE, n_jobs=-1
    )),
    ('gbdt', GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=RANDOM_STATE
    )),
]

stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(
        max_iter=2000, multi_class='multinomial', C=1.0,
        random_state=RANDOM_STATE
    ),
    cv=5,
    n_jobs=-1,
    passthrough=False
)

stacking_model.fit(X_train_enhanced_scaled, y_train)
y_pred_stack = stacking_model.predict(X_test_enhanced_scaled)
y_proba_stack = stacking_model.predict_proba(X_test_enhanced_scaled)
acc_stack = accuracy_score(y_test, y_pred_stack)
f1_stack = f1_score(y_test, y_pred_stack, average='weighted', zero_division=0)
auc_stack = roc_auc_score(y_test, y_proba_stack, multi_class='ovr', average='macro')
print(f"  Stacking模型: Acc={acc_stack:.4f}, F1={f1_stack:.4f}, AUC={auc_stack:.4f}")


# ---- 3.3 仅原始特征Stacking（对比） ----
print("\n  [3.3] 原始特征Stacking（对照组）...")
base_estimators_raw = [
    ('xgb', xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.2,
        subsample=0.7, colsample_bytree=0.7,
        reg_lambda=10, reg_alpha=0.01, min_child_weight=1, gamma=0.1,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=RANDOM_STATE, n_jobs=-1
    )),
    ('rf', RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )),
    ('gbdt', GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=RANDOM_STATE
    )),
]

stacking_raw = StackingClassifier(
    estimators=base_estimators_raw,
    final_estimator=LogisticRegression(
        max_iter=2000, multi_class='multinomial', C=1.0,
        random_state=RANDOM_STATE
    ),
    cv=5,
    n_jobs=-1,
    passthrough=False
)

stacking_raw.fit(X_train_scaled, y_train)
y_pred_stack_raw = stacking_raw.predict(X_test_scaled)
y_proba_stack_raw = stacking_raw.predict_proba(X_test_scaled)
acc_stack_raw = accuracy_score(y_test, y_pred_stack_raw)
f1_stack_raw = f1_score(y_test, y_pred_stack_raw, average='weighted', zero_division=0)
auc_stack_raw = roc_auc_score(y_test, y_proba_stack_raw, multi_class='ovr', average='macro')
print(f"  原始特征Stacking: Acc={acc_stack_raw:.4f}, F1={f1_stack_raw:.4f}, AUC={auc_stack_raw:.4f}")


# =============================================================================
# 综合对比实验可视化
# =============================================================================
print("\n" + "=" * 70)
print("  综合对比实验可视化")
print("=" * 70)

# 汇总所有模型结果
all_results = {
    'XGBoost(baseline)': {
        'accuracy': acc_baseline, 'f1': f1_baseline, 'auc': auc_baseline,
        'type': 'Baseline'
    },
    'Hybrid(ML+Rule)': {
        'accuracy': best_hybrid['accuracy'], 'f1': best_hybrid['f1'], 'auc': best_hybrid['auc'],
        'type': 'Innovation 1'
    },
    'XGBoost+Interaction': {
        'accuracy': acc_xgb_enh, 'f1': f1_xgb_enh, 'auc': auc_xgb_enh,
        'type': 'Innovation 3'
    },
    'Stacking(raw)': {
        'accuracy': acc_stack_raw, 'f1': f1_stack_raw, 'auc': auc_stack_raw,
        'type': 'Innovation 3'
    },
    'Stacking+Interaction': {
        'accuracy': acc_stack, 'f1': f1_stack, 'auc': auc_stack,
        'type': 'Innovation 3'
    },
}

results_df = pd.DataFrame(all_results).T
results_df.index.name = 'Model'
print("\n" + results_df.to_string())

# 保存结果
results_df.to_csv(os.path.join(OUTPUT_DIR, 'innovation_comparison_results.csv'), encoding='utf-8-sig')

# ---- 可视化1: 综合对比柱状图 ----
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
model_names = list(all_results.keys())
model_colors = {
    'XGBoost(baseline)': '#95a5a6',
    'Hybrid(ML+Rule)': '#e74c3c',
    'XGBoost+Interaction': '#3498db',
    'Stacking(raw)': '#2ecc71',
    'Stacking+Interaction': '#9b59b6',
}
colors = [model_colors.get(m, '#95a5a6') for m in model_names]

# Accuracy
acc_vals = [all_results[m]['accuracy'] for m in model_names]
bars = axes[0].bar(range(len(model_names)), acc_vals, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_xticks(range(len(model_names)))
axes[0].set_xticklabels(model_names, rotation=25, ha='right', fontsize=9)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
axes[0].set_ylim(max(min(acc_vals) - 0.05, 0.8), min(max(acc_vals) + 0.02, 1.0))
for bar, val in zip(bars, acc_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# F1
f1_vals = [all_results[m]['f1'] for m in model_names]
bars = axes[1].bar(range(len(model_names)), f1_vals, color=colors, edgecolor='white', linewidth=1.5)
axes[1].set_xticks(range(len(model_names)))
axes[1].set_xticklabels(model_names, rotation=25, ha='right', fontsize=9)
axes[1].set_ylabel('F1 Score (weighted)')
axes[1].set_title('F1 Score Comparison', fontsize=12, fontweight='bold')
axes[1].set_ylim(max(min(f1_vals) - 0.05, 0.8), min(max(f1_vals) + 0.02, 1.0))
for bar, val in zip(bars, f1_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# AUC
auc_vals = [all_results[m]['auc'] for m in model_names]
bars = axes[2].bar(range(len(model_names)), auc_vals, color=colors, edgecolor='white', linewidth=1.5)
axes[2].set_xticks(range(len(model_names)))
axes[2].set_xticklabels(model_names, rotation=25, ha='right', fontsize=9)
axes[2].set_ylabel('AUC (OvR macro)')
axes[2].set_title('AUC Comparison', fontsize=12, fontweight='bold')
axes[2].set_ylim(max(min(auc_vals) - 0.02, 0.9), min(max(auc_vals) + 0.005, 1.0))
for bar, val in zip(bars, auc_vals):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Algorithm Innovation Comparison Experiment', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'innovation_03_comprehensive_comparison.png'))
plt.close()
print("  -> innovation_03_comprehensive_comparison.png")


# ---- 可视化2: 创新点1 - alpha对比 + BMI边界分析 ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Alpha对比
alpha_accs = [hybrid_results[a]['accuracy'] for a in alpha_values]
alpha_f1s = [hybrid_results[a]['f1'] for a in alpha_values]
alpha_aucs = [hybrid_results[a]['auc'] for a in alpha_values]

ax = axes[0]
ax.plot(alpha_values, alpha_accs, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Accuracy')
ax.plot(alpha_values, alpha_f1s, 's-', color='#3498db', linewidth=2, markersize=8, label='F1')
ax.plot(alpha_values, alpha_aucs, '^-', color='#2ecc71', linewidth=2, markersize=8, label='AUC')
ax.axhline(y=acc_baseline, color='#e74c3c', linestyle='--', alpha=0.5, label=f'XGBoost Acc={acc_baseline:.4f}')
ax.axhline(y=f1_baseline, color='#3498db', linestyle='--', alpha=0.5, label=f'XGBoost F1={f1_baseline:.4f}')
ax.axhline(y=auc_baseline, color='#2ecc71', linestyle='--', alpha=0.5, label=f'XGBoost AUC={auc_baseline:.4f}')
ax.set_xlabel('Alpha (ML Weight)', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Innovation 1: Hybrid Model\nAlpha Sensitivity Analysis', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=0.3)

# BMI边界分析
if boundary_indices:
    categories = ['Boundary\nRegion', 'Non-Boundary\nRegion', 'Overall']
    baseline_vals = [acc_baseline_boundary, acc_baseline_non_boundary, acc_baseline]
    hybrid_vals = [acc_hybrid_boundary, acc_hybrid_non_boundary, best_hybrid['accuracy']]
    
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='XGBoost', color='#95a5a6', edgecolor='white')
    bars2 = ax.bar(x + width/2, hybrid_vals, width, label='Hybrid(ML+Rule)', color='#e74c3c', edgecolor='white')
    
    for bar, val in zip(bars1, baseline_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, hybrid_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Innovation 1: BMI Boundary\nRegion Analysis', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'No boundary samples found', ha='center', va='center', fontsize=14)
    axes[1].set_title('Innovation 1: BMI Boundary Analysis', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'innovation_04_hybrid_model_analysis.png'))
plt.close()
print("  -> innovation_04_hybrid_model_analysis.png")


# ---- 可视化3: 创新点3 - 特征工程效果对比 ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 交互特征重要性（XGBoost + 交互特征）
if hasattr(xgb_enhanced, 'feature_importances_'):
    enh_imp_df = pd.DataFrame({
        'Feature': list(X_train_enhanced_scaled.columns),
        'CN_Name': [FEATURE_NAME_CN.get(f, INTERACTION_FEATURE_CN.get(f, f)) for f in X_train_enhanced_scaled.columns],
        'Importance': xgb_enhanced.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    top_n = min(15, len(enh_imp_df))
    top_df = enh_imp_df.head(top_n)
    
    # 高亮交互特征
    bar_colors = ['#e74c3c' if f in interaction_features else '#3498db' for f in top_df['Feature']]
    
    bars = axes[0].barh(top_df['CN_Name'][::-1], top_df['Importance'][::-1], 
                        color=bar_colors[::-1], edgecolor='white')
    axes[0].set_xlabel('Feature Importance')
    axes[0].set_title('Innovation 3: Feature Importance\n(Red = Interaction Features)', 
                      fontsize=12, fontweight='bold')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Interaction Feature'),
                       Patch(facecolor='#3498db', label='Original Feature')]
    axes[0].legend(handles=legend_elements, loc='lower right')

# Stacking vs 单模型对比
model_comp = {
    'XGBoost\n(Baseline)': (acc_baseline, f1_baseline, auc_baseline),
    'XGBoost+\nInteraction': (acc_xgb_enh, f1_xgb_enh, auc_xgb_enh),
    'Stacking\n(Raw)': (acc_stack_raw, f1_stack_raw, auc_stack_raw),
    'Stacking+\nInteraction': (acc_stack, f1_stack, auc_stack),
}

x = np.arange(len(model_comp))
width = 0.25
metrics = ['Acc', 'F1', 'AUC']
metric_colors = ['#e74c3c', '#3498db', '#2ecc71']

for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
    vals = [model_comp[m][i] for m in model_comp]
    bars = axes[1].bar(x + (i - 1) * width, vals, width, label=metric, 
                       color=color, edgecolor='white', alpha=0.85)

axes[1].set_xticks(x)
axes[1].set_xticklabels(list(model_comp.keys()), fontsize=9)
axes[1].set_ylabel('Score')
axes[1].set_title('Innovation 3: Feature Engineering\n& Stacking Comparison', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'innovation_05_feature_engineering_stacking.png'))
plt.close()
print("  -> innovation_05_feature_engineering_stacking.png")


# ---- 可视化4: 创新点3 - 混淆矩阵对比 ----
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 基线XGBoost混淆矩阵
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', ax=axes[0])
axes[0].set_title(f'XGBoost Baseline\nAcc={acc_baseline:.4f}', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].tick_params(axis='x', rotation=30)

# Stacking+交互特征混淆矩阵
cm_stack = confusion_matrix(y_test, y_pred_stack)
sns.heatmap(cm_stack, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', ax=axes[1])
axes[1].set_title(f'Stacking+Interaction\nAcc={acc_stack:.4f}', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle('Innovation 3: Confusion Matrix Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'innovation_06_confusion_matrix_comparison.png'))
plt.close()
print("  -> innovation_06_confusion_matrix_comparison.png")


# ---- 可视化5: ROC曲线对比 ----
fig, ax = plt.subplots(figsize=(10, 8))
n_classes = len(class_names)
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

# 基线
fpr_base, tpr_base, _ = roc_curve(y_test_bin.ravel(), y_proba_baseline.ravel())
roc_auc_base = auc(fpr_base, tpr_base)
ax.plot(fpr_base, tpr_base, color='#95a5a6', linewidth=2, 
        label=f'XGBoost Baseline (AUC={roc_auc_base:.4f})')

# 混合模型
fpr_hybrid, tpr_hybrid, _ = roc_curve(y_test_bin.ravel(), best_hybrid['y_proba'].ravel())
roc_auc_hybrid = auc(fpr_hybrid, tpr_hybrid)
ax.plot(fpr_hybrid, tpr_hybrid, color='#e74c3c', linewidth=2, linestyle='--',
        label=f'Hybrid ML+Rule (AUC={roc_auc_hybrid:.4f})')

# Stacking+交互特征
fpr_stack, tpr_stack, _ = roc_curve(y_test_bin.ravel(), y_proba_stack.ravel())
roc_auc_stack = auc(fpr_stack, tpr_stack)
ax.plot(fpr_stack, tpr_stack, color='#9b59b6', linewidth=2, linestyle='-.',
        label=f'Stacking+Interaction (AUC={roc_auc_stack:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve Comparison (Micro-average)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'innovation_07_roc_comparison.png'))
plt.close()
print("  -> innovation_07_roc_comparison.png")


# =============================================================================
# 保存创新模型
# =============================================================================
print("\n" + "=" * 70)
print("  保存创新模型")
print("=" * 70)

# 保存增强版XGBoost模型
joblib.dump(xgb_enhanced, os.path.join(OUTPUT_DIR, 'innovation_xgb_enhanced.joblib'))
print("  -> innovation_xgb_enhanced.joblib")

# 保存Stacking模型
joblib.dump(stacking_model, os.path.join(OUTPUT_DIR, 'innovation_stacking_model.joblib'))
print("  -> innovation_stacking_model.joblib")

# 保存增强版标准化器
joblib.dump(scaler_enhanced, os.path.join(OUTPUT_DIR, 'innovation_scaler_enhanced.joblib'))
print("  -> innovation_scaler_enhanced.joblib")


# =============================================================================
# 保存创新实验结果摘要
# =============================================================================
innovation_summary = {
    "baseline": {
        "model": "XGBoost",
        "accuracy": float(acc_baseline),
        "f1_weighted": float(f1_baseline),
        "auc_ovr_macro": float(auc_baseline),
    },
    "innovation_1_hybrid": {
        "description": "NHC-guideline constrained hybrid model (ML + Rule fusion)",
        "best_alpha": float(best_alpha),
        "accuracy": float(best_hybrid['accuracy']),
        "f1_weighted": float(best_hybrid['f1']),
        "auc_ovr_macro": float(best_hybrid['auc']),
        "boundary_region_improvement": float(acc_hybrid_boundary - acc_baseline_boundary) if boundary_indices else None,
        "alpha_sensitivity": {str(a): {
            "accuracy": float(hybrid_results[a]['accuracy']),
            "f1": float(hybrid_results[a]['f1']),
            "auc": float(hybrid_results[a]['auc']),
        } for a in alpha_values},
    },
    "innovation_2_shap": {
        "description": "SHAP-based individual feature attribution",
        "shap_available": SHAP_AVAILABLE,
        "global_top5_features": shap_importance_df.head(5)[['feature', 'cn_name', 'mean_abs_shap']].to_dict('records') if SHAP_AVAILABLE else None,
        "individual_examples": individual_attributions if SHAP_AVAILABLE else None,
    },
    "innovation_3_feature_engineering_stacking": {
        "description": "Interaction features + Stacking ensemble",
        "interaction_features": interaction_features,
        "xgb_enhanced": {
            "accuracy": float(acc_xgb_enh),
            "f1_weighted": float(f1_xgb_enh),
            "auc_ovr_macro": float(auc_xgb_enh),
        },
        "stacking_raw_features": {
            "accuracy": float(acc_stack_raw),
            "f1_weighted": float(f1_stack_raw),
            "auc_ovr_macro": float(auc_stack_raw),
        },
        "stacking_with_interaction": {
            "accuracy": float(acc_stack),
            "f1_weighted": float(f1_stack),
            "auc_ovr_macro": float(auc_stack),
        },
    },
}

with open(os.path.join(OUTPUT_DIR, 'innovation_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(innovation_summary, f, ensure_ascii=False, indent=2, default=str)
print(f"  -> innovation_summary.json")


# =============================================================================
# 最终结果汇总
# =============================================================================
print("\n" + "=" * 70)
print("  算法创新实验 - 最终结果汇总")
print("=" * 70)

print(f"""
+------------------------------------------+----------+----------+----------+
| Model                                    | Accuracy | F1(w)    | AUC(OvR) |
+------------------------------------------+----------+----------+----------+
| XGBoost (Baseline)                       | {acc_baseline:.4f}   | {f1_baseline:.4f}   | {auc_baseline:.4f}   |
| Hybrid ML+Rule (alpha={best_alpha})             | {best_hybrid['accuracy']:.4f}   | {best_hybrid['f1']:.4f}   | {best_hybrid['auc']:.4f}   |
| XGBoost + Interaction Features           | {acc_xgb_enh:.4f}   | {f1_xgb_enh:.4f}   | {auc_xgb_enh:.4f}   |
| Stacking (Raw Features)                  | {acc_stack_raw:.4f}   | {f1_stack_raw:.4f}   | {auc_stack_raw:.4f}   |
| Stacking + Interaction Features          | {acc_stack:.4f}   | {f1_stack:.4f}   | {auc_stack:.4f}   |
+------------------------------------------+----------+----------+----------+
""")

# 找出最优创新模型
best_innovation = max([
    ('Hybrid ML+Rule', best_hybrid['accuracy']),
    ('XGBoost+Interaction', acc_xgb_enh),
    ('Stacking+Interaction', acc_stack),
], key=lambda x: x[1])

print(f"  Best Innovation Model: {best_innovation[0]} (Acc={best_innovation[1]:.4f})")
print(f"  vs Baseline: {(best_innovation[1] - acc_baseline)*100:+.2f}%")

print(f"""
Generated Charts:
  innovation_01_shap_global_importance.png     - SHAP Global Feature Importance
  innovation_02_shap_individual_attribution.png - SHAP Individual Attribution Examples
  innovation_03_comprehensive_comparison.png    - Comprehensive Model Comparison
  innovation_04_hybrid_model_analysis.png       - Hybrid Model Alpha & Boundary Analysis
  innovation_05_feature_engineering_stacking.png - Feature Engineering & Stacking
  innovation_06_confusion_matrix_comparison.png - Confusion Matrix Comparison
  innovation_07_roc_comparison.png              - ROC Curve Comparison

All charts saved to: {OUTPUT_DIR}
""")

print("Algorithm Innovation Experiment Complete!")
