# =============================================================================
# 肥胖风险数据集 - 完整数据分析与机器学习建模流程
# 涵盖：EDA → 数据预处理 → 变量筛选 → 模型对比 → 参数调优 → 最优模型
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合脚本运行
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pointbiserialr

# 机器学习核心库
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
    RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler, label_binarize
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif, RFE
)
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import joblib

# ─────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "obesity_level.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配置中文字体（Windows环境）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

print("=" * 70)
print("  肥胖风险数据集 - 完整数据分析与机器学习建模流程")
print("=" * 70)


# =============================================================================
# 一、数据加载与基础探索 (Data Loading & Basic EDA)
# =============================================================================
print("\n" + "─" * 70)
print("【第一步】数据加载与基础探索")
print("─" * 70)

df_raw = pd.read_csv(DATA_PATH, index_col='id')
print(f"\n数据集形状：{df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")
print("\n前5行数据预览：")
print(df_raw.head())

print("\n数据类型信息：")
print(df_raw.dtypes)

print("\n基础统计描述（数值型）：")
print(df_raw.describe())

print("\n基础统计描述（类别型）：")
print(df_raw.describe(include=['object']))

# ─── 缺失值分析 ───
print("\n【缺失值分析】")
missing = df_raw.isnull().sum()
missing_pct = (df_raw.isnull().sum() / len(df_raw)) * 100
missing_df = pd.DataFrame({'缺失数量': missing, '缺失比例(%)': missing_pct})
missing_df = missing_df[missing_df['缺失数量'] > 0]
if missing_df.empty:
    print("  → 数据集中无缺失值，数据完整性良好。")
else:
    print(missing_df)

# ─── 重复值分析 ───
print("\n【重复值分析】")
dup_count = df_raw.duplicated().sum()
print(f"  → 重复行数量：{dup_count}")
if dup_count > 0:
    df_raw = df_raw.drop_duplicates().reset_index(drop=True)
    print(f"  → 已删除重复行，剩余 {len(df_raw)} 行")

# ─── 目标变量分布 ───
print("\n【目标变量分布（0be1dad）】")
target_counts = df_raw['0be1dad'].value_counts()
target_pct = df_raw['0be1dad'].value_counts(normalize=True) * 100
target_df = pd.DataFrame({'数量': target_counts, '占比(%)': target_pct.round(2)})
print(target_df)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_palette = sns.color_palette("Set2", n_colors=len(target_counts))

# 条形图
bars = axes[0].bar(range(len(target_counts)), target_counts.values, color=colors_palette, edgecolor='white', linewidth=0.8)
axes[0].set_xticks(range(len(target_counts)))
axes[0].set_xticklabels(target_counts.index, rotation=30, ha='right', fontsize=9)
axes[0].set_title('肥胖等级分布（频数）', fontsize=13, fontweight='bold')
axes[0].set_ylabel('样本数量')
axes[0].set_xlabel('肥胖等级')
for bar, val in zip(bars, target_counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(val), ha='center', va='bottom', fontsize=9)

# 饼图
wedges, texts, autotexts = axes[1].pie(
    target_counts.values, labels=target_counts.index,
    autopct='%1.1f%%', colors=colors_palette, startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1}
)
for text in texts:
    text.set_fontsize(8)
axes[1].set_title('肥胖等级分布（占比）', fontsize=13, fontweight='bold')

plt.suptitle('目标变量：肥胖等级分布', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_target_distribution.png'))
plt.close()
print("  → 图表已保存：01_target_distribution.png")


# =============================================================================
# 二、深度探索性数据分析 (Deep EDA)
# =============================================================================
print("\n" + "─" * 70)
print("【第二步】深度探索性数据分析（EDA）")
print("─" * 70)

# 区分数值型与类别型特征（不含目标变量）
num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_raw.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c != '0be1dad']
print(f"\n  数值型特征（{len(num_cols)}个）：{num_cols}")
print(f"  类别型特征（{len(cat_cols)}个）：{cat_cols}")
print(f"  目标变量：0be1dad")

# ─── 数值型特征分布图 ───
n_num = len(num_cols)
n_cols_per_row = 4
n_rows = (n_num + n_cols_per_row - 1) // n_cols_per_row
fig, axes = plt.subplots(n_rows, n_cols_per_row, figsize=(18, 4 * n_rows))
axes = axes.flatten()
for idx, col in enumerate(num_cols):
    ax = axes[idx]
    data = df_raw[col].dropna()
    ax.hist(data, bins=30, color=sns.color_palette("Blues_d", 1)[0],
            edgecolor='white', alpha=0.85)
    ax2 = ax.twinx()
    data.plot.kde(ax=ax2, color='tomato', linewidth=2)
    ax2.set_ylabel('')
    ax2.set_yticks([])
    ax.set_title(f'{col}\n偏度={data.skew():.2f}  峰度={data.kurtosis():.2f}',
                 fontsize=9)
    ax.set_xlabel('')
for idx in range(n_num, len(axes)):
    axes[idx].set_visible(False)
plt.suptitle('数值型特征分布（直方图 + KDE）', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_numerical_distribution.png'))
plt.close()
print("  → 图表已保存：02_numerical_distribution.png")

# ─── 箱线图：数值特征按目标分类 ───
fig, axes = plt.subplots(n_rows, n_cols_per_row, figsize=(20, 4 * n_rows))
axes = axes.flatten()
target_order = sorted(df_raw['0be1dad'].unique())
for idx, col in enumerate(num_cols):
    ax = axes[idx]
    groups = [df_raw[df_raw['0be1dad'] == t][col].dropna().values for t in target_order]
    bp = ax.boxplot(groups, patch_artist=True, notch=False,
                    medianprops={'color': 'black', 'linewidth': 2})
    for patch, color in zip(bp['boxes'], sns.color_palette("Set2", len(target_order))):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticklabels([t[:12] for t in target_order], rotation=30, ha='right', fontsize=7)
    ax.set_title(f'{col} vs 肥胖等级', fontsize=9)
for idx in range(n_num, len(axes)):
    axes[idx].set_visible(False)
plt.suptitle('数值型特征 vs 肥胖等级（箱线图）', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_boxplot_by_target.png'))
plt.close()
print("  → 图表已保存：03_boxplot_by_target.png")

# ─── 类别型特征频率分布 ───
n_cat = len(cat_cols)
n_rows_cat = (n_cat + 2) // 3
fig, axes = plt.subplots(n_rows_cat, 3, figsize=(18, 4 * n_rows_cat))
axes = axes.flatten()
for idx, col in enumerate(cat_cols):
    ax = axes[idx]
    vc = df_raw[col].value_counts()
    bars = ax.bar(range(len(vc)), vc.values,
                  color=sns.color_palette("Pastel1", len(vc)), edgecolor='gray', linewidth=0.5)
    ax.set_xticks(range(len(vc)))
    ax.set_xticklabels(vc.index, rotation=25, ha='right', fontsize=8)
    ax.set_title(f'{col}', fontsize=11, fontweight='bold')
    ax.set_ylabel('频数')
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontsize=8)
for idx in range(n_cat, len(axes)):
    axes[idx].set_visible(False)
plt.suptitle('类别型特征频率分布', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_categorical_distribution.png'))
plt.close()
print("  → 图表已保存：04_categorical_distribution.png")

# ─── 相关性热力图（数值型特征）───
print("\n【相关性矩阵分析】")
corr_matrix = df_raw[num_cols].corr(method='pearson')
print("Pearson 相关系数矩阵：")
print(corr_matrix.round(3))
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor='white', ax=ax,
            annot_kws={'size': 9})
ax.set_title('数值型特征 Pearson 相关系数热力图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_correlation_heatmap.png'))
plt.close()
print("  → 图表已保存：05_correlation_heatmap.png")

# ─── 异常值分析（IQR方法）───
print("\n【异常值分析（IQR 方法）】")
outlier_summary = {}
for col in num_cols:
    Q1 = df_raw[col].quantile(0.25)
    Q3 = df_raw[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df_raw[(df_raw[col] < lower) | (df_raw[col] > upper)]
    outlier_summary[col] = {
        'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
        '下界': lower, '上界': upper,
        '异常值数量': len(outliers),
        '异常值比例(%)': round(len(outliers) / len(df_raw) * 100, 2)
    }
outlier_df = pd.DataFrame(outlier_summary).T
print(outlier_df[['下界', '上界', '异常值数量', '异常值比例(%)']])

# ─── 正态性检验（Shapiro-Wilk，样本≤5000时适用）───
print("\n【正态性检验（Shapiro-Wilk）】")
normality_results = {}
sample_size = min(1000, len(df_raw))
for col in num_cols:
    sample_data = df_raw[col].dropna().sample(n=sample_size, random_state=RANDOM_STATE)
    stat, p_val = stats.shapiro(sample_data)
    normality_results[col] = {
        'W统计量': round(stat, 4),
        'p值': round(p_val, 6),
        '正态性': '是（p>0.05）' if p_val > 0.05 else '否（p≤0.05）'
    }
normality_df = pd.DataFrame(normality_results).T
print(normality_df)


# =============================================================================
# 三、数据预处理 (Data Preprocessing)
# =============================================================================
print("\n" + "─" * 70)
print("【第三步】数据预处理")
print("─" * 70)

df = df_raw.copy()

# ─── 标签统一化 ───
# 原数据目标列有 '0rmal_Weight' 拼写错误，统一为 'Normal_Weight'
df['0be1dad'] = df['0be1dad'].str.strip()
df['0be1dad'] = df['0be1dad'].replace({'0rmal_Weight': 'Normal_Weight',
                                        'Ormal_Weight': 'Normal_Weight'})
print(f"\n目标变量唯一值（修正后）：\n{df['0be1dad'].value_counts()}")

# ─── 类别型特征编码 ───
print("\n【类别型特征编码】")
# Gender: 二分类 → 0/1
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
print(f"  Gender 编码：Male→1, Female→0")

# family_history_with_overweight, FAVC, SMOKE, SCC: 已是 0/1
# 确认其值域
binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
for col in binary_cols:
    uniq = df[col].unique()
    print(f"  {col} 唯一值：{uniq}")

# CAEC（两餐间食物消耗频率）→ 有序编码
caec_map = {'no': 0, '0': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
df['CAEC'] = df['CAEC'].map(caec_map)
print(f"  CAEC 有序编码：no/0→0, Sometimes→1, Frequently→2, Always→3")
if df['CAEC'].isnull().any():
    df['CAEC'] = df['CAEC'].fillna(df['CAEC'].mode()[0])

# CALC（酒精消耗频率）→ 有序编码
calc_map = {'no': 0, '0': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
df['CALC'] = df['CALC'].map(calc_map)
print(f"  CALC 有序编码：no/0→0, Sometimes→1, Frequently→2, Always→3")
if df['CALC'].isnull().any():
    df['CALC'] = df['CALC'].fillna(df['CALC'].mode()[0])

# ─── 删除 MTRANS 列（不参与训练，简化特征体系）───
if 'MTRANS' in df.columns:
    df = df.drop(columns=['MTRANS'])
    print(f"  MTRANS 列已删除（不参与本次建模，共13个核心特征）")

# ─── 目标变量编码 ───
print("\n【目标变量 LabelEncoding】")
le = LabelEncoder()
y = le.fit_transform(df['0be1dad'])
class_names = le.classes_
print(f"  类别映射：")
for i, cls in enumerate(class_names):
    print(f"    {i} → {cls}")

# ─── 构建特征矩阵 ───
feature_cols = [c for c in df.columns if c != '0be1dad']
X = df[feature_cols].copy()
# 确保所有列都是数值型
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())
    X[col] = X[col].fillna(X[col].median())

print(f"\n特征矩阵形状：{X.shape}")
print(f"目标向量形状：{y.shape}")
print(f"特征列表：\n{list(X.columns)}")

# ─── 训练集/测试集划分（分层抽样）───
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n数据集划分（分层抽样，test_size={TEST_SIZE}）：")
print(f"  训练集：{X_train.shape[0]} 样本  |  测试集：{X_test.shape[0]} 样本")

# ─── 特征标准化 ───
scaler = StandardScaler()
# 只对连续型数值特征标准化，不对 0/1 编码特征标准化
continuous_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
binary_encoded_cols = [c for c in X.columns if c not in continuous_cols]
print(f"\n需要标准化的连续型特征：{continuous_cols}")
print(f"不需要标准化的编码特征（共 {len(binary_encoded_cols)} 个）已保持原值")

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

# ─── 保存预处理后数据快照 ───
X_train_scaled.to_csv(os.path.join(OUTPUT_DIR, 'X_train_preprocessed.csv'), index=False)
X_test_scaled.to_csv(os.path.join(OUTPUT_DIR, 'X_test_preprocessed.csv'), index=False)
print("\n  → 预处理后训练/测试集已保存")


# =============================================================================
# 四、变量筛选 (Feature Selection)
# =============================================================================
print("\n" + "─" * 70)
print("【第四步】变量筛选")
print("─" * 70)

feature_names = list(X_train_scaled.columns)

# ─── 4.1 方差过滤 ───
print("\n【4.1 方差过滤】")
variances = X_train_scaled.var()
low_var_threshold = 0.01
low_var_features = variances[variances < low_var_threshold].index.tolist()
print(f"  方差阈值：{low_var_threshold}")
if low_var_features:
    print(f"  低方差特征（建议删除）：{low_var_features}")
else:
    print(f"  无低方差特征，所有特征方差均 ≥ {low_var_threshold}")
print(f"  各特征方差：\n{variances.sort_values().round(4)}")

# ─── 4.2 多重共线性分析（VIF）───
print("\n【4.2 多重共线性分析（VIF）】")
# 只对连续型特征做 VIF
vif_cols = continuous_cols
X_vif = X_train_scaled[vif_cols].copy()
X_vif = sm.add_constant(X_vif)
vif_data = pd.DataFrame()
vif_data['特征'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i)
                   for i in range(X_vif.shape[1])]
vif_data = vif_data[vif_data['特征'] != 'const']
vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
print("  VIF结果（VIF > 10 表示存在严重多重共线性）：")
print(vif_data)
high_vif_features = vif_data[vif_data['VIF'] > 10]['特征'].tolist()
if high_vif_features:
    print(f"  → 高 VIF 特征：{high_vif_features}，建议进一步处理")
else:
    print(f"  → 所有连续特征 VIF < 10，无严重多重共线性问题")

# VIF 可视化
fig, ax = plt.subplots(figsize=(8, 5))
colors_vif = ['tomato' if v > 10 else ('gold' if v > 5 else 'steelblue')
               for v in vif_data['VIF']]
bars = ax.barh(vif_data['特征'], vif_data['VIF'], color=colors_vif, edgecolor='white')
ax.axvline(x=5, color='orange', linestyle='--', linewidth=1.5, label='VIF=5（警告线）')
ax.axvline(x=10, color='red', linestyle='--', linewidth=1.5, label='VIF=10（危险线）')
ax.set_title('方差膨胀因子（VIF）', fontsize=13, fontweight='bold')
ax.set_xlabel('VIF 值')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_vif_analysis.png'))
plt.close()
print("  → 图表已保存：06_vif_analysis.png")

# ─── 4.3 单变量统计筛选（ANOVA F值 & 互信息）───
print("\n【4.3 单变量统计筛选（ANOVA F值 & 互信息）】")
# F 检验
selector_f = SelectKBest(f_classif, k='all')
selector_f.fit(X_train_scaled, y_train)
f_scores = selector_f.scores_
f_pvalues = selector_f.pvalues_

# 互信息
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=RANDOM_STATE)

# 整合结果
feat_stat_df = pd.DataFrame({
    '特征': feature_names,
    'F值': f_scores.round(4),
    'F值_p': f_pvalues.round(6),
    '互信息得分': mi_scores.round(6),
    'F值显著': (f_pvalues < 0.05).astype(int)
}).sort_values('互信息得分', ascending=False).reset_index(drop=True)
print(feat_stat_df.to_string())

# 双轴对比可视化
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
top_n = min(20, len(feature_names))
df_top = feat_stat_df.head(top_n)

axes[0].barh(df_top['特征'][::-1], df_top['F值'][::-1],
             color=sns.color_palette("Blues_d", top_n), edgecolor='white')
axes[0].set_title('ANOVA F值（特征 vs 目标）', fontsize=12, fontweight='bold')
axes[0].set_xlabel('F 统计量')

axes[1].barh(df_top['特征'][::-1], df_top['互信息得分'][::-1],
             color=sns.color_palette("Greens_d", top_n), edgecolor='white')
axes[1].set_title('互信息得分（特征 vs 目标）', fontsize=12, fontweight='bold')
axes[1].set_xlabel('互信息得分')

plt.suptitle('单变量特征重要性评估', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_univariate_feature_selection.png'))
plt.close()
print("  → 图表已保存：07_univariate_feature_selection.png")

# ─── 4.4 基于随机森林的特征重要性（初步）───
print("\n【4.4 随机森林特征重要性（初步筛选）】")
rf_for_fs = RandomForestClassifier(
    n_estimators=200, max_depth=None, min_samples_split=2,
    min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1
)
rf_for_fs.fit(X_train_scaled, y_train)
rf_importances = rf_for_fs.feature_importances_
rf_feat_df = pd.DataFrame({
    '特征': feature_names,
    '重要性': rf_importances
}).sort_values('重要性', ascending=False).reset_index(drop=True)
print("  随机森林特征重要性排名（前20）：")
print(rf_feat_df.head(20).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
top_rf = rf_feat_df.head(20)
colors_rf = sns.color_palette("YlOrRd_r", len(top_rf))
bars_rf = ax.barh(top_rf['特征'][::-1], top_rf['重要性'][::-1],
                   color=colors_rf[::-1], edgecolor='white')
ax.set_title('随机森林特征重要性排名（Top 20）', fontsize=13, fontweight='bold')
ax.set_xlabel('重要性（Mean Decrease Impurity）')
for bar, val in zip(bars_rf, top_rf['重要性'][::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_rf_feature_importance.png'))
plt.close()
print("  → 图表已保存：08_rf_feature_importance.png")

# ─── 4.5 基于 XGBoost 的特征重要性 ───
print("\n【4.5 XGBoost 特征重要性】")
xgb_for_fs = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=RANDOM_STATE, n_jobs=-1
)
xgb_for_fs.fit(X_train_scaled, y_train)
xgb_importances = xgb_for_fs.feature_importances_
xgb_feat_df = pd.DataFrame({
    '特征': feature_names,
    '重要性': xgb_importances
}).sort_values('重要性', ascending=False).reset_index(drop=True)
print("  XGBoost 特征重要性排名（前20）：")
print(xgb_feat_df.head(20).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
top_xgb = xgb_feat_df.head(20)
colors_xgb = sns.color_palette("PuRd_r", len(top_xgb))
bars_xgb = ax.barh(top_xgb['特征'][::-1], top_xgb['重要性'][::-1],
                    color=colors_xgb[::-1], edgecolor='white')
ax.set_title('XGBoost 特征重要性排名（Top 20）', fontsize=13, fontweight='bold')
ax.set_xlabel('特征重要性')
for bar, val in zip(bars_xgb, top_xgb['重要性'][::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_xgb_feature_importance.png'))
plt.close()
print("  → 图表已保存：09_xgb_feature_importance.png")

# ─── 4.6 递归特征消除（RFE）───
print("\n【4.6 递归特征消除（RFE）】")
rfe_estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
n_features_to_select = max(5, len(feature_names) // 2)
rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_features_to_select, step=1)
rfe.fit(X_train_scaled, y_train)
rfe_support = rfe.support_
rfe_ranking = rfe.ranking_
rfe_result_df = pd.DataFrame({
    '特征': feature_names,
    'RFE支持': rfe_support,
    'RFE排名': rfe_ranking
}).sort_values('RFE排名').reset_index(drop=True)
print(f"  RFE 选择特征数：{n_features_to_select}")
print("  RFE 结果：")
print(rfe_result_df.to_string())
rfe_selected_features = rfe_result_df[rfe_result_df['RFE支持'] == True]['特征'].tolist()
print(f"\n  RFE 最终选择的 {len(rfe_selected_features)} 个特征：")
print(f"  {rfe_selected_features}")

# ─── 综合特征筛选结论 ───
print("\n【综合特征筛选结论】")
# 以 RF + XGBoost + 互信息 三种方法综合排名
rf_ranks = {row['特征']: idx + 1 for idx, row in rf_feat_df.iterrows()}
xgb_ranks = {row['特征']: idx + 1 for idx, row in xgb_feat_df.iterrows()}
mi_ranks = {row['特征']: idx + 1 for idx, row in feat_stat_df.sort_values('互信息得分', ascending=False).reset_index(drop=True).iterrows()}

综合排名 = {}
for feat in feature_names:
    综合排名[feat] = (rf_ranks.get(feat, len(feature_names)) +
                   xgb_ranks.get(feat, len(feature_names)) +
                   mi_ranks.get(feat, len(feature_names))) / 3

综合df = pd.DataFrame({'特征': list(综合排名.keys()),
                       '综合排名均值': list(综合排名.values())})
综合df = 综合df.sort_values('综合排名均值').reset_index(drop=True)
print("  三方法综合排名（越小越重要）：")
print(综合df.to_string())

# 选取排名靠前的特征（综合排名 ≤ 所有特征数的 60%）
cutoff = len(feature_names) * 0.6
selected_features = 综合df[综合df['综合排名均值'] <= cutoff]['特征'].tolist()
# 保底：至少保留 RFE 选中的特征
selected_features = list(set(selected_features + rfe_selected_features))
print(f"\n  → 特征筛选分析选定 {len(selected_features)} 个重要特征：")
print(f"  {selected_features}")

# 用选定特征重构训练/测试集（用于特征筛选效果对比）
X_train_sel = X_train_scaled[selected_features]
X_test_sel = X_test_scaled[selected_features]
print(f"\n  → 特征筛选后训练集形状：{X_train_sel.shape}")
print(f"  → 特征筛选后测试集形状：{X_test_sel.shape}")


# =============================================================================
# 五、多模型训练与对比 (Model Comparison)
# =============================================================================
print("\n" + "─" * 70)
print("【第五步】多模型训练与对比")
print("─" * 70)

# 定义参与对比的模型（使用特征筛选后的特征，保证训练和预测一致性）
models_dict = {
    '逻辑回归 (LR)': LogisticRegression(
        max_iter=2000, solver='lbfgs', multi_class='multinomial',
        random_state=RANDOM_STATE, C=1.0
    ),
    '决策树 (DT)': DecisionTreeClassifier(
        max_depth=None, min_samples_split=2, min_samples_leaf=1,
        random_state=RANDOM_STATE
    ),
    '随机森林 (RF)': RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1
    ),
    '极端随机树 (ET)': ExtraTreesClassifier(
        n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    ),
    '梯度提升树 (GBT)': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5,
        random_state=RANDOM_STATE
    ),
    'XGBoost (XGB)': xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    '支持向量机 (SVM)': SVC(
        kernel='rbf', C=1.0, gamma='scale',
        probability=True, random_state=RANDOM_STATE
    ),
    'K近邻 (KNN)': KNeighborsClassifier(
        n_neighbors=7, weights='distance', algorithm='auto', n_jobs=-1
    ),
    '朴素贝叶斯 (NB)': GaussianNB()
    # 注：AdaBoost 在本数据集上表现极差（准确率<30%），已从对比中移除
}

# ─── 5折交叉验证 + 测试集评估 ───
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\n开始训练各模型（5折交叉验证）...\n")
results = {}
cv_scores_all = {}

for model_name, model in models_dict.items():
    print(f"  >> 训练 {model_name}...")

    # 5折交叉验证（在训练集上，使用筛选后的特征）
    cv_scores = cross_val_score(
        model, X_train_sel, y_train,
        cv=cv, scoring='accuracy', n_jobs=-1
    )
    cv_scores_all[model_name] = cv_scores

    # 在完整训练集上拟合（使用筛选后的特征）
    model.fit(X_train_sel, y_train)

    # 测试集预测（使用筛选后的特征）
    y_pred = model.predict(X_test_sel)
    y_pred_proba = model.predict_proba(X_test_sel)

    # 计算各项指标
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    # 多分类 ROC-AUC（OVR）
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    except Exception:
        auc_score = np.nan

    results[model_name] = {
        '5折CV均值': round(cv_scores.mean(), 4),
        '5折CV标准差': round(cv_scores.std(), 4),
        '测试集准确率': round(acc, 4),
        '测试集F1(宏)': round(f1_macro, 4),
        '测试集F1(加权)': round(f1_weighted, 4),
        '测试集精确率(加权)': round(precision, 4),
        '测试集召回率(加权)': round(recall, 4),
        '测试集AUC(OVR)': round(auc_score, 4) if not np.isnan(auc_score) else 'N/A'
    }
    auc_str = f"{auc_score:.4f}" if not np.isnan(auc_score) else "N/A"
    print(f"    CV准确率：{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}  "
          f"测试准确率：{acc:.4f}  AUC：{auc_str}")

# ─── 结果汇总表 ───
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('测试集准确率', ascending=False)
print("\n【模型对比汇总表】")
print(results_df.to_string())
results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison_results.csv'), encoding='utf-8-sig')

# ─── 可视化：测试集准确率对比 ───
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
model_labels = list(results_df.index)
acc_values = results_df['测试集准确率'].astype(float).values
cv_means = results_df['5折CV均值'].astype(float).values
cv_stds = results_df['5折CV标准差'].astype(float).values

x = np.arange(len(model_labels))
width = 0.35
colors_test = sns.color_palette("coolwarm_r", len(model_labels))

bars1 = axes[0].bar(x - width / 2, cv_means, width, label='5折CV均值',
                    color=sns.color_palette("Blues_d", len(model_labels)),
                    yerr=cv_stds, capsize=4, ecolor='gray')
bars2 = axes[0].bar(x + width / 2, acc_values, width, label='测试集准确率',
                    color=sns.color_palette("Reds_d", len(model_labels)))
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_labels, rotation=30, ha='right', fontsize=9)
axes[0].set_ylabel('准确率')
axes[0].set_ylim(0.5, 1.05)
axes[0].set_title('各模型准确率对比（CV vs 测试集）', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, acc_values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# F1 对比
f1_macro_vals = results_df['测试集F1(宏)'].astype(float).values
f1_weighted_vals = results_df['测试集F1(加权)'].astype(float).values
bars3 = axes[1].bar(x - width / 2, f1_macro_vals, width, label='F1(宏平均)',
                    color=sns.color_palette("Purples_d", len(model_labels)))
bars4 = axes[1].bar(x + width / 2, f1_weighted_vals, width, label='F1(加权平均)',
                    color=sns.color_palette("Greens_d", len(model_labels)))
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_labels, rotation=30, ha='right', fontsize=9)
axes[1].set_ylabel('F1 Score')
axes[1].set_ylim(0.5, 1.05)
axes[1].set_title('各模型 F1 Score 对比', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('多模型性能对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '10_model_comparison.png'))
plt.close()
print("\n  → 图表已保存：10_model_comparison.png")

# ─── 交叉验证箱线图 ───
fig, ax = plt.subplots(figsize=(14, 6))
cv_data = [cv_scores_all[m] for m in models_dict.keys()]
model_labels_full = list(models_dict.keys())
bp = ax.boxplot(cv_data, patch_artist=True, notch=False,
                medianprops={'color': 'black', 'linewidth': 2})
for patch, color in zip(bp['boxes'], sns.color_palette("tab10", len(model_labels_full))):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_xticklabels(model_labels_full, rotation=25, ha='right', fontsize=9)
ax.set_ylabel('5折CV 准确率')
ax.set_title('各模型 5折交叉验证准确率分布（箱线图）', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '11_cv_boxplot.png'))
plt.close()
print("  → 图表已保存：11_cv_boxplot.png")

# ─── 确定最优模型 ───
best_model_name = results_df.index[0]
print(f"\n【最优模型确认】")
print(f"  → 综合测试集准确率最高：{best_model_name}")
print(f"  → 测试集准确率：{results_df.loc[best_model_name, '测试集准确率']}")
print(f"  → 5折CV均值：{results_df.loc[best_model_name, '5折CV均值']}")


# =============================================================================
# 六、最优模型详细评估（调参前）
# =============================================================================
print("\n" + "─" * 70)
print(f"【第六步】最优模型详细评估（{best_model_name}，调参前）")
print("─" * 70)

best_model_before = models_dict[best_model_name]
best_model_before.fit(X_train_sel, y_train)
y_pred_before = best_model_before.predict(X_test_sel)

print("\n分类报告（调参前）：")
print(classification_report(y_test, y_pred_before, target_names=class_names))

# 混淆矩阵
cm_before = confusion_matrix(y_test, y_pred_before)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_before, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', ax=ax)
ax.set_title(f'混淆矩阵 - {best_model_name}（调参前）', fontsize=13, fontweight='bold')
ax.set_xlabel('预测标签')
ax.set_ylabel('真实标签')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'12_confusion_matrix_before_tuning.png'))
plt.close()
print("  → 图表已保存：12_confusion_matrix_before_tuning.png")


# =============================================================================
# 七、超参数调优 (Hyperparameter Tuning)
# =============================================================================
print("\n" + "─" * 70)
print(f"【第七步】超参数调优（{best_model_name}）")
print("─" * 70)

# 根据最优模型类型选择调参策略
if 'XGBoost' in best_model_name or 'XGB' in best_model_name:
    print("\n  → 对 XGBoost 进行 RandomizedSearchCV 超参数调优")
    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.5],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0.1, 1, 5, 10]
    }
    tuning_model = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    searcher = RandomizedSearchCV(
        tuning_model, param_distributions,
        n_iter=80, cv=5, scoring='accuracy',
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
        refit=True
    )

elif '随机森林' in best_model_name or 'RF' in best_model_name:
    print("\n  → 对随机森林进行 RandomizedSearchCV 超参数调优")
    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500, 600],
        'max_depth': [None, 5, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced']
    }
    tuning_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    searcher = RandomizedSearchCV(
        tuning_model, param_distributions,
        n_iter=80, cv=5, scoring='accuracy',
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
        refit=True
    )

elif '极端随机树' in best_model_name or 'ET' in best_model_name:
    print("\n  → 对极端随机树进行 RandomizedSearchCV 超参数调优")
    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 5, 10, 15, 20, 25],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 3, 4],
        'max_features': ['sqrt', 'log2', None, 0.5, 0.7]
    }
    tuning_model = ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    searcher = RandomizedSearchCV(
        tuning_model, param_distributions,
        n_iter=60, cv=5, scoring='accuracy',
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
        refit=True
    )

elif '梯度提升树' in best_model_name or 'GBT' in best_model_name:
    print("\n  → 对梯度提升树进行 RandomizedSearchCV 超参数调优")
    param_distributions = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 3],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    tuning_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    searcher = RandomizedSearchCV(
        tuning_model, param_distributions,
        n_iter=60, cv=5, scoring='accuracy',
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
        refit=True
    )

elif 'SVM' in best_model_name or '支持向量机' in best_model_name:
    print("\n  → 对 SVM 进行 GridSearchCV 超参数调优")
    param_grid = {
        'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly']
    }
    tuning_model = SVC(probability=True, random_state=RANDOM_STATE)
    searcher = GridSearchCV(
        tuning_model, param_grid,
        cv=5, scoring='accuracy',
        n_jobs=-1, verbose=1, refit=True
    )

else:
    # 通用逻辑回归调参
    print("\n  → 对逻辑回归进行 GridSearchCV 超参数调优")
    param_grid = {
        'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
        'solver': ['lbfgs', 'saga'],
        'penalty': ['l2']
    }
    tuning_model = LogisticRegression(
        max_iter=3000, multi_class='multinomial', random_state=RANDOM_STATE
    )
    searcher = GridSearchCV(
        tuning_model, param_grid,
        cv=5, scoring='accuracy',
        n_jobs=-1, verbose=1, refit=True
    )

print("\n  开始搜索最优超参数，请稍候...")
searcher.fit(X_train_sel, y_train)

print(f"\n  最优超参数组合：{searcher.best_params_}")
print(f"  最优 CV 准确率：{searcher.best_score_:.4f}")

# ─── 调参后测试集评估 ───
best_model_tuned = searcher.best_estimator_
y_pred_tuned = best_model_tuned.predict(X_test_sel)
y_pred_proba_tuned = best_model_tuned.predict_proba(X_test_sel)

acc_before = accuracy_score(y_test, y_pred_before)
acc_tuned = accuracy_score(y_test, y_pred_tuned)
f1_before = f1_score(y_test, y_pred_before, average='weighted', zero_division=0)
f1_tuned = f1_score(y_test, y_pred_tuned, average='weighted', zero_division=0)

try:
    auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned, multi_class='ovr', average='macro')
except:
    auc_tuned = np.nan

print(f"\n【调参效果对比】")
print(f"  调参前  测试集准确率：{acc_before:.4f}  F1(加权)：{f1_before:.4f}")
print(f"  调参后  测试集准确率：{acc_tuned:.4f}  F1(加权)：{f1_tuned:.4f}")
print(f"  准确率提升：{(acc_tuned - acc_before) * 100:.2f}%")
print(f"  调参后 AUC(OVR)：{auc_tuned:.4f}")

# ─── 额外：用全特征训练Web端模型（确保特征一致性）───
print("\n" + "─" * 70)
print("【额外】用全部15个特征训练Web端模型（确保训练/预测一致性）")
print("─" * 70)

# 使用相同的超参数，在全特征上训练
best_params_full = searcher.best_params_.copy()
if 'XGBoost' in best_model_name or 'XGB' in best_model_name:
    web_model = xgb.XGBClassifier(
        **best_params_full,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=RANDOM_STATE, n_jobs=-1
    )
else:
    web_model = RandomForestClassifier(
        **best_params_full, random_state=RANDOM_STATE, n_jobs=-1
    )

web_model.fit(X_train_scaled, y_train)
y_pred_web = web_model.predict(X_test_scaled)
y_pred_proba_web = web_model.predict_proba(X_test_scaled)

acc_web = accuracy_score(y_test, y_pred_web)
f1_web = f1_score(y_test, y_pred_web, average='weighted', zero_division=0)
try:
    auc_web = roc_auc_score(y_test, y_pred_proba_web, multi_class='ovr', average='macro')
except:
    auc_web = np.nan

print(f"\n  全特征模型（Web端用）性能：")
print(f"    特征数：{X_train_scaled.shape[1]} 个")
print(f"    测试集准确率：{acc_web:.4f}  F1(加权)：{f1_web:.4f}  AUC：{auc_web:.4f}")
print(f"    vs 筛选特征模型：准确率差值 {(acc_web - acc_tuned)*100:+.2f}%")

print("\n分类报告（调参后）：")
print(classification_report(y_test, y_pred_tuned, target_names=class_names))

# ─── 调参后混淆矩阵 ───
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(cm_before, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', ax=axes[0])
axes[0].set_title(f'混淆矩阵 - 调参前\n准确率={acc_before:.4f}', fontsize=12, fontweight='bold')
axes[0].set_xlabel('预测标签')
axes[0].set_ylabel('真实标签')
axes[0].tick_params(axis='x', rotation=30)

sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', ax=axes[1])
axes[1].set_title(f'混淆矩阵 - 调参后\n准确率={acc_tuned:.4f}', fontsize=12, fontweight='bold')
axes[1].set_xlabel('预测标签')
axes[1].set_ylabel('真实标签')
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle(f'{best_model_name} 调参前后混淆矩阵对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '13_confusion_matrix_comparison.png'))
plt.close()
print("  → 图表已保存：13_confusion_matrix_comparison.png")

# ─── 多分类 ROC 曲线（OvR）───
n_classes = len(class_names)
y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes))
fig, ax = plt.subplots(figsize=(10, 8))
color_roc = sns.color_palette("tab10", n_classes)
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba_tuned[:, i])
    roc_auc_i = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color_roc[i], linewidth=2,
            label=f'{class_names[i]} (AUC={roc_auc_i:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='随机分类基准')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.set_xlabel('假阳性率（FPR）', fontsize=12)
ax.set_ylabel('真阳性率（TPR）', fontsize=12)
ax.set_title(f'{best_model_name}（调参后）多分类 ROC 曲线', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '14_roc_curves_multiclass.png'))
plt.close()
print("  → 图表已保存：14_roc_curves_multiclass.png")

# ─── 调参结果曲线（如果是 RandomizedSearch，绘制CV得分分布）───
cv_results_df = pd.DataFrame(searcher.cv_results_)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(len(cv_results_df)), cv_results_df['mean_test_score'],
        'o-', color='steelblue', markersize=3, linewidth=1, label='CV均值准确率')
ax.fill_between(range(len(cv_results_df)),
                cv_results_df['mean_test_score'] - cv_results_df['std_test_score'],
                cv_results_df['mean_test_score'] + cv_results_df['std_test_score'],
                alpha=0.3, color='steelblue')
best_idx = cv_results_df['mean_test_score'].idxmax()
ax.axvline(x=best_idx, color='red', linestyle='--', linewidth=2, label=f'最优参数（Idx={best_idx}）')
ax.scatter(best_idx, cv_results_df.loc[best_idx, 'mean_test_score'],
           color='red', s=100, zorder=5)
ax.set_xlabel('参数组合索引')
ax.set_ylabel('CV 准确率')
ax.set_title('超参数搜索过程（所有候选参数CV得分）', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '15_hyperparameter_search_curve.png'))
plt.close()
print("  → 图表已保存：15_hyperparameter_search_curve.png")


# =============================================================================
# 八、最优模型特征重要性（调参后）
# =============================================================================
print("\n" + "─" * 70)
print("【第八步】最优模型特征重要性（调参后）")
print("─" * 70)

if hasattr(best_model_tuned, 'feature_importances_'):
    imp_tuned = best_model_tuned.feature_importances_
    # 使用筛选后的特征名
    feat_imp_tuned_df = pd.DataFrame({
        '特征': selected_features,
        '重要性': imp_tuned
    }).sort_values('重要性', ascending=False).reset_index(drop=True)
    print("  调参后模型特征重要性（前15）：")
    print(feat_imp_tuned_df.head(15).to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    top_tuned = feat_imp_tuned_df.head(15)
    colors_tuned = sns.color_palette("rocket_r", len(top_tuned))
    ax.barh(top_tuned['特征'][::-1], top_tuned['重要性'][::-1],
            color=colors_tuned[::-1], edgecolor='white')
    ax.set_title(f'{best_model_name}（调参后）特征重要性 Top15', fontsize=13, fontweight='bold')
    ax.set_xlabel('特征重要性')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '16_tuned_model_feature_importance.png'))
    plt.close()
    print("  → 图表已保存：16_tuned_model_feature_importance.png")
else:
    # 使用置换重要性（model-agnostic）
    print("  使用置换重要性（Permutation Importance）评估...")
    perm_imp = permutation_importance(
        best_model_tuned, X_test_sel, y_test,
        n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1
    )
    # 使用筛选后的特征名
    perm_imp_df = pd.DataFrame({
        '特征': selected_features,
        '均值': perm_imp.importances_mean,
        '标准差': perm_imp.importances_std
    }).sort_values('均值', ascending=False).reset_index(drop=True)
    print("  置换重要性（前15）：")
    print(perm_imp_df.head(15).to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    top_perm = perm_imp_df.head(15)
    ax.barh(top_perm['特征'][::-1], top_perm['均值'][::-1],
            xerr=top_perm['标准差'][::-1],
            color=sns.color_palette("mako_r", len(top_perm))[::-1],
            edgecolor='white', capsize=4, ecolor='gray')
    ax.set_title(f'{best_model_name}（调参后）置换特征重要性 Top15', fontsize=13, fontweight='bold')
    ax.set_xlabel('准确率下降均值')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '16_tuned_model_permutation_importance.png'))
    plt.close()
    print("  → 图表已保存：16_tuned_model_permutation_importance.png")


# =============================================================================
# 九、最终结果汇总
# =============================================================================
print("\n" + "=" * 70)
print("  最终分析结果汇总")
print("=" * 70)

print(f"\n  数据集规模：{df_raw.shape[0]} 个样本，{df_raw.shape[1]} 个特征（含目标变量）")
print(f"  目标类别数：{n_classes} 类")
print(f"  特征筛选后保留：{len(selected_features)} 个特征")
print(f"\n  最优模型：{best_model_name}")
print(f"  最优超参数：{searcher.best_params_}")
print(f"  调参前测试集准确率：{acc_before:.4f} ({acc_before * 100:.2f}%)")
print(f"  调参后测试集准确率：{acc_tuned:.4f} ({acc_tuned * 100:.2f}%)")
print(f"  准确率提升：{(acc_tuned - acc_before) * 100:.2f} 个百分点")
print(f"  调参后 F1(加权)：{f1_tuned:.4f}")
print(f"  调参后 AUC(OVR)：{auc_tuned:.4f}")

# 保存两个版本的模型
# 1. 筛选特征模型（用于论文分析）
model_sel_path = os.path.join(OUTPUT_DIR, 'best_model_selected.joblib')
joblib.dump(best_model_tuned, model_sel_path)
print(f"\n  → 筛选特征模型已保存：{model_sel_path} ({len(selected_features)}个特征)")

# 2. 全特征模型（用于Web端预测）
model_full_path = os.path.join(OUTPUT_DIR, 'best_model_tuned.joblib')
joblib.dump(web_model, model_full_path)
print(f"  → 全特征模型已保存：{model_full_path} ({len(feature_names)}个特征)")

# 保存标准化器和标签编码器
scaler_save_path = os.path.join(OUTPUT_DIR, 'scaler.joblib')
le_save_path = os.path.join(OUTPUT_DIR, 'label_encoder.joblib')
joblib.dump(scaler, scaler_save_path)
joblib.dump(le, le_save_path)
print(f"  → 标准化器已保存：{scaler_save_path}")
print(f"  → 标签编码器已保存：{le_save_path}")

# 保存最终特征信息
feature_info = {
    'all_features': feature_names,
    'selected_features': selected_features,
    'continuous_cols': continuous_cols,
    'best_model': best_model_name,
    'best_params': searcher.best_params_,
    'test_accuracy_before': float(acc_before),
    'test_accuracy_after': float(acc_tuned),
    'test_accuracy_full_features': float(acc_web),
    'class_names': list(class_names)
}
import json
with open(os.path.join(OUTPUT_DIR, 'feature_info.json'), 'w', encoding='utf-8') as f:
    json.dump(feature_info, f, ensure_ascii=False, indent=2)
print(f"  → 特征信息已保存：feature_info.json")

print("\n" + "─" * 70)
print("  所有分析图表均已保存至：", OUTPUT_DIR)
print("─" * 70)

print("\n完整数据分析流程运行完毕！")
