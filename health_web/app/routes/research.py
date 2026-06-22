# -*- coding: utf-8 -*-
"""
research.py - 数据分析展示板块（研究者/答辩专用）

展示 obesity_analysis.py 生成的所有图表和分析结果
仅供研究人员和毕设答辩使用
"""

import os
import json
from flask import Blueprint, render_template, current_app, send_from_directory

research_bp = Blueprint("research", __name__, url_prefix="/research")


@research_bp.route("/")
def index():
    """数据分析展示首页"""
    analysis_dir = os.path.join(current_app.root_path, "..", "..", "analysis_output")
    analysis_dir = os.path.abspath(analysis_dir)
    
    # 加载特征信息
    feature_info = {}
    feature_info_path = os.path.join(analysis_dir, "feature_info.json")
    if os.path.exists(feature_info_path):
        with open(feature_info_path, "r", encoding="utf-8") as f:
            feature_info = json.load(f)
    
    # 计算真正的筛选后特征（基于特征重要性排名）
    all_features = feature_info.get('all_features', [])
    selected_features = _calculate_selected_features(analysis_dir, all_features)
    feature_info['selected_features'] = selected_features
    
    # 加载模型对比结果（正确处理模型名称列）
    model_comparison = []
    comparison_path = os.path.join(analysis_dir, "model_comparison_results.csv")
    if os.path.exists(comparison_path):
        import pandas as pd
        df = pd.read_csv(comparison_path, encoding="utf-8-sig", index_col=0)
        df = df.reset_index()
        df.rename(columns={'index': '模型'}, inplace=True)
        # 按测试集准确率排序
        df = df.sort_values('测试集准确率', ascending=False)
        df['排名'] = range(1, len(df) + 1)
        model_comparison = df.to_dict("records")
    
    # 图表列表
    charts = [
        {"id": "01", "name": "目标变量分布", "file": "01_target_distribution.png", 
         "desc": "肥胖等级分布的频数和占比"},
        {"id": "02", "name": "数值型特征分布", "file": "02_numerical_distribution.png",
         "desc": "各数值特征的直方图和KDE曲线"},
        {"id": "03", "name": "数值特征箱线图", "file": "03_boxplot_by_target.png",
         "desc": "数值特征按肥胖等级的箱线对比"},
        {"id": "04", "name": "类别型特征分布", "file": "04_categorical_distribution.png",
         "desc": "各分类特征的频率分布"},
        {"id": "05", "name": "相关性热力图", "file": "05_correlation_heatmap.png",
         "desc": "数值特征间的Pearson相关系数"},
        {"id": "06", "name": "VIF分析", "file": "06_vif_analysis.png",
         "desc": "方差膨胀因子（多重共线性检测）"},
        {"id": "07", "name": "单变量特征筛选", "file": "07_univariate_feature_selection.png",
         "desc": "ANOVA F值和互信息得分"},
        {"id": "08", "name": "随机森林特征重要性", "file": "08_rf_feature_importance.png",
         "desc": "基于随机森林的特征重要性排名"},
        {"id": "09", "name": "XGBoost特征重要性", "file": "09_xgb_feature_importance.png",
         "desc": "基于XGBoost的特征重要性排名"},
        {"id": "10", "name": "模型性能对比", "file": "10_model_comparison.png",
         "desc": "各模型的准确率和F1分数对比"},
        {"id": "11", "name": "交叉验证箱线图", "file": "11_cv_boxplot.png",
         "desc": "5折交叉验证准确率分布"},
        {"id": "12", "name": "调参前混淆矩阵", "file": "12_confusion_matrix_before_tuning.png",
         "desc": "最优模型调参前的混淆矩阵"},
        {"id": "13", "name": "调参前后对比", "file": "13_confusion_matrix_comparison.png",
         "desc": "超参数调优前后的混淆矩阵对比"},
        {"id": "14", "name": "多分类ROC曲线", "file": "14_roc_curves_multiclass.png",
         "desc": "各类别的ROC曲线和AUC值"},
        {"id": "15", "name": "超参数搜索过程", "file": "15_hyperparameter_search_curve.png",
         "desc": "RandomizedSearchCV的搜索过程"},
        {"id": "16", "name": "调参后特征重要性", "file": "16_tuned_model_feature_importance.png",
         "desc": "调参后模型的特征重要性Top15"},
    ]
    
    # 检查哪些图表存在
    available_charts = []
    for chart in charts:
        chart_path = os.path.join(analysis_dir, chart["file"])
        chart["exists"] = os.path.exists(chart_path)
        available_charts.append(chart)
    
    return render_template("research/index.html",
                         feature_info=feature_info,
                         model_comparison=model_comparison,
                         charts=available_charts)


@research_bp.route("/chart/<filename>")
def chart(filename):
    """提供图表文件"""
    analysis_dir = os.path.join(current_app.root_path, "..", "..", "analysis_output")
    analysis_dir = os.path.abspath(analysis_dir)
    return send_from_directory(analysis_dir, filename)


@research_bp.route("/feature-analysis")
def feature_analysis():
    """特征分析详情页"""
    analysis_dir = os.path.join(current_app.root_path, "..", "..", "analysis_output")
    analysis_dir = os.path.abspath(analysis_dir)
    
    # 加载特征信息
    feature_info = {}
    feature_info_path = os.path.join(analysis_dir, "feature_info.json")
    if os.path.exists(feature_info_path):
        with open(feature_info_path, "r", encoding="utf-8") as f:
            feature_info = json.load(f)
    
    # 加载预处理后的数据样本
    X_train_sample = []
    X_train_path = os.path.join(analysis_dir, "X_train_preprocessed.csv")
    if os.path.exists(X_train_path):
        import pandas as pd
        df = pd.read_csv(X_train_path)
        X_train_sample = df.head(10).to_dict("records")
    
    return render_template("research/feature_analysis.html",
                         feature_info=feature_info,
                         X_train_sample=X_train_sample)


@research_bp.route("/model-comparison")
def model_comparison_detail():
    """模型对比详情页"""
    analysis_dir = os.path.join(current_app.root_path, "..", "..", "analysis_output")
    analysis_dir = os.path.abspath(analysis_dir)
    
    # 加载模型对比结果（正确处理模型名称列）
    model_comparison = []
    comparison_path = os.path.join(analysis_dir, "model_comparison_results.csv")
    if os.path.exists(comparison_path):
        import pandas as pd
        df = pd.read_csv(comparison_path, encoding="utf-8-sig", index_col=0)
        df = df.reset_index()
        df.rename(columns={'index': '模型'}, inplace=True)
        # 按测试集准确率排序
        df = df.sort_values('测试集准确率', ascending=False)
        df['排名'] = range(1, len(df) + 1)
        model_comparison = df.to_dict("records")
    
    # 加载特征信息获取最佳模型
    feature_info = {}
    feature_info_path = os.path.join(analysis_dir, "feature_info.json")
    if os.path.exists(feature_info_path):
        with open(feature_info_path, "r", encoding="utf-8") as f:
            feature_info = json.load(f)
    
    # 为图表准备数据
    chart_data = {
        'models': [m['模型'] for m in model_comparison],
        'cv_scores': [m['5折CV均值'] for m in model_comparison],
        'test_scores': [m['测试集准确率'] for m in model_comparison],
        'f1_scores': [m['测试集F1(加权)'] for m in model_comparison],
        'auc_scores': [m['测试集AUC(OVR)'] for m in model_comparison]
    }
    
    return render_template("research/model_comparison.html",
                         model_comparison=model_comparison,
                         feature_info=feature_info,
                         chart_data=chart_data)


@research_bp.route("/dataset-info")
def dataset_info():
    """数据集信息页"""
    analysis_dir = os.path.join(current_app.root_path, "..", "..", "analysis_output")
    analysis_dir = os.path.abspath(analysis_dir)
    
    # 加载预处理后数据的统计信息
    stats = {}
    X_train_path = os.path.join(analysis_dir, "X_train_preprocessed.csv")
    if os.path.exists(X_train_path):
        import pandas as pd
        df = pd.read_csv(X_train_path)
        stats = {
            "train_samples": len(df),
            "features": list(df.columns),
            "feature_count": len(df.columns),
            "numeric_summary": df.describe().to_dict()
        }
    
    X_test_path = os.path.join(analysis_dir, "X_test_preprocessed.csv")
    if os.path.exists(X_test_path):
        import pandas as pd
        df = pd.read_csv(X_test_path)
        stats["test_samples"] = len(df)
    
    return render_template("research/dataset_info.html", stats=stats)


def _calculate_selected_features(analysis_dir, all_features):
    """
    基于模型特征重要性计算筛选后的特征列表
    从训练好的XGBoost模型中提取特征重要性，取前60%
    """
    try:
        import joblib
        import numpy as np
        
        # 尝试从模型文件中提取特征重要性
        model_path = os.path.join(analysis_dir, "best_model_tuned.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # 获取特征重要性
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # 按重要性排序
                feature_importance_pairs = list(zip(all_features, importances))
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # 取前60%的特征
                cutoff = int(len(all_features) * 0.6)
                cutoff = max(cutoff, 8)  # 至少保留8个
                
                selected = [f for f, _ in feature_importance_pairs[:cutoff]]
                return selected
    except Exception as e:
        print(f"提取特征重要性失败: {e}")
    
    # 如果无法从模型提取，使用默认的9个重要特征
    # 基于obesity_analysis.py中的典型特征重要性排名
    default_selected = [
        'Weight', 'Height', 'Age', 'FCVC', 'FAF', 
        'CH2O', 'TUE', 'NCP', 'family_history_with_overweight'
    ]
    # 只返回存在于all_features中的特征
    return [f for f in default_selected if f in all_features]
