# -*- coding: utf-8 -*-
"""
app/ml/predictor.py  -  肥胖风险预测引擎（含3项算法创新 + 特征筛选双模式）

创新点1：NHC指南约束的混合预测模型（ML + 规则引擎加权融合）
创新点2：基于SHAP的个体化特征归因解释
创新点3：交互特征工程 + Stacking集成模型

特征模式：
- 模式B（默认，推荐）：15个原始特征 → XGBoost（测试准确率91.09%, AUC=0.9905）
- 模式A（精简）：9个筛选特征 → XGBoost（测试准确率90.29%, AUC=0.9887）

15个核心特征（与 obesity_analysis.py 训练集列顺序完全一致）：
Gender, Age, Height, Weight, family_history_with_overweight, FAVC,
FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC

方案A 筛选后保留的9个重要特征：
Weight, Height, FCVC, Gender, Age, CH2O, TUE, NCP, FAF

注意：
- 训练时连续型特征（Age/Height/Weight/FCVC/NCP/CH2O/FAF/TUE）已通过 StandardScaler 标准化
- 预测时必须加载 scaler.joblib 并对同样列做 transform，再送入模型
- MTRANS（交通方式）不参与训练和预测（已从特征体系中剔除）
"""

import os
import numpy as np
import joblib
from flask import current_app

LABEL_MAP = {
    "Insufficient_Weight":  "体重不足",
    "Normal_Weight":        "正常体重",
    "Overweight_Level_I":   "超重I级",
    "Overweight_Level_II": "超重II级",
    "Obesity_Type_I":      "肥胖I型",
    "Obesity_Type_II":     "肥胖II型",
    "Obesity_Type_III":    "肥胖III型",
}

# LabelEncoder数字索引到字符串标签的映射
LABEL_MAP_REV = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III",
}

# 所有7个类别的字符串标签（按LabelEncoder顺序）
ALL_LABELS = [
    "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
    "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II",
    "Obesity_Type_III",
]

RISK_MAP = {
    "Insufficient_Weight": "中",
    "Normal_Weight":       "低",
    "Overweight_Level_I":  "中",
    "Overweight_Level_II": "高",
    "Obesity_Type_I":      "高",
    "Obesity_Type_II":     "高",
    "Obesity_Type_III":    "高",
}

CAEC_MAP = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
CALC_MAP = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}

# 连续型特征（需标准化），顺序与 obesity_analysis.py continuous_cols 一致
CONTINUOUS_COLS = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
# 连续特征在15维特征向量中的索引位置 [1, 2, 3, 6, 7, 10, 12, 13]
CONTINUOUS_IDX = [1, 2, 3, 6, 7, 10, 12, 13]

# 方案A：严格筛选的9个重要特征（综合排名 Top-9）
PLAN_A_FEATURES = ['Weight', 'Height', 'FCVC', 'Gender', 'Age',
                   'CH2O', 'TUE', 'NCP', 'FAF']
PLAN_A_FEATURE_SET = set(PLAN_A_FEATURES)
# 方案A中连续特征的索引（在9维向量中）
PLAN_A_CONTINUOUS_COLS = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
# 15维全特征中，方案A保留的特征索引
PLAN_A_IDX_IN_FULL = []  # 延迟初始化

# SHAP特征中文名映射
FEATURE_CN_MAP = {
    "Gender": "性别", "Age": "年龄", "Height": "身高", "Weight": "体重",
    "family_history_with_overweight": "家族肥胖史", "FAVC": "高热量食物",
    "FCVC": "蔬菜摄入频率", "NCP": "主餐次数", "CAEC": "零食频率",
    "SMOKE": "吸烟", "CH2O": "每日饮水量", "SCC": "热量监测",
    "FAF": "运动频率", "TUE": "屏幕时间", "CALC": "饮酒频率",
    "BMI": "BMI指数", "Age_x_FAF": "年龄*运动频率",
    "Weight_x_FAVC": "体重*高热量食物", "FAF_x_TUE": "运动*屏幕时间",
    "FCVC_x_CH2O": "蔬菜*饮水量",
}

# 创新点1：NHC BMI分类阈值（WS/T 428-2013）
NHC_BMI_THRESHOLDS = [
    (18.5, "Insufficient_Weight"),
    (24.0, "Normal_Weight"),
    (27.0, "Overweight_Level_I"),
    (30.0, "Overweight_Level_II"),
    (35.0, "Obesity_Type_I"),
    (40.0, "Obesity_Type_II"),
]

# 创新点1：混合模型权重（实验最优alpha=0.7）
HYBRID_ALPHA = 0.7


class ObesityPredictor:
    _instance = None
    _model      = None       # 模式B（15特征，默认主模型）
    _scaler     = None       # 模式B scaler
    _model_a    = None       # 模式A（9特征，筛选后）
    _scaler_a   = None       # 模式A scaler info
    _explainer  = None  # 创新点2：SHAP解释器
    _stacking   = None  # 创新点3：Stacking集成模型
    _enhanced_scaler = None  # 创新点3：增强特征标准化器

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        """延迟加载模型和标准化器（含方案A/B双套）"""
        if self._model is not None:
            return True
        try:
            model_path = current_app.config.get("ML_MODEL_PATH", "")
            scaler_path = current_app.config.get("ML_SCALER_PATH", "")

            # --- 模式B：15特征主模型（当前生产模型）---
            if os.path.exists(model_path):
                self._model = joblib.load(model_path)
                current_app.logger.info(f"模式B(15特征)模型加载成功: {model_path}")
            else:
                current_app.logger.warning(f"模型文件不存在: {model_path}")

            if os.path.exists(scaler_path):
                self._scaler = joblib.load(scaler_path)
                current_app.logger.info("模式B scaler加载成功")
            else:
                current_app.logger.warning(f"标准化器文件不存在: {scaler_path}")

            # --- 模式A：9特征筛选模型（对比实验模型） ---
            output_dir = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(__file__))))
            plan_a_model_path = os.path.join(output_dir, "analysis_output",
                                             "plan_a_model_9feat.joblib")
            plan_a_scaler_path = os.path.join(output_dir, "analysis_output",
                                               "plan_a_scaler_info.joblib")
            
            if os.path.exists(plan_a_model_path):
                self._model_a = joblib.load(plan_a_model_path)
                current_app.logger.info("模式A(9特征)模型加载成功: %s" % plan_a_model_path)
            else:
                current_app.logger.warning("模式A模型不存在（仅使用模式B预测）")

            if os.path.exists(plan_a_scaler_path):
                self._scaler_a = joblib.load(plan_a_scaler_path)
                current_app.logger.info("模式A scaler_info加载成功")
            else:
                current_app.logger.warning("模式A scaler_info不存在")

            # 初始化方案A的特征索引映射
            global PLAN_A_IDX_IN_FULL
            full_feature_names = [
                "Gender", "Age", "Height", "Weight", "family_history_with_overweight",
                "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC"
            ]
            PLAN_A_IDX_IN_FULL = [full_feature_names.index(f) for f in PLAN_A_FEATURES]

            # 创新点2：尝试加载SHAP解释器
            try:
                import shap
                if self._model is not None:
                    self._explainer = shap.TreeExplainer(self._model)
                    current_app.logger.info("SHAP解释器初始化成功")
            except ImportError:
                current_app.logger.info("shap库未安装，SHAP归因功能不可用")
            except Exception as e:
                current_app.logger.warning("SHAP解释器初始化失败: %s" % e)

            # 创新点3：尝试加载Stacking模型
            try:
                self._stacking_path = os.path.join(
                    output_dir, "analysis_output", "innovation_stacking_model.joblib")
                self._enhanced_scaler_path = os.path.join(
                    output_dir, "analysis_output", "innovation_scaler_enhanced.joblib")
                if os.path.exists(self._stacking_path):
                    current_app.logger.info("Stacking模型文件已找到（延迟加载）")
                else:
                    self._stacking_path = None
                if os.path.exists(self._enhanced_scaler_path):
                    current_app.logger.info("增强标准化器文件已找到（延迟加载）")
                else:
                    self._enhanced_scaler_path = None
            except Exception as e:
                current_app.logger.warning("Stacking模型路径检查失败: %s" % e)
                self._stacking_path = None
                self._enhanced_scaler_path = None

            if self._model is not None:
                expected_features = getattr(self._model, 'n_features_in_', None)
                if expected_features:
                    current_app.logger.info("模式B模型期望特征数: %d" % expected_features)
                if self._model_a is not None:
                    expected_a = getattr(self._model_a, 'n_features_in_', None)
                    if expected_a:
                        current_app.logger.info("模式A模型期望特征数: %d" % expected_a)
                return True
            return False
        except Exception as e:
            current_app.logger.error("模型/标准化器加载失败: %s" % e)
            return False

    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        """对连续型特征列应用 StandardScaler（与训练时一致）"""
        if self._scaler is None:
            current_app.logger.warning("scaler 未加载，连续特征未标准化，预测结果可能不准确")
            return features

        scaled = features.copy()
        # 提取所有连续列，一次性transform
        continuous_values = features[:, CONTINUOUS_IDX]  # shape: (1, 8)
        scaled_continuous = self._scaler.transform(continuous_values)  # shape: (1, 8)

        for i, col_idx in enumerate(CONTINUOUS_IDX):
            scaled[0, col_idx] = scaled_continuous[0, i]
        return scaled

    def _encode_features(self, data: dict) -> np.ndarray:
        """
        将用户输入字典编码为模型特征向量。
        特征顺序（与训练集 obesity_analysis.py 完全一致）：
        Gender, Age, Height, Weight, family_history_with_overweight, FAVC,
        FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC
        """
        gender   = int(data.get("Gender", data.get("gender", 1)))
        age      = float(data.get("Age", data.get("age", 25)))
        # 注意：训练数据中Height单位为**米(meters)**，mean=1.70
        # 前端表单输入单位为**厘米(cm)**，必须除以100转换为米！
        height_cm = float(data.get("Height", data.get("height", 170)))
        if height_cm > 3:
            # 大于3说明是厘米单位，需要转换
            height = height_cm / 100.0
        else:
            # 已经是米，直接使用
            height = height_cm
        weight   = float(data.get("Weight", data.get("weight", 65)))
        fam_hist = int(data.get("family_history_with_overweight",
                                 data.get("family_history", 0)))
        favc     = int(data.get("FAVC", data.get("high_calorie_food", 1)))
        fcvc     = float(data.get("FCVC", data.get("vegetable_frequency", 2)))
        ncp      = float(data.get("NCP", data.get("main_meals", 2)))

        caec_raw = str(data.get("CAEC", data.get("caec", "Sometimes")))
        caec     = CAEC_MAP.get(caec_raw, 1)

        smoke    = int(data.get("SMOKE", data.get("smoking", 0)))
        ch2o     = float(data.get("CH2O", data.get("water_consumption", 2)))

        scc      = int(data.get("SCC", data.get("calorie_monitoring", 0)))

        faf      = float(data.get("FAF", data.get("physical_activity", 1)))
        tue      = float(data.get("TUE", data.get("screen_time", 1)))

        calc_raw = str(data.get("CALC", data.get("alcohol", "no")))
        calc     = CALC_MAP.get(calc_raw, 0)

        features = np.array([[
            gender, age, height, weight,
            fam_hist, favc, fcvc, ncp,
            caec, smoke, ch2o, scc, faf, tue, calc
        ]], dtype=float)

        return features

    def _encode_features_plan_a(self, data: dict) -> np.ndarray:
        """
        方案A：将用户输入编码为9维筛选特征向量。
        特征顺序（与训练集严格一致）：
        Weight, Height, FCVC, Gender, Age, CH2O, TUE, NCP, FAF
        """
        gender   = int(data.get("Gender", data.get("gender", 1)))
        age      = float(data.get("Age", data.get("age", 25)))
        height_cm = float(data.get("Height", data.get("height", 170)))
        if height_cm > 3:
            height = height_cm / 100.0
        else:
            height = height_cm
        weight   = float(data.get("Weight", data.get("weight", 65)))
        fcvc     = float(data.get("FCVC", data.get("vegetable_frequency", 2)))
        ncp      = float(data.get("NCP", data.get("main_meals", 2)))
        ch2o     = float(data.get("CH2O", data.get("water_consumption", 2)))
        faf      = float(data.get("FAF", data.get("physical_activity", 1)))
        tue      = float(data.get("TUE", data.get("screen_time", 1)))

        # 方案A特征顺序: Weight, Height, FCVC, Gender, Age, CH2O, TUE, NCP, FAF
        features_a = np.array([[
            weight, height, fcvc, gender,
            age, ch2o, tue, ncp, faf
        ]], dtype=float)
        return features_a

    def _scale_features_plan_a(self, features_a: np.ndarray) -> np.ndarray:
        """对方案A的9维特征中的连续列应用标准化"""
        if self._scaler_a is None:
            current_app.logger.warning("模式A scaler未加载，使用原始值")
            return features_a

        scaled = features_a.copy()
        cont_cols = self._scaler_a.get("continuous_features", PLAN_A_CONTINUOUS_COLS)
        scaler_mean = self._scaler_a["mean"]
        scaler_scale = self._scaler_a["scale"]

        for i, col_name in enumerate(cont_cols):
            if col_name in PLAN_A_FEATURES:
                idx_in_9 = PLAN_A_FEATURES.index(col_name)
                scaled[0, idx_in_9] = (features_a[0, idx_in_9] - scaler_mean[i]) / scaler_scale[i]

        return scaled

    def predict_plan_a(self, data: dict) -> dict:
        """方案A预测入口（9个筛选特征），用于对比实验展示"""
        if self._model_a is None:
            return {"error": "模式A模型未加载", "success": False}

        try:
            features = self._encode_features_plan_a(data)
            features_scaled = self._scale_features_plan_a(features)

            pred_label_raw = self._model_a.predict(features_scaled)[0]
            proba_arr = self._model_a.predict_proba(features_scaled)[0]
            classes = list(self._model_a.classes_)

            if isinstance(pred_label_raw, (int, np.integer)):
                pred_label = LABEL_MAP_REV.get(int(pred_label_raw), str(pred_label_raw))
            else:
                pred_label = str(pred_label_raw)

            ml_proba = {}
            for cls, p in zip(classes, proba_arr):
                if isinstance(cls, (int, np.integer)):
                    label_key = LABEL_MAP_REV.get(int(cls), str(cls))
                else:
                    label_key = str(cls)
                ml_proba[label_key] = round(float(p), 4)

            risk_level = RISK_MAP.get(pred_label, "中")

            # 也做混合预测（创新点1）
            hybrid_info = {"enabled": False}
            final_label, final_proba, alpha_used = self._hybrid_predict(data, ml_proba)
            hybrid_info = {
                "enabled": True,
                "alpha": alpha_used,
                "ml_proba": ml_proba,
                "method": "ML(%.0f%%) + NHC(%.0f%%)" % (alpha_used*100, (1-alpha_used)*100),
            }

            return {
                "label": final_label,
                "label_cn": LABEL_MAP.get(final_label, final_label),
                "risk_level": RISK_MAP.get(final_label, "中"),
                "probabilities": final_proba,
                "success": True,
                "plan": "A",
                "plan_description": "Strict Top-9 Feature Selection",
                "n_features": 9,
                "features_used": list(PLAN_A_FEATURES),
                "innovation_hybrid": hybrid_info,
                "innovation_shap": {"enabled": False, "top_factors": []},
                "innovation_stacking": {"enabled": False},
            }
        except Exception as e:
            current_app.logger.error("模式A预测异常: %s" % e)
            return {"error": str(e), "success": False, "plan": "A"}

    # ========== BMI一致性校验（Stacking结果合理性检查） ==========
    @staticmethod
    def _bmi_consistency_check(data: dict, predicted_label: str) -> dict:
        """
        检查预测标签与BMI值是否基本一致。
        当Stacking等辅助模型的预测明显偏离BMI参考分类时返回警告信息。
        """
        height = float(data.get("Height", data.get("height", 170)))
        weight = float(data.get("Weight", data.get("weight", 65)))
        h = height if height <= 3 else height / 100.0
        bmi = weight / (h ** 2) if h > 0 else 25.0

        # 根据BMI确定NHC参考类别
        bmi_label = "Obesity_Type_III"
        for threshold, cls_name in NHC_BMI_THRESHOLDS:
            if bmi < threshold:
                bmi_label = cls_name
                break

        # 定义"正常/偏低/偏高/肥胖"的大类分组，用于宽松比较
        NORMAL_SET   = {"Insufficient_Weight", "Normal_Weight"}
        OVERWEIGHT_SET = {"Overweight_Level_I", "Overweight_Level_II"}
        OBESITY_SET  = {"Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"}

        pred_group = None
        if predicted_label in NORMAL_SET:
            pred_group = "normal"
        elif predicted_label in OVERWEIGHT_SET:
            pred_group = "overweight"
        elif predicted_label in OBESITY_SET:
            pred_group = "obesity"

        bmi_group = None
        if bmi_label in NORMAL_SET:
            bmi_group = "normal"
        elif bmi_label in OVERWEIGHT_SET:
            bmi_group = "overweight"
        elif bmi_label in OBESITY_SET:
            bmi_group = "obesity"

        # 一致性判断：分组不一致则告警
        is_consistent = (pred_group == bmi_group)

        return {
            "consistent": is_consistent,
            "bmi": round(bmi, 1),
            "bmi_label_cn": LABEL_MAP.get(bmi_label, bmi_label),
            "predicted_label_cn": LABEL_MAP.get(predicted_label, predicted_label),
            "warning": not is_consistent,
        }

    # ========== 创新点1：NHC指南约束的混合预测模型 ==========
    def _nhc_bmi_classify(self, data: dict) -> tuple:
        """
        基于NHC《成人肥胖食养指南》(2024版) BMI分类规则
        返回 (label, probability_dict)
        WS/T 428-2013 标准：正常18.5-24/超重24-28/肥胖>=28
        """
        height = float(data.get("Height", data.get("height", 170)))
        weight = float(data.get("Weight", data.get("weight", 65)))
        bmi = weight / ((height / 100.0) ** 2) if height > 0 else 25.0

        # 根据BMI确定基础分类
        label = "Obesity_Type_III"  # 默认最高级
        for threshold, cls_name in NHC_BMI_THRESHOLDS:
            if bmi < threshold:
                label = cls_name
                break

        # NHC规则修正：考虑行为风险因子
        risk_score = 0
        risk_score += int(data.get("FAVC", data.get("high_calorie_food", 1)) >= 1)
        risk_score += int(float(data.get("FAF", data.get("physical_activity", 1))) < 1)
        risk_score += int(data.get("family_history_with_overweight",
                                    data.get("family_history", 0)) == 1)

        # 风险因子修正分类边界
        if risk_score >= 2 and label == "Normal_Weight":
            label = "Overweight_Level_I"
        elif risk_score >= 2 and label == "Overweight_Level_I":
            label = "Overweight_Level_II"

        # 生成概率分布（以预测类别为中心的正态分布）
        classes = ALL_LABELS
        pred_idx = classes.index(label)
        sigma = 0.8
        raw_proba = np.exp(-0.5 * ((np.arange(len(classes)) - pred_idx) / sigma) ** 2)
        proba = raw_proba / raw_proba.sum()

        proba_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        return label, proba_dict

    def _hybrid_predict(self, data: dict, ml_proba: dict) -> dict:
        """
        创新点1：NHC指南约束的混合预测（增强版）

        策略：
        - 正常区域：ML为主(alpha=0.7)，规则辅助
        - 边界区域：规则权重增加(alpha=0.5)
        - 明显矛盾区域(如BMI>=30但ML预测超重)：规则强覆盖(alpha<=0.35)
        """
        height = float(data.get("Height", data.get("height", 170)))
        weight = float(data.get("Weight", data.get("weight", 65)))
        
        # 注意：predictor.py中Height可能已被转换为米，需兼容两种情况
        h_for_bmi = height
        if h_for_bmi > 3:
            h_for_bmi = h_for_bmi / 100.0
        bmi = weight / (h_for_bmi ** 2) if h_for_bmi > 0 else 25.0

        # 获取NHC规则概率和标签
        nhc_label, nhc_proba = self._nhc_bmi_classify(data)

        # 获取ML预测标签
        ml_label = max(ml_proba, key=ml_proba.get)

        # === 动态alpha计算 ===
        alpha = HYBRID_ALPHA  # 默认0.7

        # 定义"非肥胖类别"（ML预测为这些时，如果BMI明显肥胖则需要规则纠正）
        NON_OBESITY_LABELS = {
            "Insufficient_Weight", "Normal_Weight",
            "Overweight_Level_I", "Overweight_Level_II",
        }

        ml_is_non_obesity = ml_label in NON_OBESITY_LABELS
        bmi_clearly_obese = bmi >= 30.0
        
        # 场景A：BMI明确肥胖(>=30)但ML未识别为任何Obesity类型 → 强制规则覆盖
        if bmi_clearly_obese and ml_is_non_obesity:
            if bmi >= 40.0:
                alpha = 0.20  # 极度肥胖：规则占80%
            elif bmi >= 35.0:
                alpha = 0.25  # 重度肥胖：规则占75%
            else:  # 30 <= bmi < 35
                alpha = 0.30  # 一般肥胖：规则占70%
                
        # 场景B：BMI明确超重(>=27)但ML预测正常体重 → 规则修正
        elif bmi >= 27.0 and ml_label in ("Normal_Weight", "Insufficient_Weight"):
            alpha = 0.35
            
        # 场景C：边界区域检测（BMI在分类阈值附近 +-1.5）
        else:
            boundary_bmis = [18.5, 24.0, 27.0, 30.0, 35.0, 40.0]
            is_boundary = any(abs(bmi - t) < 1.5 for t in boundary_bmis)
            if is_boundary:
                alpha = max(0.45, alpha - 0.2)  # 边界区域适度降低ML权重

        # 加权融合
        classes = ALL_LABELS
        fused_proba = {}
        for cls in classes:
            ml_p = ml_proba.get(cls, 0.0)
            rule_p = nhc_proba.get(cls, 0.0)
            fused_proba[cls] = round(alpha * ml_p + (1 - alpha) * rule_p, 4)

        # 归一化
        total = sum(fused_proba.values())
        if total > 0:
            fused_proba = {k: round(v / total, 4) for k, v in fused_proba.items()}

        pred_label = max(fused_proba, key=fused_proba.get)
        return pred_label, fused_proba, alpha

    # ========== 创新点2：SHAP个体化特征归因 ==========
    def _shap_attribution(self, features: np.ndarray) -> list:
        """
        创新点2：基于SHAP的个体化特征归因解释
        返回 Top-3 风险因素列表
        """
        if self._explainer is None:
            return []

        try:
            shap_values = self._explainer.shap_values(features)

            # XGBoost多分类：shap_values 可能是 list[arrays] 或 3D array
            shap_arr = np.array(shap_values)

            # 获取预测类别
            pred_class_raw = self._model.predict(features)[0]
            classes = list(self._model.classes_)

            # 将数字类别映射为索引
            if isinstance(pred_class_raw, (int, np.integer)):
                pred_idx = int(pred_class_raw)
            else:
                pred_idx = classes.index(pred_class_raw)

            # 取预测类别的SHAP值
            if shap_arr.ndim == 3:
                if shap_arr.shape[0] == 1:
                    # shape = (1, n_features, n_classes) - XGBoost多分类
                    sample_shap = shap_arr[0, :, pred_idx]
                elif shap_arr.shape[2] <= 7:
                    # shape = (n_classes, n_samples, n_features)
                    sample_shap = shap_arr[pred_idx, 0, :]
                else:
                    sample_shap = shap_arr[0, pred_idx, :]
            elif isinstance(shap_values, list):
                # list of (n_samples, n_features) per class
                class_shap = np.array(shap_values[pred_idx])
                if class_shap.ndim == 2:
                    sample_shap = class_shap[0, :]
                else:
                    sample_shap = class_shap
            elif shap_arr.ndim == 2:
                # shape = (n_classes, n_features) for single sample
                sample_shap = shap_arr[pred_idx, :]
            else:
                return []

            # 特征名（原始15个）
            feature_names = [
                "Gender", "Age", "Height", "Weight",
                "family_history_with_overweight", "FAVC",
                "FCVC", "NCP", "CAEC", "SMOKE",
                "CH2O", "SCC", "FAF", "TUE", "CALC"
            ]

            # 按绝对SHAP值排序，取Top-3
            abs_shap = np.abs(sample_shap)
            top3_idx = np.argsort(abs_shap)[-3:][::-1]

            factors = []
            for idx in top3_idx:
                if abs_shap[idx] < 0.01:
                    continue
                feature = feature_names[idx]
                shap_val = float(sample_shap[idx])
                factors.append({
                    "feature": feature,
                    "feature_cn": FEATURE_CN_MAP.get(feature, feature),
                    "shap_value": round(shap_val, 4),
                    "direction": "增加风险" if shap_val > 0 else "降低风险",
                    "abs_contribution": round(abs_shap[idx], 4),
                })

            return factors
        except Exception as e:
            current_app.logger.warning(f"SHAP归因计算失败: {e}")
            return []

    # ========== 创新点3：交互特征工程 ==========
    def _add_interaction_features(self, features: np.ndarray, raw_data: dict) -> np.ndarray:
        """
        创新点3：构建交互特征
        BMI = Weight / (Height/100)^2
        Age_x_FAF = Age * FAF
        Weight_x_FAVC = Weight * FAVC
        FAF_x_TUE = FAF * TUE
        FCVC_x_CH2O = FCVC * CH2O
        """
        gender, age, height, weight, fam_hist, favc, fcvc, ncp, \
            caec, smoke, ch2o, scc, faf, tue, calc = features[0]

        bmi = weight / ((height / 100.0) ** 2) if height > 0 else 25.0
        age_x_faf = age * faf
        weight_x_favc = weight * favc
        faf_x_tue = faf * tue
        fcvc_x_ch2o = fcvc * ch2o

        interaction = np.array([[bmi, age_x_faf, weight_x_favc, faf_x_tue, fcvc_x_ch2o]])
        return np.hstack([features, interaction])

    def _scale_enhanced_features(self, enhanced_features: np.ndarray) -> np.ndarray:
        """对增强特征（含交互特征）应用增强标准化器"""
        if self._enhanced_scaler is None:
            return enhanced_features
        try:
            # 增强标准化器只标准化连续列+交互列
            scaled = enhanced_features.copy()
            n_continuous = len(CONTINUOUS_IDX)
            n_interaction = 5
            enhanced_continuous_idx = list(range(n_continuous)) + \
                list(range(15, 15 + n_interaction))

            continuous_values = enhanced_features[:, enhanced_continuous_idx]
            scaled_continuous = self._enhanced_scaler.transform(continuous_values)

            for i, idx in enumerate(enhanced_continuous_idx):
                scaled[0, idx] = scaled_continuous[0, i]
            return scaled
        except Exception as e:
            current_app.logger.warning(f"增强特征标准化失败: {e}")
            return enhanced_features

    # ========== 主预测入口 ==========
    def predict(self, data: dict, use_hybrid: bool = True,
                use_shap: bool = True, use_stacking: bool = True) -> dict:
        """
        预测入口（含3项算法创新），返回完整结果字典

        Args:
            data: 用户输入特征字典
            use_hybrid: 是否使用创新点1（NHC混合模型）
            use_shap: 是否使用创新点2（SHAP归因）
            use_stacking: 是否使用创新点3（Stacking集成）
        """
        if not self._load():
            return self._fallback_predict(data)

        try:
            features = self._encode_features(data)
            features_scaled = self._scale_features(features)

            # 验证特征维度
            expected = getattr(self._model, 'n_features_in_', None)
            if expected and features_scaled.shape[1] != expected:
                current_app.logger.error(
                    f"特征维度不匹配: 输入{features_scaled.shape[1]} vs 期望{expected}")
                return self._fallback_predict(data)

            # 基础XGBoost预测
            pred_label_raw = self._model.predict(features_scaled)[0]
            proba_arr  = self._model.predict_proba(features_scaled)[0]
            classes    = list(self._model.classes_)

            # 将数字类别映射回字符串标签
            if isinstance(pred_label_raw, (int, np.integer)):
                pred_label = LABEL_MAP_REV.get(int(pred_label_raw), str(pred_label_raw))
            else:
                pred_label = str(pred_label_raw)

            # 构建概率字典（统一使用字符串标签作为key）
            ml_proba = {}
            for cls, p in zip(classes, proba_arr):
                if isinstance(cls, (int, np.integer)):
                    label_key = LABEL_MAP_REV.get(int(cls), str(cls))
                else:
                    label_key = str(cls)
                ml_proba[label_key] = round(float(p), 4)

            # ---- 创新点1：NHC混合预测 ----
            hybrid_info = {}
            final_proba = ml_proba
            final_label = pred_label
            if use_hybrid:
                final_label, final_proba, used_alpha = self._hybrid_predict(data, ml_proba)
                hybrid_info = {
                    "enabled": True,
                    "alpha": used_alpha,
                    "ml_proba": ml_proba,
                    "method": f"ML({used_alpha:.0%}) + NHC({1-used_alpha:.0%})",
                }
            else:
                hybrid_info = {"enabled": False}

            # ---- 创新点2：SHAP个体化归因 ----
            shap_factors = []
            if use_shap:
                shap_factors = self._shap_attribution(features_scaled)

            # ---- 创新点3：Stacking集成预测 ----
            stacking_info = {"enabled": False}
            if use_stacking:
                # 延迟加载Stacking模型
                if self._stacking is None and getattr(self, '_stacking_path', None):
                    try:
                        self._stacking = joblib.load(self._stacking_path)
                        current_app.logger.info("Stacking集成模型延迟加载成功")
                    except Exception as e:
                        current_app.logger.warning(f"Stacking模型延迟加载失败: {e}")
                        self._stacking = None

                # 延迟加载增强标准化器
                if self._enhanced_scaler is None and getattr(self, '_enhanced_scaler_path', None):
                    try:
                        self._enhanced_scaler = joblib.load(self._enhanced_scaler_path)
                        current_app.logger.info("增强标准化器延迟加载成功")
                    except Exception as e:
                        current_app.logger.warning(f"增强标准化器延迟加载失败: {e}")
                        self._enhanced_scaler = None

                if self._stacking is not None:
                    try:
                        enhanced = self._add_interaction_features(features, data)
                        enhanced_scaled = self._scale_enhanced_features(enhanced)

                        expected_stack = getattr(self._stacking, 'n_features_in_', None)
                        if expected_stack and enhanced_scaled.shape[1] == expected_stack:
                            stack_pred_raw = self._stacking.predict(enhanced_scaled)[0]
                            stack_proba_arr = self._stacking.predict_proba(enhanced_scaled)[0]
                            stack_classes = list(self._stacking.classes_)

                            # 将数字类别映射回字符串标签
                            if isinstance(stack_pred_raw, (int, np.integer)):
                                stack_pred = LABEL_MAP_REV.get(int(stack_pred_raw), str(stack_pred_raw))
                            else:
                                stack_pred = str(stack_pred_raw)

                            stack_proba = {}
                            for cls, p in zip(stack_classes, stack_proba_arr):
                                if isinstance(cls, (int, np.integer)):
                                    label_key = LABEL_MAP_REV.get(int(cls), str(cls))
                                else:
                                    label_key = str(cls)
                                stack_proba[label_key] = round(float(p), 4)

                            stacking_info = {
                                "enabled": True,
                                "label": stack_pred,
                                "label_cn": LABEL_MAP.get(stack_pred, stack_pred),
                                "probabilities": stack_proba,
                                # BMI一致性校验
                                **self._bmi_consistency_check(data, stack_pred),
                            }
                    except Exception as e:
                        current_app.logger.warning(f"Stacking预测失败: {e}")

            risk_level = RISK_MAP.get(final_label, "中")

            result = {
                "label":       final_label,
                "label_cn":    LABEL_MAP.get(final_label, final_label),
                "risk_level":  risk_level,
                "probabilities": final_proba,
                "success":     True,
                # 模式标识
                "plan":        "B",
                "plan_description": "Full 15 Features (Production)",
                "n_features":  15,
                # 创新点信息
                "innovation_hybrid": hybrid_info,
                "innovation_shap": {
                    "enabled": use_shap,
                    "top_factors": shap_factors,
                },
                "innovation_stacking": stacking_info,
            }

            return result
        except Exception as e:
            current_app.logger.error(f"预测异常: {e}")
            return self._fallback_predict(data)

    def predict_compare(self, data: dict) -> dict:
        """
        双模式预测对比：同时用方案A(9特征)和方案B(15特征)预测，返回完整对比结果。
        用于论文展示和前端对比页面。
        """
        # 方案B（默认主模型）
        result_b = self.predict(data, use_hybrid=True, use_shap=True, use_stacking=False)

        # 方案A（筛选模型）
        result_a = self.predict_plan_a(data)

        # 构建对比摘要
        same_prediction = (result_b.get("label") == result_a.get("label"))
        
        comparison = {
            "same_prediction": same_prediction,
            "plan_a": result_a,
            "plan_b": result_b,
            "summary": {
                "plan_a_label": result_a.get("label"),
                "plan_a_label_cn": LABEL_MAP.get(result_a.get("label", ""), ""),
                "plan_b_label": result_b.get("label"),
                "plan_b_label_cn": LABEL_MAP.get(result_b.get("label", ""), ""),
                "agreement": "一致" if same_prediction else "不一致",
                "recommendation": (
                    "方案B更优(15特征, 准确率91.09% vs 90.29%)" if not same_prediction else
                    "两方案预测一致"
                ),
                "experiment_data": {
                    "plan_a_accuracy": 0.9029,
                    "plan_b_accuracy": 0.9109,
                    "plan_a_auc": 0.9887,
                    "plan_b_auc": 0.9905,
                    "accuracy_improvement": "+0.80%",
                    "n_features_a": 9,
                    "n_features_b": 15,
                    "excluded_in_plan_a": ["family_history_with_overweight", "FAVC",
                                            "CAEC", "SMOKE", "SCC", "CALC"],
                },
            },
            "success": True,
        }

        return comparison

    def _fallback_predict(self, data: dict) -> dict:
        """BMI + 规则引擎兜底预测（模型未加载时使用）"""
        height   = float(data.get("Height", data.get("height", 170)))
        weight   = float(data.get("Weight", data.get("weight", 65)))
        bmi      = weight / ((height / 100.0) ** 2) if height > 0 else 25.0

        score = 0
        score += int(data.get("FAVC", data.get("high_calorie_food", 1)) >= 2)
        score += int(data.get("FAF", data.get("physical_activity", 1)) == 0)
        score += int(data.get("family_history_with_overweight", data.get("family_history", 0)) == 1)
        score += int(data.get("CAEC", data.get("snacking", "Sometimes")) in ["Frequently", "Always"])

        if bmi < 18.5:
            label = "Insufficient_Weight"
        elif bmi < 24.0:
            label = "Normal_Weight" if score < 2 else "Overweight_Level_I"
        elif bmi < 27.0:
            label = "Overweight_Level_I"
        elif bmi < 30.0:
            label = "Overweight_Level_II"
        elif bmi < 35.0:
            label = "Obesity_Type_I"
        elif bmi < 40.0:
            label = "Obesity_Type_II"
        else:
            label = "Obesity_Type_III"

        classes    = ALL_LABELS
        proba_dict = {c: 0.01 for c in classes}
        proba_dict[label] = 0.88
        proba_dict[classes[max(0, classes.index(label) - 1)]] = 0.06
        proba_dict[classes[min(len(classes) - 1, classes.index(label) + 1)]] = 0.05

        return {
            "label":       label,
            "label_cn":    LABEL_MAP.get(label, label),
            "risk_level":  RISK_MAP.get(label, "中"),
            "probabilities": proba_dict,
            "success":     True,
            "fallback":    True,
            "innovation_hybrid": {"enabled": False},
            "innovation_shap": {"enabled": False, "top_factors": []},
            "innovation_stacking": {"enabled": False},
        }


predictor = ObesityPredictor()
