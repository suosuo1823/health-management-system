# =============================================================================
# 肥胖风险预测系统 - 基于最优模型的完整预测流水线
# 使用说明：
#   1. 先运行 obesity_analysis.py 完成训练和模型保存
#   2. 再运行本文件进行预测
#   运行方式：python obesity_predict.py
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import json
import numpy as np
import pandas as pd
import joblib

# ─────────────────────────────────────────────
# 全局路径配置
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model_tuned.joblib')
SCALER_PATH = os.path.join(OUTPUT_DIR, 'scaler.joblib')
LE_PATH = os.path.join(OUTPUT_DIR, 'label_encoder.joblib')
FEATURE_INFO_PATH = os.path.join(OUTPUT_DIR, 'feature_info.json')

# ─────────────────────────────────────────────
# 加载已保存的模型、标准化器和标签编码器
# ─────────────────────────────────────────────
print("=" * 65)
print("  肥胖风险预测系统 - 加载模型中...")
print("=" * 65)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"找不到模型文件：{MODEL_PATH}\n"
        f"请先运行 obesity_analysis.py 完成训练。"
    )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LE_PATH)

with open(FEATURE_INFO_PATH, 'r', encoding='utf-8') as f:
    feature_info = json.load(f)

ALL_FEATURES = feature_info['all_features']
CONTINUOUS_COLS = feature_info['continuous_cols']
CLASS_NAMES = feature_info['class_names']
BEST_MODEL_NAME = feature_info['best_model']

print(f"  模型类型：{BEST_MODEL_NAME}")
print(f"  特征数量：{len(ALL_FEATURES)}")
print(f"  目标类别：{CLASS_NAMES}")
print(f"  已保存的最优超参数：{feature_info['best_params']}")
print(f"  训练时测试集准确率：{feature_info['test_accuracy_after']:.4f}")
print()


# =============================================================================
# 数据预处理函数（与训练时完全一致）
# =============================================================================
def preprocess_input(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    对输入的原始数据执行与训练时完全一致的预处理步骤：
    1. 修正目标变量拼写（如有）
    2. Gender 二值编码
    3. CAEC 有序编码
    4. CALC 有序编码
    5. 缺失值填充
    7. 对齐特征列（按训练集顺序，补全缺失列）
    8. 连续特征标准化

    参数
    ------
    df_input : pd.DataFrame
        原始输入数据（不含目标变量 '0be1dad'，也不含 'id' 列）

    返回
    ------
    pd.DataFrame : 预处理后、可直接用于预测的特征矩阵
    """
    df = df_input.copy()

    # ── Gender 编码 ──
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 1: 1, 0: 0})
        df['Gender'] = df['Gender'].fillna(0)

    # ── CAEC 有序编码 ──
    caec_map = {'no': 0, '0': 0, 0: 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    if 'CAEC' in df.columns:
        df['CAEC'] = df['CAEC'].map(caec_map)
        df['CAEC'] = df['CAEC'].fillna(1)

    # ── CALC 有序编码 ──
    calc_map = {'no': 0, '0': 0, 0: 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    if 'CALC' in df.columns:
        df['CALC'] = df['CALC'].map(calc_map)
        df['CALC'] = df['CALC'].fillna(0)

        # ── 类型转换 ──
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    # ── 连续特征标准化（使用训练时的 scaler）──
    df[CONTINUOUS_COLS] = scaler.transform(df[CONTINUOUS_COLS])

    return df


# =============================================================================
# 单条样本预测函数
# =============================================================================
def predict_single(
    gender: str,
    age: float,
    height: float,
    weight: float,
    family_history_with_overweight: int,
    favc: int,
    fcvc: float,
    ncp: float,
    caec: str,
    smoke: int,
    ch2o: float,
    scc: int,
    faf: float,
    tue: float,
    calc: str,
) -> dict:
    """
    对单条样本进行肥胖等级预测。

    参数说明
    ---------
    gender                      : 性别，'Male' 或 'Female'
    age                         : 年龄（岁）
    height                      : 身高（米，例如 1.75）
    weight                      : 体重（千克）
    family_history_with_overweight : 家族肥胖史，1=有 0=无
    favc                        : 频繁食用高热量食物，1=是 0=否
    fcvc                        : 蔬菜摄入频次（1-3）
    ncp                         : 主餐次数（1-4）
    caec                        : 两餐间零食，'no'/'Sometimes'/'Frequently'/'Always'
    smoke                       : 是否吸烟，1=是 0=否
    ch2o                        : 每日饮水量（1-3）
    scc                         : 高热量饮料摄入，1=是 0=否
    faf                         : 运动频率（0-3）
    tue                         : 使用电子设备时间（0-2）
    calc                        : 饮酒频率，'no'/'Sometimes'/'Frequently'/'Always'

    返回
    ------
    dict 包含：
        'predicted_class'     : 预测肥胖等级（中文映射）
        'predicted_label'     : 预测肥胖等级（原始英文标签）
        'confidence'          : 该类别的置信概率
        'all_probabilities'   : 所有类别的预测概率（字典）
        'bmi'                 : 计算所得 BMI 值
        'bmi_standard'        : BMI 标准等级
        'input_features'      : 输入特征汇总
    """
    # ── 计算 BMI（辅助参考）──
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        bmi_std = '偏瘦（BMI < 18.5）'
    elif bmi < 24.0:
        bmi_std = '正常（18.5 ≤ BMI < 24）'
    elif bmi < 28.0:
        bmi_std = '超重（24 ≤ BMI < 28）'
    else:
        bmi_std = f'肥胖（BMI ≥ 28）'

    # ── 构建单样本 DataFrame ──
    raw_data = {
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history_with_overweight': [family_history_with_overweight],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [caec],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'SCC': [scc],
        'FAF': [faf],
        'TUE': [tue],
        'CALC': [calc]
    }
    df_single = pd.DataFrame(raw_data)

    # ── 预处理 ──
    X_processed = preprocess_input(df_single)

    # ── 预测 ──
    pred_class_idx = model.predict(X_processed)[0]
    pred_proba = model.predict_proba(X_processed)[0]
    pred_label = le.inverse_transform([pred_class_idx])[0]
    pred_confidence = pred_proba[pred_class_idx]

    # 中文标签映射
    label_cn_map = {
        'Insufficient_Weight': '体重不足',
        'Normal_Weight': '正常体重',
        'Overweight_Level_I': '超重一级',
        'Overweight_Level_II': '超重二级',
        'Obesity_Type_I': '肥胖 I 型',
        'Obesity_Type_II': '肥胖 II 型',
        'Obesity_Type_III': '肥胖 III 型',
        # 处理原数据中的拼写变体
        'Insufficient Weight': '体重不足',
        '0rmal_Weight': '正常体重',
    }
    pred_cn = label_cn_map.get(pred_label, pred_label)

    # 所有类别概率
    all_prob_dict = {
        CLASS_NAMES[i]: round(float(pred_proba[i]), 6)
        for i in range(len(CLASS_NAMES))
    }
    all_prob_sorted = dict(sorted(all_prob_dict.items(),
                                   key=lambda x: x[1], reverse=True))

    result = {
        'predicted_class': pred_cn,
        'predicted_label': pred_label,
        'confidence': round(float(pred_confidence), 6),
        'all_probabilities': all_prob_sorted,
        'bmi': round(bmi, 2),
        'bmi_standard': bmi_std,
        'input_features': {
            '性别': gender,
            '年龄': age,
            '身高': f'{height} m',
            '体重': f'{weight} kg',
            '家族肥胖史': '有' if family_history_with_overweight else '无',
            '频繁吃高热量食物': '是' if favc else '否',
            '蔬菜摄入频次': fcvc,
            '主餐次数/天': ncp,
            '两餐间零食': caec,
            '是否吸烟': '是' if smoke else '否',
            '每日饮水量': ch2o,
            '高热量饮料': '是' if scc else '否',
            '运动频率': faf,
            '电子设备使用时间': tue,
            '饮酒频率': calc
        }
    }
    return result


# =============================================================================
# 批量预测函数（处理 CSV 文件）
# =============================================================================
def predict_batch(input_csv_path: str, output_csv_path: str = None) -> pd.DataFrame:
    """
    对 CSV 文件进行批量预测，支持含目标变量（评估模式）和不含目标变量（预测模式）两种格式。

    参数
    ------
    input_csv_path  : 输入 CSV 文件路径
    output_csv_path : 预测结果输出路径（可选）

    返回
    ------
    pd.DataFrame : 附加预测结果的 DataFrame
    """
    df_input = pd.read_csv(input_csv_path)
    print(f"  批量预测：读取 {len(df_input)} 条样本")

    # 判断是否包含目标变量
    has_target = '0be1dad' in df_input.columns
    if has_target:
        y_true_raw = df_input['0be1dad'].copy()
        df_features = df_input.drop(columns=['0be1dad'], errors='ignore')
    else:
        df_features = df_input.copy()

    # 去除 id 列（如有）
    if 'id' in df_features.columns:
        id_col = df_features['id'].copy()
        df_features = df_features.drop(columns=['id'])
    else:
        id_col = pd.RangeIndex(len(df_features))

    # 预处理
    X_processed = preprocess_input(df_features)

    # 预测
    pred_indices = model.predict(X_processed)
    pred_probas = model.predict_proba(X_processed)
    pred_labels = le.inverse_transform(pred_indices)
    pred_confidences = [pred_probas[i, pred_indices[i]] for i in range(len(pred_indices))]

    # 中文映射
    label_cn_map = {
        'Insufficient_Weight': '体重不足',
        'Normal_Weight': '正常体重',
        'Overweight_Level_I': '超重一级',
        'Overweight_Level_II': '超重二级',
        'Obesity_Type_I': '肥胖 I 型',
        'Obesity_Type_II': '肥胖 II 型',
        'Obesity_Type_III': '肥胖 III 型',
        'Insufficient Weight': '体重不足',
        '0rmal_Weight': '正常体重',
    }
    pred_cn_labels = [label_cn_map.get(lbl, lbl) for lbl in pred_labels]

    # 汇总结果
    result_df = df_input.copy()
    result_df['预测肥胖等级_英文'] = pred_labels
    result_df['预测肥胖等级_中文'] = pred_cn_labels
    result_df['置信度'] = [round(c, 4) for c in pred_confidences]

    # 每个类别概率列
    for i, cls in enumerate(CLASS_NAMES):
        result_df[f'P_{cls}'] = pred_probas[:, i].round(4)

    # 如有真实标签，输出准确率
    if has_target:
        from sklearn.metrics import accuracy_score, classification_report
        y_true_raw_clean = y_true_raw.str.strip().replace({'0rmal_Weight': 'Normal_Weight',
                                                             'Ormal_Weight': 'Normal_Weight'})
        y_true_encoded = le.transform(y_true_raw_clean)
        acc = accuracy_score(y_true_encoded, pred_indices)
        print(f"\n  批量评估模式 → 准确率：{acc:.4f} ({acc * 100:.2f}%)")
        print("\n  分类报告：")
        print(classification_report(y_true_encoded, pred_indices,
                                    target_names=CLASS_NAMES))
        result_df['真实肥胖等级'] = y_true_raw.values
        result_df['预测是否正确'] = (y_true_encoded == pred_indices).astype(int)

    # 保存结果
    if output_csv_path:
        result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n  批量预测结果已保存：{output_csv_path}")

    return result_df


# =============================================================================
# 交互式命令行预测
# =============================================================================
def interactive_predict():
    """
    交互式命令行预测模式：逐一提示用户输入特征值，完成预测后打印结果。
    """
    print("\n" + "=" * 65)
    print("  交互式预测模式")
    print("  请按提示输入各项特征值（直接回车使用默认值）")
    print("=" * 65)

    def prompt(text, default, cast_fn=str):
        val = input(f"  {text} [默认={default}]: ").strip()
        if val == '':
            return default
        try:
            return cast_fn(val)
        except Exception:
            print(f"    输入格式错误，使用默认值 {default}")
            return default

    gender = prompt("性别 (Male/Female)", "Male")
    age = prompt("年龄（岁）", 25.0, float)
    height = prompt("身高（米，例如1.75）", 1.75, float)
    weight = prompt("体重（千克）", 70.0, float)
    fh = prompt("家族肥胖史 (1=有/0=无)", 1, int)
    favc = prompt("频繁吃高热量食物 (1=是/0=否)", 1, int)
    fcvc = prompt("蔬菜摄入频次 (1.0-3.0)", 2.0, float)
    ncp = prompt("每日主餐次数 (1.0-4.0)", 3.0, float)
    caec = prompt("两餐间零食 (no/Sometimes/Frequently/Always)", "Sometimes")
    smoke = prompt("是否吸烟 (1=是/0=否)", 0, int)
    ch2o = prompt("每日饮水量 (1.0-3.0)", 2.0, float)
    scc = prompt("高热量饮料摄入 (1=是/0=否)", 0, int)
    faf = prompt("运动频率 (0.0-3.0)", 1.0, float)
    tue = prompt("电子设备使用时间 (0.0-2.0)", 1.0, float)
    calc = prompt("饮酒频率 (no/Sometimes/Frequently/Always)", "Sometimes")

    result = predict_single(
        gender=gender, age=age, height=height, weight=weight,
        family_history_with_overweight=fh, favc=favc, fcvc=fcvc,
        ncp=ncp, caec=caec, smoke=smoke, ch2o=ch2o, scc=scc,
        faf=faf, tue=tue, calc=calc
    )

    print("\n" + "─" * 65)
    print("  【预测结果】")
    print(f"  BMI 值        ：{result['bmi']}")
    print(f"  BMI 标准分级  ：{result['bmi_standard']}")
    print(f"  预测肥胖等级  ：{result['predicted_class']} ({result['predicted_label']})")
    print(f"  预测置信度    ：{result['confidence'] * 100:.2f}%")
    print("\n  各类别概率分布：")
    for cls, prob in result['all_probabilities'].items():
        bar_len = int(prob * 40)
        bar = '█' * bar_len + '░' * (40 - bar_len)
        print(f"    {cls:<30} {bar} {prob * 100:.2f}%")
    print("─" * 65)
    return result


# =============================================================================
# 演示用例：使用几条真实数据验证预测流水线
# =============================================================================
def run_demo_cases():
    """
    运行内置演示用例，展示预测流水线的完整使用方式。
    这些用例直接来自数据集的真实样本（id=0,1,2,3,4），
    包含不同肥胖等级的代表性记录。
    """
    print("\n" + "=" * 65)
    print("  演示用例：对真实样本进行预测验证")
    print("=" * 65)

    # 真实样本，来自原始数据集（id=0~4）
    demo_cases = [
        {
            'desc': '样本 id=0（真实：Overweight_Level_II）',
            'gender': 'Male', 'age': 24.443011, 'height': 1.699998, 'weight': 81.66995,
            'fh': 1, 'favc': 1, 'fcvc': 2.0, 'ncp': 2.983297,
            'caec': 'Sometimes', 'smoke': 0, 'ch2o': 2.763573, 'scc': 0,
            'faf': 0.0, 'tue': 0.976473, 'calc': 'Sometimes'
        },
        {
            'desc': '样本 id=1（真实：Normal_Weight）',
            'gender': 'Female', 'age': 18.0, 'height': 1.56, 'weight': 57.0,
            'fh': 1, 'favc': 1, 'fcvc': 2.0, 'ncp': 3.0,
            'caec': 'Frequently', 'smoke': 0, 'ch2o': 2.0, 'scc': 0,
            'faf': 1.0, 'tue': 1.0, 'calc': 'no'
        },
        {
            'desc': '样本 id=2（真实：Insufficient_Weight）',
            'gender': 'Female', 'age': 18.0, 'height': 1.71146, 'weight': 50.165754,
            'fh': 1, 'favc': 1, 'fcvc': 1.880534, 'ncp': 1.411685,
            'caec': 'Sometimes', 'smoke': 0, 'ch2o': 1.910378, 'scc': 0,
            'faf': 0.866045, 'tue': 1.673584, 'calc': 'no'
        },
        {
            'desc': '样本 id=3（真实：Obesity_Type_III）',
            'gender': 'Female', 'age': 20.952737, 'height': 1.71073, 'weight': 131.274851,
            'fh': 1, 'favc': 1, 'fcvc': 3.0, 'ncp': 3.0,
            'caec': 'Sometimes', 'smoke': 0, 'ch2o': 1.674061, 'scc': 0,
            'faf': 1.467863, 'tue': 0.780199, 'calc': 'Sometimes'
        },
        {
            'desc': '样本 id=4（真实：Overweight_Level_II）',
            'gender': 'Male', 'age': 31.641081, 'height': 1.914186, 'weight': 93.798055,
            'fh': 1, 'favc': 1, 'fcvc': 2.679664, 'ncp': 1.971472,
            'caec': 'Sometimes', 'smoke': 0, 'ch2o': 1.979848, 'scc': 0,
            'faf': 1.967973, 'tue': 0.931721, 'calc': 'Sometimes'
        },
    ]

    for case in demo_cases:
        result = predict_single(
            gender=case['gender'],
            age=case['age'],
            height=case['height'],
            weight=case['weight'],
            family_history_with_overweight=case['fh'],
            favc=case['favc'],
            fcvc=case['fcvc'],
            ncp=case['ncp'],
            caec=case['caec'],
            smoke=case['smoke'],
            ch2o=case['ch2o'],
            scc=case['scc'],
            faf=case['faf'],
            tue=case['tue'],
            calc=case['calc']
        )
        print(f"\n  >> {case['desc']}")
        print(f"    BMI={result['bmi']}  ({result['bmi_standard']})")
        print(f"    预测：{result['predicted_class']} [{result['predicted_label']}]  "
              f"置信度：{result['confidence'] * 100:.2f}%")
        print("    概率分布（前3）：", end='')
        top3 = list(result['all_probabilities'].items())[:3]
        for cls, prob in top3:
            print(f"{cls}={prob * 100:.1f}%", end='  ')
        print()


# =============================================================================
# 批量预测整个数据集（评估模式）
# =============================================================================
def run_full_dataset_evaluation():
    """
    对完整数据集进行批量预测并评估准确率。
    （仅用于验证预测流水线完整性，实际使用时应用新数据）
    """
    print("\n" + "=" * 65)
    print("  全数据集批量评估（验证流水线正确性）")
    print("=" * 65)
    input_path = r"c:\Users\WXS\Desktop\学校\毕设数据分析2\obesity_level.csv"
    output_path = os.path.join(OUTPUT_DIR, 'batch_prediction_results.csv')
    result_df = predict_batch(input_path, output_path)
    print(f"\n  预测结果前5行：")
    display_cols = ['预测肥胖等级_中文', '预测肥胖等级_英文', '置信度']
    if '真实肥胖等级' in result_df.columns:
        display_cols = ['真实肥胖等级'] + display_cols + ['预测是否正确']
    print(result_df[display_cols].head(10).to_string())
    return result_df


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == '__main__':
    import sys

    print("\n请选择运行模式：")
    print("  1. 演示用例（验证真实样本预测）")
    print("  2. 全数据集批量评估")
    print("  3. 交互式预测（手动输入）")
    print("  4. 运行所有模式")

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = input("\n请输入模式编号（默认=4）：").strip() or '4'

    if mode == '1':
        run_demo_cases()
    elif mode == '2':
        run_full_dataset_evaluation()
    elif mode == '3':
        interactive_predict()
    elif mode == '4':
        run_demo_cases()
        run_full_dataset_evaluation()
        print("\n（跳过交互式输入模式，如需测试请运行 python obesity_predict.py 3）")
    else:
        print(f"未知模式 '{mode}'，运行演示用例...")
        run_demo_cases()
