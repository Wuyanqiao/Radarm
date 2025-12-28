import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# 机器学习核心库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def preprocess_data(df, target=None):
    """
    自动化数据预处理：
    1. 填充缺失值
    2. 将非数值列转为数值 (Label Encoding)
    3. 分离特征(X)和目标(y)
    """
    df_clean = df.copy()
    
    # 1. 简单填充缺失值
    # 数值列填均值，非数值列填众数
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='mean')
        df_clean[num_cols] = imputer_num.fit_transform(df_clean[num_cols])
    
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_clean[cat_cols] = imputer_cat.fit_transform(df_clean[cat_cols])

    # 2. 编码非数值特征 (把文字变成数字)
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        le_dict[col] = le

    # 3. 分离 X 和 y
    if target:
        if target not in df_clean.columns:
            raise ValueError(f"目标列 '{target}' 不存在")
        X = df_clean.drop(columns=[target])
        y = df_clean[target]
        return X, y, le_dict
    else:
        return df_clean, None, le_dict

def save_plot_to_base64():
    """ 辅助函数：将 Matplotlib 图片转为 Base64 """
    if plt.get_fignums():
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_str
    return None

def run(df, target=None, task='regression', k=3):
    """
    统一入口函数
    :param df: Pandas DataFrame
    :param target: 目标列名 (聚类任务不需要)
    :param task: 'regression' | 'classification' | 'clustering'
    :param k: 聚类数量 (仅 clustering 有效)
    """
    plt.clf() # 清空画布
    result_text = ""
    
    try:
        # --- 任务 1: 回归 (预测数值) ---
        if task == 'regression':
            if not target: return "错误：回归任务必须指定 target 目标列"
            X, y, _ = preprocess_data(df, target)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 使用随机森林回归，因为它稳健且无需缩放
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 评估
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            result_text = (
                f"### 回归分析报告 (目标: {target})\n"
                f"- **模型**: Random Forest Regressor\n"
                f"- **R² (拟合优度)**: {r2:.4f} (越接近 1 越好)\n"
                f"- **MSE (均方误差)**: {mse:.4f}\n"
                f"- **测试集样本数**: {len(y_test)}"
            )
            
            # 绘图：真实值 vs 预测值
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # 对角线
            plt.xlabel('真实值 (Actual)')
            plt.ylabel('预测值 (Predicted)')
            plt.title(f'回归分析: {target} 预测效果图')

        # --- 任务 2: 分类 (预测类别) ---
        elif task == 'classification':
            if not target: return "错误：分类任务必须指定 target 目标列"
            X, y, le_dict = preprocess_data(df, target)
            
            # 检查类别数量，太多则不适合分类
            if len(np.unique(y)) > 20:
                return f"错误：目标列 '{target}' 的唯一值过多 ({len(np.unique(y))})，看起来像连续数值，建议使用回归分析。"

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            
            # 尝试还原标签名称
            target_le = le_dict.get(target)
            class_names = [str(c) for c in np.unique(y)]
            if target_le:
                try:
                    class_names = [str(c) for c in target_le.inverse_transform(np.unique(y))]
                except: pass

            result_text = (
                f"### 分类分析报告 (目标: {target})\n"
                f"- **模型**: Random Forest Classifier\n"
                f"- **准确率 (Accuracy)**: {acc:.2%}\n"
            )

            # 绘图：混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('预测类别')
            plt.ylabel('真实类别')
            plt.title(f'分类分析: {target} 混淆矩阵')

        # --- 任务 3: 聚类 (自动分组) ---
        elif task == 'clustering':
            X, _, _ = preprocess_data(df, target=None) # 聚类不需要 y
            
            # K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            
            result_text = (
                f"### 聚类分析报告 (K-Means)\n"
                f"- **聚类簇数 (k)**: {k}\n"
                f"- **样本总数**: {len(df)}\n"
                f"- **各簇样本分布**: {dict(pd.Series(clusters).value_counts())}"
            )
            
            # 绘图：使用 PCA 降维到 2D 进行可视化
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title(f'K-Means 聚类结果 (k={k})')
            plt.colorbar(scatter, label='Cluster ID')

        else:
            return "错误：未知的任务类型。请选择 regression, classification 或 clustering。"

        # 保存图表
        img_base64 = save_plot_to_base64()
        return f"{result_text}\n\n[系统注]：图表已生成。"
        
    except Exception as e:
        import traceback
        return f"机器学习执行报错:\n{str(e)}\n{traceback.format_exc()}"