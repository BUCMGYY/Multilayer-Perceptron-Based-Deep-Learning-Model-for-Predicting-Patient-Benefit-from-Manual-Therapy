import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# 读取数据
data = pd.read_csv('newdata.csv')
data.head()
categorical_columns = [
    'Gender',
    'spring test',
    'Presence of muscle tightness',
    'Exacerbation on Flexion',
    'Exacerbation on Extension',
    'Exacerbation on Lateral Flexion',
    'Exacerbation on Rotation'
]  # 根据你的实际数据修改分类列名

numerical_columns = [col for col in data.columns if col not in categorical_columns]
# 分离特征和标签
X = data.drop(columns=["Prediction"])  # 根据实际目标列名修改
y = data["Prediction"]
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 定义分类列和数值列
categorical_columns = [
    'Gender',
    'spring test',
    'Presence of muscle tightness',
    'Exacerbation on Flexion',
    'Exacerbation on Extension',
    'Exacerbation on Lateral Flexion',
    'Exacerbation on Rotation'
]  # 根据你的实际数据修改分类列名

numerical_columns = [col for col in X.columns if col not in categorical_columns]

# 创建一个预处理器：对数值数据进行标准化，对分类数据进行独热编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),  # 对数值数据进行标准化
        ('cat', OneHotEncoder(), categorical_columns)   # 对分类数据进行独热编码
    ])

# 对数据进行预处理：拟合和转换
X_scaled = preprocessor.fit_transform(X)

# 将处理后的数据转回DataFrame，保持列名
ohe_columns = preprocessor.transformers_[1][1].get_feature_names_out(categorical_columns)
new_columns = numerical_columns + list(ohe_columns)

X_scaled = pd.DataFrame(X_scaled, columns=new_columns)

# 输出前几行结果
print(X_scaled.head())

# 划分数据集
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 使用最佳的参数配置初始化模型
model = MLPClassifier(
    activation='relu',
    alpha=0.1,
    batch_size=128,
    hidden_layer_sizes=(64, 32),
    learning_rate='constant',
    max_iter=200,
    solver='adam',
    random_state=42
)
 # 训练模型
model.fit(X_train, y_train)
import shap

# 使用 TreeExplainer 对 MLP 模型进行解释（这里使用的是模型的输出）
explainer = shap.KernelExplainer(model.predict_proba, X_train)

# 计算 SHAP 值，返回每个样本每个特征的 SHAP 值
shap_values = explainer.shap_values(X_train)
print(shap_values)
# 计算每个特征的重要性
shap_importance = np.abs(shap_values[1]).mean(axis=0)  # 如果是二分类任务，通常我们关注 SHAP 值的第二类（[1]）

# Create feature importance DataFrame
if len(shap_importance) == X_train.shape[1]:
    shap_feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'SHAP Importance': shap_importance
    })

    # Select top 14 features
    top_14_features = shap_feature_importance.sort_values(by='SHAP Importance', ascending=False).head(14)
    print(top_14_features)

    # Select top features
    selected_features = top_14_features['Feature'].values
    X_selected = X_train[selected_features]

    # Retrain model with selected features
    model_selected = MLPClassifier(
        activation='relu',
        alpha=0.1,
        batch_size=128,
        hidden_layer_sizes=(64, 32),
        learning_rate='constant',
        max_iter=200,
        solver='adam',
        random_state=42
    )
    model_selected.fit(X_selected, y_train)

    # Save the new model
    with open('MLP_model_selected.pkl', 'wb') as file:
        pickle.dump(model_selected, file)

else:
    print("Mismatch in SHAP importance and X_train column lengths")
    print(f"SHAP importance length: {len(shap_importance)}")
    print(f"X_train columns length: {X_train.shape[1]}")