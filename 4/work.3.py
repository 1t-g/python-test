# -*- coding:utf-8 -*-
"""
SVM二分类实验
使用模拟数据集完成有监督数据分类任务
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings("ignore")

# ====================== 1.生成模拟数据集（无需联网，无需外部文件） ======================
# 生成二分类数据集，模拟4600样本，57个特征，和spambase规模接近
X, y = make_classification(
    n_samples=4600,
    n_features=57,
    n_informative=30,
    n_classes=2,
    random_state=42
)

print(f"数据集总样本数：{X.shape[0]}, 特征数：{X.shape[1]}")
print(f"类别1占比：{np.mean(y):.2%}")

# ====================== 2.数据集划分 + 标准化 ======================
# SVM对特征尺度极度敏感，必须标准化！
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================== 3.课件参数模型训练 ======================
# ①线性SVM C=1
print("\n===== 线性SVM C=1 =====")
lin_svc = LinearSVC(C=1, max_iter=10000, random_state=42)
lin_svc.fit(X_train_scaled, y_train)
y_pred_lin = lin_svc.predict(X_test_scaled)
acc_lin = accuracy_score(y_test, y_pred_lin)
print(f"测试集准确率: {acc_lin:.4f}")
print(classification_report(y_test, y_pred_lin))

# ②RBF核SVM C=10, gamma=0.01
print("\n===== RBF核SVM C=10, gamma=0.01 =====")
rbf_svc = SVC(kernel="rbf", C=10, gamma=0.01, random_state=42)
rbf_svc.fit(X_train_scaled, y_train)
y_pred_rbf = rbf_svc.predict(X_test_scaled)
acc_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"测试集准确率: {acc_rbf:.4f}")
print(classification_report(y_test, y_pred_rbf))

# ====================== 4.网格搜索最优超参数 ======================
print("\n===== 网格搜索调参RBF SVM =====")
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "gamma": [1e-4, 1e-3, 0.01, 0.1, 1]
}
grid = GridSearchCV(
    estimator=SVC(kernel="rbf"),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring="accuracy"
)
grid.fit(X_train_scaled, y_train)
print(f"最优参数: {grid.best_params_}")
print(f"5折交叉验证最优得分: {grid.best_score_:.4f}")

best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
acc_best = accuracy_score(y_test, y_pred_best)
print(f"最优模型测试集准确率: {acc_best:.4f}")
print(classification_report(y_test, y_pred_best))

# ====================== 5.支持向量统计 ======================
print(f"\nRBF模型支持向量数量：{rbf_svc.n_support_}")
print(f"类别0支持向量:{rbf_svc.n_support_[0]}, 类别1支持向量:{rbf_svc.n_support_[1]}")

# ====================== 6.混淆矩阵可视化 ======================
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("SVM混淆矩阵")
plt.colorbar()
classes = ["类别0","类别1"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,cm[i][j],ha="center",va="center",color="black")
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
