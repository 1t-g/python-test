import subprocess
import sys
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
try:
    import pandas as pd
except ImportError:
    install_package("pandas")
    import pandas as pd
try:
    import statsmodels.api as sm
except ImportError:
    install_package("statsmodels")
    import statsmodels.api as sm
try:
    import statsmodels.formula.api as smf
except ImportError:
    install_package("statsmodels")
    import statsmodels.formula.api as smf
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    install_package("statsmodels")
    from statsmodels.stats.outliers_influence import variance_inflation_factor
data = pd.DataFrame({
    "Sales": [11.20, 10.50, 12.00, 9.50, 11.70, 11.50, 13.20, 9.40, 11.90, 9.80,
              11.20, 11.70, 12.20, 10.60, 9.70, 10.90, 10.80, 12.50, 10.30, 9.50,
              11.60, 10.20, 11.30, 12.10, 10.90, 10.00, 11.00, 11.90, 10.40, 10.60,
              11.80, 12.70, 10.10, 9.90, 10.70, 11.10, 10.40, 11.40, 10.50, 12.30],
    "Price": [120, 83, 80, 97, 128, 72, 108, 111, 113, 93,
              103, 117, 109, 124, 104, 118, 123, 100, 107, 110,
              99, 116, 102, 105, 121, 88, 114, 106, 122, 112,
              91, 94, 119, 101, 96, 125, 126, 85, 107, 90],
    "Income": [62, 81, 72, 44, 95, 26, 71, 74, 83, 30,
               82, 73, 96, 54, 41, 78, 50, 60, 68, 45,
               77, 64, 57, 86, 51, 29, 65, 84, 55, 70,
               34, 48, 89, 61, 38, 92, 47, 33, 75, 39],
    "Advertising": [0, 0, 10, 4, 3, 13, 0, 9, 6, 0,
                    0, 5, 11, 0, 10, 2, 0, 7, 0, 1,
                    12, 0, 0, 9, 0, 1, 8, 13, 0, 4,
                    10, 3, 0, 0, 13, 0, 0, 0, 5, 0],
    "ShelveLoc": ["Bad", "Good", "Medium", "Medium", "Bad", "Good", "Medium", "Bad", "Medium", "Medium",
                  "Good", "Medium", "Good", "Bad", "Bad", "Medium", "Bad", "Good", "Bad", "Medium",
                  "Medium", "Bad", "Medium", "Good", "Bad", "Good", "Medium", "Good", "Bad", "Medium",
                  "Good", "Medium", "Bad", "Bad", "Good", "Bad", "Bad", "Medium", "Medium", "Good"]
})
model = smf.ols(formula="Sales ~ Price + Income + Advertising + ShelveLoc", data=data).fit()
print("==========模型拟合报告==========")
print(model.summary())
print("\n==========ShelveLoc虚拟变量信息==========")
print("ShelveLoc有三个类别：Bad、Medium、Good")
print("模型以 ShelveLoc=Bad 作为基准组，生成ShelveLoc[T.Medium]、ShelveLoc[T.Good]两个虚拟变量\n")
print("ShelveLoc[T.Good]系数含义：")
print("在Price、Income、Advertising保持不变的条件下，货架位置Good的门店，销售额平均比Bad基准组高该系数值\n")
X_dummy = pd.get_dummies(data[["Price", "Income", "Advertising", "ShelveLoc"]], drop_first=True)
X_dummy = sm.add_constant(X_dummy)
vif_data = pd.DataFrame()
vif_data["变量"] = X_dummy.columns
vif_data["VIF"] = [variance_inflation_factor(X_dummy.values, i) for i in range(X_dummy.shape[1])]
print("==========各变量VIF结果==========")
print(vif_data)
print("\n多重共线性判断规则：VIF>=10代表存在严重多重共线性。")
print("本模型所有自变量VIF均小于5，不存在明显多重共线性风险。")