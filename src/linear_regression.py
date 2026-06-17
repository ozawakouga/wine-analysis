import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv('data/wine.csv')
explanation = ['alcohol', 'volatile acidity']  # 説明変数
target = 'quality'                          # 目的変数

x = df[explanation].to_numpy()
y = df[target].to_numpy()

# バイアス項（切片）として1の列を先頭に足す
X = np.column_stack([np.ones(len(x)), x])

# 最小二乗法で係数を求める
w, *_ = np.linalg.lstsq(X, y, rcond=None)

# 予測値と決定係数R²
y_pred = X @ w
r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

print(f"R2 = {r2:.4f}")
for name, coef in zip(['bias'] + explanation, w):
    print(f"  {name}: {coef:.4f}")

# 予測値 vs 実測値の散布図
plt.scatter(y_pred, y, alpha=0.3)
plt.plot([3, 8], [3, 8], 'r-', label='ideal')
plt.xlabel('predicted quality')
plt.ylabel('actual quality')
plt.legend()
plt.tight_layout()
plt.savefig('linear_regression.png', dpi=150)
plt.show()