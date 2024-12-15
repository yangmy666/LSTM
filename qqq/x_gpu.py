import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas_ta as ta

# 使用绝对路径读取数据
df = pd.read_csv(r'C:\py_project\LSTM\stock_data\QQQ.csv')

# 假设 'Date' 列是字符串类型，需要转换为日期类型
df['Date'] = pd.to_datetime(df['Date'])

# 数据预处理
df['DateTime'] = (df['Date'] - df['Date'].min()) / np.timedelta64(1, 'D')
df['Month'] = df['Date'].dt.month
df['Prev_Month'] = df['Month'].shift(1)
df['Prev_Open'] = df['Open'].shift(1)
df['Prev_Close'] = df['Close'].shift(1)
df['Prev_High'] = df['High'].shift(1)
df['Prev_Low'] = df['Low'].shift(1)
df['Prev_Volume'] = df['Volume'].shift(1)
df['Rise'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
df['Prev_Rise'] = df['Rise'].shift(1)
df['RSI'] = ta.rsi(df['Close'], length=14)
df['Prev_RSI'] = df['RSI'].shift(1)
df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
df['ATR_RATIO'] = df['ATR'] / df['Close'] * 100
df['Prev_ATR_RATIO'] = df['ATR_RATIO'].shift(1)
df['SMA_125'] = ta.sma(df['Close'], length=125)
df['Prev_SMA_125'] = df['SMA_125'].shift(1)
df['SMA_186'] = ta.sma(df['Close'], length=186)
df['Prev_SMA_186'] = df['SMA_186'].shift(1)

# 按日期排序
df = df.sort_values(by='Date')
df = df.dropna()

# 特征列
features = ['DateTime', 'Prev_Open', 'Prev_Close', 'Prev_High', 'Prev_Low', 'Prev_Rise',
            'Prev_Volume', 'Prev_RSI', 'Prev_ATR_RATIO', 'Prev_Month', 'Prev_SMA_125', 'Prev_SMA_186']
target = 'Close'

X = df[features]
y = df[target]

train_size = int(len(df) * 0.9)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 定义 XGBoost 参数，启用 GPU 加速
params = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',  # 使用 CPU/GPU 通用的 hist 方法
    'device': 'cuda',       # 明确指定使用 CUDA 设备（GPU）
}

# 转换为 DMatrix 格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# 训练初始模型
model = xgb.train(params, dtrain, num_boost_round=100)

# 滚动预测
predictions = []
for i in range(len(X_test)):
    # 获取当前时刻的测试数据
    X_current = X_test.iloc[i:i + 1]
    dcurrent = xgb.DMatrix(X_current)
    # 获取当前的 DateTime 值 9406
    current_dateTime = X_current['DateTime'].values[0]
    print(f"Current DateTime: {current_dateTime}")

    # 使用当前模型预测
    y_pred = model.predict(dcurrent)

    # 保存预测值
    predictions.append(y_pred[0])

    # 获取当前真实值
    X_current_label = y_test.iloc[i:i + 1]

    # 将当前预测的数据和真实值加入训练集
    X_train = pd.concat([X_train, X_current], ignore_index=True)
    y_train = pd.concat([y_train, X_current_label], ignore_index=True)

    # 更新训练数据的 DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # 使用新增数据重新训练模型
    model = xgb.train(params, dtrain, num_boost_round=100)

# 计算均方误差
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# 可视化预测结果与真实值
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='red')
plt.title('Stock Price Prediction (Actual vs Predicted)')
plt.xlabel('Sample Index')
plt.ylabel('Stock Close Price')
plt.legend()
plt.show()
