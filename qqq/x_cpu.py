import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas_ta as ta
import pandas as pd
import mplfinance as mpf
from qqq.method import getDf

# 读取数据
df = getDf('C:\py_project\LSTM\stock_data\QQQ.csv')

# 特征列（使用前一天的数据）
features = ['DateTime','Prev_Month', 'Prev_Open', 'Prev_Close', 'Prev_High', 'Prev_Low',
            'Prev_Volume', 'Prev_SMA', 'Prev_EMA', 'Prev_WMA',
            'Prev_HMA', 'Prev_RMA', 'Prev_ADX', 'Prev_DI_PLUS', 'Prev_DI_MINUS',
            'Prev_KAMA', 'Prev_RSI', 'Prev_WILLR', 'Prev_CCI', 'Prev_MOM', 'Prev_ROC', 'Prev_BB_LOWER',
            'Prev_BB_MIDDLE', 'Prev_BB_UPPER', 'Prev_BB_WIDTH', 'Prev_BB_PERCENT',
            'Prev_KC_LOWER', 'Prev_KC_MIDDLE', 'Prev_KC_UPPER', 'Prev_DC_LOWER', 'Prev_DC_MIDDLE',
            'Prev_DC_UPPER', 'Prev_ATR', 'Prev_ATR_RATIO', 'Prev_HVOL', 'Prev_VWMA', 'Prev_OBV',
            'Prev_CMF', 'Prev_AD', 'Prev_SKEW', 'Prev_KURT', 'Prev_ZSCORE'
]

# features = ['Prev_Open', 'Prev_Close', 'Prev_High', 'Prev_Low','Prev_Rise']

# 目标列（预测第二天的收盘价）
target = 'Close'

# 分割数据集为特征和目标
X = df[features]
y = df[target]

# 按时间划分训练集和测试集
train_size = int(len(df) * 0.97)
X_train, y_train = X[:train_size] , y[:train_size]

X_test, y_test = X[train_size:] , y[train_size:]
d_test=df[train_size:]

# 定义 XGBoost 模型
model = xgb.XGBRegressor(objective='reg:squarederror')

# 训练模型
model.fit(X_train, y_train)

# 用来存储预测值
predictions = []
# 测试集K线图
k_test = pd.DataFrame(columns=['Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'y_pred'])

# 滚动预测过程
for i in range(len(X_test)):
    # 获取当前时刻的测试特征
    X_current = X_test.iloc[i:i + 1]

    # 使用模型进行预测目标
    y_pred = model.predict(X_current)

    # 保存预测值
    predictions.append(y_pred[0])

    # 获取当前的测试数据
    d_current = d_test.iloc[i:i + 1]
    date = pd.to_datetime(d_current['Date'].values[0])  # 强制转换为 datetime 类型

    #最后一行不训练了
    if i == len(X_test)-1:
        # 创建新行作为临时DataFrame
        new_row = pd.DataFrame([[date, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, y_pred[0]]],
                               columns=k_test.columns)
        # 删除全空的列
        new_row = new_row.dropna(axis=1, how='all')
        # 使用pd.concat拼接
        k_test = pd.concat([k_test, new_row], ignore_index=True)
        break
    else:
        open_ = d_current['Open'].values[0]
        close = d_current['Close'].values[0]
        high = d_current['High'].values[0]
        low = d_current['Low'].values[0]
        volume = d_current['Volume'].values[0]
        # 将数据添加到 k_test 中
        k_test.loc[len(k_test)] = [date, open_, close, high, low, volume, y_pred[0]]


    # 获取真实的目标值
    y_current = y_test.iloc[i:i + 1]

    # 将当前真实数据和目标值添加到训练集
    X_train = pd.concat([X_train, X_current], ignore_index=True)
    y_train = np.append(y_train, y_current)

    # 使用新的训练数据重新训练模型
    model.fit(X_train, y_train)

# 确保 'Date' 列是 datetime 类型，并设置为索引
k_test['Date'] = pd.to_datetime(k_test['Date'])  # 强制转换为 datetime 类型
k_test.set_index('Date', inplace=True)

# 创建 K 线图
y_pred = mpf.make_addplot(k_test['y_pred'], color='orange', width=1)
mpf.plot(k_test, type='candle', style='charles', title='K-line Chart', ylabel='Price', volume=True, returnfig=True,addplot=[y_pred])
# 显示图形
plt.show()

# predictions=model.predict(X_test)
# # 计算均方误差
# # mse = mean_squared_error(y_test, predictions)
# # print(f"Mean Squared Error: {mse:.2f}")
# # 可视化预测结果与真实值
# plt.figure(figsize=(10, 6))
# plt.plot(y_test.values, label='Actual', color='blue')
# plt.plot(predictions, label='Predicted', color='red')
# plt.title('Stock Price Prediction (Actual vs Predicted)')
# plt.xlabel('Sample Index')
# plt.ylabel('Stock Close Price')
# plt.legend()
# plt.show()
