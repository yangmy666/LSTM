import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from qqq.v2.data_treat import getDf

#预测未来第几天
future_days=5

# 读取数据
df = getDf('C:\py_project\LSTM\stock_data\\QQQ.csv',future_days)

# 特征列（使用前一天的数据）
#ALL
features = ['DateTime','Month','Prev_Month',
            'Prev_Open', 'Prev_Close', 'Prev_High', 'Prev_Low','Prev_Volume',
            # **均线类 (Moving Averages)**
            'Prev_SMA_14',
            'Prev_EMA_7', 'Prev_EMA_14', 'Prev_EMA_28',
            'Prev_EMA_56','Prev_EMA_112','Prev_EMA_224',
            'Prev_WMA', 'Prev_HMA', 'Prev_RMA',
            # **动量类 (Momentum Indicators)**
            'Prev_RSI', 'Prev_WEEK_RSI', 'Prev_MONTH_RSI',
            'Prev_KAMA', 'Prev_MACD', 'Prev_SIGNAL', 'Prev_HIST',
            'Prev_MOM', 'Prev_ROC',
            'Prev_WILLR', 'Prev_CCI',
            # **趋势类 (Trend Indicators)**
            'Prev_ADX', 'Prev_DI_PLUS', 'Prev_DI_MINUS',
            # **均值回归类 (Mean Reversion Indicators)**
            'Prev_BB_LOWER', 'Prev_BB_MIDDLE', 'Prev_BB_UPPER', 'Prev_BB_WIDTH', 'Prev_BB_PERCENT',
            'Prev_KC_LOWER', 'Prev_KC_MIDDLE', 'Prev_KC_UPPER',
            'Prev_DC_LOWER', 'Prev_DC_MIDDLE', 'Prev_DC_UPPER',
            # **波动性类 (Volatility Indicators)**
            'Prev_ATR', 'Prev_ATR_RATIO', 'Prev_HVOL',
            # **成交量类 (Volume Indicators)**
            'Prev_VWMA', 'Prev_OBV', 'Prev_CMF', 'Prev_AD',
            'Prev_VOL_EMA_7', 'Prev_VOL_EMA_14', 'Prev_VOL_EMA_28',
            'Prev_VOL_EMA_56', 'Prev_VOL_EMA_112', 'Prev_VOL_EMA_224',
            # **统计类 (Statistical Indicators)**
            'Prev_SKEW', 'Prev_KURT', 'Prev_ZSCORE'
]

# features = ['DateTime', 'Prev_Open', 'Prev_Close', 'Prev_High', 'Prev_Low']

# 目标列（预测未来的收盘价）
target = 'Close'

# 分割数据集为特征和目标
X = df[features]
y = df[target]

# 按时间划分训练集和测试集
train_size = int(len(df) * 0.95)
X_train, y_train = X[:train_size] , y[:train_size]
X_test, y_test = X[train_size:] , y[train_size:]
df_test=df[train_size:]

# 检查 NaN 或 Infinity
if y_train.isna().any() or np.isinf(y_train).any():
    print("标签数据中存在 NaN 或 Infinity!")

# 定义 XGBoost 模型d
model = xgb.XGBRegressor(objective='reg:squarederror')
# 训练模型
model.fit(X_train, y_train)

# 绘制特征重要性图
xgb.plot_importance(model, importance_type='weight', max_num_features=len(X.columns))
plt.show()

print("开始预测-----")

# 用来存储预测值
y_preds = []
# 测试集K线图
k_test = pd.DataFrame(columns=['Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'y_pred'])

# 滚动预测过程
for i in range(len(X_test)):
    # 获取当前时刻的测试特征
    X_current = X_test.iloc[i:i + 1]

    # 使用模型进行预测目标
    y_pred = model.predict(X_current)

    # 获取当前的测试数据
    df_current = df_test.iloc[i:i + 1]
    date = pd.to_datetime(df_current['Date'].values[0])  # 强制转换为 datetime 类型

    #最后未知目标的行不训练了只预测用
    if i >= len(X_test)-future_days:
        # 创建新行作为临时DataFrame
        new_row = pd.DataFrame([[date, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, y_pred[0]]],
                               columns=k_test.columns)
        # 删除全空的列
        new_row = new_row.dropna(axis=1, how='all')
        # 使用pd.concat拼接
        k_test = pd.concat([k_test, new_row], ignore_index=True)
    else:
        # 保存预测值
        y_preds.append(y_pred[0])

        open_ = df_current['Open'].values[0]
        close = df_current['Close'].values[0]
        high = df_current['High'].values[0]
        low = df_current['Low'].values[0]
        volume = df_current['Volume'].values[0]
        # 将数据添加到 k_test 中
        k_test.loc[len(k_test)] = [date, open_, close, high, low, volume, y_pred[0]]

        # 获取真实的目标值
        y_current = y_test.iloc[i:i + 1]

        # 将当前真实数据和目标值添加到训练集
        X_train = pd.concat([X_train, X_current], ignore_index=True)
        y_train = np.append(y_train, y_current)

        # 使用新的训练数据重新训练模型
        #model.fit(X_train, y_train)

y_test = y_test.dropna()
# 计算均方误差
mse = mean_squared_error(y_test, y_preds)
# 计算标准差
std_dev = np.sqrt(mse)
# 计算平均值
average_price = y_test.mean()
# 计算误差百分比
error_percentage = (std_dev / average_price) * 100
# 打印结果
print(f"MSE: {mse:.2f}")
print(f"标准差: {std_dev:.2f}")
print(f"测试集目标值均值: {average_price:.2f}")
print(f"误差百分比: {error_percentage:.2f}%")

# 确保 'Date' 列是 datetime 类型，并设置为索引
k_test['Date'] = pd.to_datetime(k_test['Date'])  # 强制转换为 datetime 类型
k_test.set_index('Date', inplace=True)
# 创建 K 线图
y_pred = mpf.make_addplot(k_test['y_pred'], color='orange', width=1)
mpf.plot(k_test, type='candle', style='charles', title='K-line Chart', ylabel='Price', volume=True, returnfig=True,addplot=[y_pred])
# 显示图形
plt.show()

# #y_preds=model.predict(X_test)
# # 可视化预测结果与真实值
# plt.figure(figsize=(10, 6))
# plt.plot(y_test.values, label='Actual', color='blue')
# plt.plot(y_preds, label='Predicted', color='red')
# plt.title('Stock Price Prediction (Actual vs Predicted)')
# plt.xlabel('Sample Index')
# plt.ylabel('Stock Close Price')
# plt.legend()
# plt.show()
