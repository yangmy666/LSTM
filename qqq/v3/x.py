import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from qqq.v3.data_treat import getDf, generate_feature_columns

#预测未来第几天
future_days=10
#根据最近几天的特征来预测
num_prev_days=1
#训练集比例
train_scale=0.95

# 读取数据
df = getDf('C:\py_project\LSTM\stock_data\\QQQ.csv',future_days,num_prev_days)

# 特征列（使用前一天的数据）
# 传入的特征列应该没有 'Prev_' 前缀
features = ['Open', 'Close', 'Volume','Month',
            # **均线类 (Moving Averages)**
            'SMA_14','SMA_125','Bull_Bear',
            # **动量类 (Momentum Indicators)**
            'RSI', 'WEEK_RSI',
            # **趋势类 (Trend Indicators)**
            'ADX', 'DI_PLUS', 'DI_MINUS',
            # **波动性类 (Volatility Indicators)**
            'ATR_RATIO',
            # **成交量类 (Volume Indicators)**
            'VOL_SMA_14', 'VOL_SMA_125',
]
# features = ['Open', 'Close', 'High', 'Low', 'Volume','Month',
#             # **均线类 (Moving Averages)**
#             'SMA_14','SMA_125','SMA_186','Bull_Bear',
#             'EMA_7', 'EMA_14', 'EMA_28',
#             'EMA_56', 'EMA_112', 'EMA_224',
#             'WMA', 'HMA', 'RMA',
#             # **动量类 (Momentum Indicators)**
#             'RSI', 'WEEK_RSI', 'MONTH_RSI',
#             'KAMA', 'MACD', 'SIGNAL', 'HIST',
#             'MOM', 'ROC',
#             'WILLR', 'CCI',
#             # **趋势类 (Trend Indicators)**
#             'ADX', 'DI_PLUS', 'DI_MINUS',
#             # **均值回归类 (Mean Reversion Indicators)**
#             'BB_LOWER', 'BB_MIDDLE', 'BB_UPPER', 'BB_WIDTH', 'BB_PERCENT',
#             'KC_LOWER', 'KC_MIDDLE', 'KC_UPPER',
#             'DC_LOWER', 'DC_MIDDLE', 'DC_UPPER',
#             # **波动性类 (Volatility Indicators)**
#             'ATR', 'ATR_RATIO', 'HVOL',
#             # **成交量类 (Volume Indicators)**
#             'VWMA', 'OBV', 'CMF', 'AD',
#             'VOL_SMA_14', 'VOL_SMA_125',
#             'VOL_EMA_7', 'VOL_EMA_14', 'VOL_EMA_28',
#             'VOL_EMA_56', 'VOL_EMA_112', 'VOL_EMA_224',
#             # **统计类 (Statistical Indicators)**
#             'SKEW', 'KURT', 'ZSCORE'
# ]

# 动态生成特征列
features = generate_feature_columns(features, num_prev_days)
features.extend(['DateTime',])

# 目标列（预测未来的收盘价）
target = 'Close'

# 分割数据集为特征和目标
X = df[features]
y = df[target]

# 确定训练集和测试集大小
train_size = int(len(df) * train_scale)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]
df_test = df[train_size:]

# 检查 NaN 或 Infinity
if y_train.isna().any() or np.isinf(y_train).any():
    print("标签数据中存在 NaN 或 Infinity!")

# 定义 XGBoost 模型,训练模型
print("开始训练-----")
# 定义 XGBoost 模型
model = xgb.XGBRegressor(objective='reg:squarederror')
# 训练模型
model.fit(X_train, y_train)

# 保存模型
model.save_model('C:\py_project\LSTM\model\\xgb_model.json')
# 绘制特征重要性图
# xgb.plot_importance(model,importance_type='gain',max_num_features=10)
# plt.show()

print("开始测试-----")

# 用来存储预测值
y_preds = []
# 测试集K线图
k_test = pd.DataFrame(columns=['Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'y_pred'])

# 滚动预测过程
for i in range(future_days-1,len(X_test)):
    # 获取当前时刻的测试特征
    X_current = X_test.iloc[i:i + 1]
    # 使用模型进行预测目标
    y_pred = model.predict(X_current)

    # 获取当前的测试数据
    df_current = df_test.iloc[i:i + 1]
    date = pd.to_datetime(df_current['Date'].values[0])  # 强制转换为 datetime 类型

    if i >= len(X_test)-future_days:
        # 创建新行作为临时DataFrame
        new_row = pd.DataFrame([[date, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, y_pred[0]]],
                               columns=k_test.columns)
        # 删除全空的列
        new_row = new_row.dropna(axis=1, how='all')
        # 使用pd.concat拼接
        k_test = pd.concat([k_test, new_row], ignore_index=True)
    else:
        # 保存预测值用于计算MSE
        y_preds.append(y_pred[0])
        # 保存到k_test用于展示
        open_ = df_current['Open'].values[0]
        close = df_current['Close'].values[0]
        high = df_current['High'].values[0]
        low = df_current['Low'].values[0]
        volume = df_current['Volume'].values[0]
        # 将数据添加到 k_test 中
        k_test.loc[len(k_test)] = [date, open_, close, high, low, volume, y_pred[0]]

    # 使用最新的训练数据重新训练模型
    if i<len(X_test)-1:
        # 将future_days-1天前的最新真实数据和目标值添加到训练集
        # 为什么用future_days-1天前的而不是当前的？如果用当前的就泄露未来数据了
        X_train_last = X_test.iloc[i - future_days + 1:i - future_days + 2]
        y_train_last = y_test.iloc[i - future_days + 1:i - future_days + 2]
        X_train = pd.concat([X_train, X_train_last], ignore_index=True)
        y_train = np.append(y_train, y_train_last)

        df_train_last = df_test.iloc[i - future_days + 1:i - future_days + 2]
        train_last_date = pd.to_datetime(df_train_last['Date'].values[0])  # 强制转换为 datetime 类型
        print(train_last_date.strftime('%Y-%m-%d'))

        model.fit(X_train, y_train)

y_test = y_test.dropna().iloc[future_days-1:]
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


