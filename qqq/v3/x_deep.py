import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from qqq.v3.data_treat import getDf, generate_feature_columns

#预测未来第几天
future_days=30
#根据最近几天的特征来预测
num_prev_days=30
#训练集比例
train_scale=0.7

# 读取数据
df = getDf('C:\py_project\LSTM\stock_data\\QQQ.csv',future_days,num_prev_days)

# 特征列（使用前一天的数据）
# 传入的特征列应该没有 'Prev_' 前缀
# features = ['DateTime', 'Prev_Open', 'Prev_Close', 'Prev_High', 'Prev_Low']
# features = ['Open', 'Close', 'Volume',
#             # **均线类 (Moving Averages)**
#             'SMA_14','SMA_125','Bull_Bear',
#             # **动量类 (Momentum Indicators)**
#             'RSI', 'WEEK_RSI',
#             # **趋势类 (Trend Indicators)**
#             'ADX', 'DI_PLUS', 'DI_MINUS',
#             # **波动性类 (Volatility Indicators)**
#             'ATR_RATIO',
#             # **成交量类 (Volume Indicators)**
#             'VOL_SMA_14', 'VOL_SMA_125',
# ]
features = ['Open', 'Close', 'High', 'Low', 'Volume','Month',
            # **均线类 (Moving Averages)**
            'SMA_14','SMA_125','SMA_186','Bull_Bear',
            'EMA_7', 'EMA_14', 'EMA_28',
            'EMA_56', 'EMA_112', 'EMA_224',
            'WMA', 'HMA', 'RMA',
            # **动量类 (Momentum Indicators)**
            'RSI', 'WEEK_RSI', 'MONTH_RSI',
            'KAMA', 'MACD', 'SIGNAL', 'HIST',
            'MOM', 'ROC',
            'WILLR', 'CCI',
            # **趋势类 (Trend Indicators)**
            'ADX', 'DI_PLUS', 'DI_MINUS',
            # **均值回归类 (Mean Reversion Indicators)**
            'BB_LOWER', 'BB_MIDDLE', 'BB_UPPER', 'BB_WIDTH', 'BB_PERCENT',
            'KC_LOWER', 'KC_MIDDLE', 'KC_UPPER',
            'DC_LOWER', 'DC_MIDDLE', 'DC_UPPER',
            # **波动性类 (Volatility Indicators)**
            'ATR', 'ATR_RATIO', 'HVOL',
            # **成交量类 (Volume Indicators)**
            'VWMA', 'OBV', 'CMF', 'AD',
            'VOL_SMA_14', 'VOL_SMA_125',
            'VOL_EMA_7', 'VOL_EMA_14', 'VOL_EMA_28',
            'VOL_EMA_56', 'VOL_EMA_112', 'VOL_EMA_224',
            # **统计类 (Statistical Indicators)**
            'SKEW', 'KURT', 'ZSCORE'
]

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

# XGBoost参数配置
params = {
    'objective': 'reg:squarederror',  # 适用于回归问题，二分类可以用'binary:logistic'
    'booster': 'gbtree',  # 使用树模型
    'eval_metric': 'rmse',  # 评估指标，这里使用均方根误差
    'learning_rate': 0.01,  # 学习率 (较低的学习率通常更精确，但需要更多树)
    # 'n_estimators': 1000,  # 最大树的数量
    'max_depth': 10,  # 树的最大深度
    'min_child_weight': 1,  # 叶节点最小权重
    'gamma': 0,  # 控制分裂的复杂度
    'subsample': 0.9,  # 每棵树训练时使用数据的比例
    'colsample_bytree': 0.9,  # 每棵树训练时使用特征的比例
    'lambda': 1,  # L2正则化项
    'alpha': 0,  # L1正则化项
    'scale_pos_weight': 1,  # 类别不平衡时调整
    # 'early_stopping_rounds': 50,  # 早停策略，如果验证集上的错误50轮没有改善，则停止训练
    'tree_method': 'hist',  # 使用 CPU/GPU 通用的 hist 方法
    'device': 'cuda',       # 明确指定使用 CUDA 设备（GPU）
    # 'predictor': 'gpu_predictor'  # 使用GPU加速进行预测
}

# 删除 y_test 中的 NaN 行，同时删除 X_test 中对应的行
valid_rows = y_test.dropna().index
X_test_cleaned = X_test.loc[valid_rows]
y_test_cleaned = y_test.loc[valid_rows]
# 创建 DMatrix 对象
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test_cleaned, label=y_test_cleaned)
# 设置 eval_set 以启用早停
evals = [(dtrain, 'train'), (dtest, 'eval')]
# 定义 XGBoost 模型,训练模型，指定验证集
print("开始训练-----")
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,  # 设置训练的轮数
    evals=evals,  # 设置验证集
    early_stopping_rounds=50  # 设置提前停止的轮数
)
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
for i in range(len(X_test)):
    # 获取当前时刻的测试特征
    X_current = X_test.iloc[i:i + 1]
    # 转换成DMatrix
    d_X_current = xgb.DMatrix(X_current)
    # 使用模型进行预测目标
    y_pred = model.predict(d_X_current)

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
