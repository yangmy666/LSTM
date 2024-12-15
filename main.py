import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import backtrader as bt

print('读取数据')
# 1. 读取数据
file_path = "C:\\py_project\\LSTM\\stock_data\\QQQ.csv"  # 确保文件在存储路径中

#,"Open","Close","High",'Low','Volume'
data = pd.read_csv(file_path, parse_dates=["Date"])
data.sort_values("Date", inplace=True)
data.set_index("Date", inplace=True)

print('数据预处理')
# 2. 数据预处理
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length, :])
        y.append(data[i + seq_length, 1])  # 1 refers to 'Close'
    return np.array(x), np.array(y)

SEQ_LENGTH = 60  # 例如指定值
X, y = create_sequences(data_scaled, SEQ_LENGTH)

print('构建简单的回归模型')
# 3. 构建简单的回归模型
X_flattened = X.reshape(X.shape[0], -1)  # 展平为二维数据
model = LinearRegression()

print('添加训练')
# 4. 添加训练
train_size = int(len(X_flattened) * 0.8)
X_train, X_test = X_flattened[:train_size], X_flattened[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model.fit(X_train, y_train)
predicted_prices = model.predict(X_test)

print('反归一化预测值')
# 反归一化预测值
predicted_prices = scaler.inverse_transform(
    np.concatenate((np.zeros((len(predicted_prices), data_scaled.shape[1] - 1)), predicted_prices.reshape(-1, 1)), axis=1)
)[:, -1]

# 5. 在 BackTrader 中实现量化策略
class RegressionStrategy(bt.Strategy):
    params = (('initial_cash', 10000),)

    def __init__(self):
        self.predicted = predicted_prices
        self.index = 0
        self.order = None

    def next(self):
        if self.index >= len(self.predicted):
            return

        if self.order:
            return  # 如果有未完成的订单，跳过

        if not self.position:  # 如果当前没有持仓
            if self.predicted[self.index] > self.data.close[0]:
                self.order = self.buy(size=self.broker.get_cash() // self.data.close[0])
        else:  # 如果当前有持仓
            if self.predicted[self.index] < self.data.close[0]:
                self.order = self.sell(size=self.position.size)

        self.index += 1

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

print('返测框架')
# 返测框架
cerebro = bt.Cerebro()
cerebro.broker.set_cash(10000)  # 初始资金 10000

data_feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(data_feed)
cerebro.addstrategy(RegressionStrategy)
cerebro.run()
cerebro.plot()
