import backtrader as bt  # 导入 Backtrader 库，用于进行回测
from datetime import datetime  # 导入 datetime 库，用于处理日期和时间
import pandas as pd  # 导入 pandas 库，用于处理数据和读取CSV文件
import os  # 导入 os 库，用于文件和目录操作

# 创建一个名为 QQQStrategy 的策略类，继承自 Backtrader 的 Strategy 类
class QQQStrategy(bt.Strategy):
    # 定义策略的参数列表，可以在回测时调整这些参数的值
    params = (
        ('shortLine', 24),
        ('mediumLine', 59),
        ('longLine', 110),
        ('CallBackLine', 125),
        ('bullBearLine', 186),
        ('generalFastLine', 6),
        ('generalSlowLine', 23),
        ('generalBuyAtrRatio', 3.1),
        ('generalSellAtrRatio', 4.8),
        ('bull_generalBuyRsi', 60.7),
        ('longAtrRatio', 1.8),
        ('dailyDropAtrRatioMultiple', 1.25),
        ('dailyDropRsi', 42.7),
        ('callBackRsi', 37.3),
        ('callBackAtrRatio', 2.4),
        ('newVhDays', 98),
        ('breakThroughAtrRatio', 2.8),
        ('bearDailyRise_openAtrRatioMultiple', 1.7),
        ('neckRatio', 3.4),
        ('neckRsi', 63.4),
        ('bull_volBearishRsi', 82.1),
        ('bear_volBearishRsi', 76.5),
        ('bull_volBullishRsi', 36.2),
        ('bear_volBullishRsi', 25.6),
        ('pinBarAtrMuplti', 0.79),
        ('pinBarRsi', 41.0),
        ('supportRsi', 43.5),
        ('callBack_crisis', 9.2),
        ('bear_vol_dr_close_1', 1.8),
        ('bear_vol_dr_close_1_rsi', 49.2),
        ('bull_vol_dr_close_1', 2.8),
        ('bull_vol_dr_close_1_rsi', 61.2),
        ('volSmaDays', 14),
        ('start_date', datetime(1900, 1, 1)),
        ('end_date', datetime(2030, 1, 1))
    )

    def __init__(self):  # 策略的初始化方法
        # Moving Averages - 创建几个移动平均线指标
        self.smaShort = bt.indicators.SMA(self.data.close, period=self.params.shortLine)  # 短期SMA
        self.smaMedium = bt.indicators.SMA(self.data.close, period=self.params.mediumLine)  # 中期SMA
        self.smaLong = bt.indicators.SMA(self.data.close, period=self.params.longLine)  # 长期SMA
        self.smaCallBack = bt.indicators.SMA(self.data.close, period=self.params.CallBackLine)  # 回调线
        self.smaBullBear = bt.indicators.SMA(self.data.close, period=self.params.bullBearLine)  # 牛熊线
        self.smaGeneralFast = bt.indicators.SMA(self.data.close, period=self.params.generalFastLine)  # 一般的快速SMA
        self.smaGeneralSlow = bt.indicators.SMA(self.data.close, period=self.params.generalSlowLine)  # 一般的慢速SMA

        # RSI - 相对强弱指数
        self.rsi = bt.indicators.RSI(self.data.close, period=14)

        # ATR - 平均真实波幅
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.atrRatio = self.atr / self.data.close * 100  # ATR比率

        # Volume SMA - 成交量的SMA
        self.volSma = bt.indicators.SMA(self.data.volume, period=self.params.volSmaDays)

        # Variables for tracking conditions - 用于追踪条件的变量
        self.chance_CrossUp = False  # 是否有向上的交叉信号
        self.order = None  # 当前没有订单

    def next(self):  # 每个时间步长的逻辑
        current_time = self.data.datetime.datetime(0)  # 获取当前时间
        if current_time < self.params.start_date or current_time > self.params.end_date:  # 如果当前时间不在设定的日期范围内，则跳过
            return

        if self.order:  # 如果当前有未完成的订单，则跳过
            return

        # General Crossover Logic - 一般的交叉逻辑
        crossUp = self.smaGeneralFast[0] > self.smaGeneralSlow[0] and self.smaGeneralFast[-1] <= self.smaGeneralSlow[-1]  # 快速SMA上穿慢速SMA
        crossDown = self.smaGeneralFast[0] < self.smaGeneralSlow[0] and self.smaGeneralFast[-1] >= self.smaGeneralSlow[-1]  # 快速SMA下穿慢速SMA

        # Update the chance_CrossUp flag - 更新交叉信号的标志
        if crossUp and not (self.data.close[0] > self.smaBullBear[0] and (self.data.close[0] - self.smaCallBack[0]) / self.smaCallBack[0] * 100 > self.params.callBack_crisis):
            self.chance_CrossUp = True

        if crossDown:  # 如果发生下穿信号
            self.chance_CrossUp = False

        # Trading Logic - 交易逻辑
        if self.position:  # 如果已经有持仓
            # Sell logic: Bearish Crossover - 卖出逻辑：如果发生熊市交叉（快速SMA下穿慢速SMA）
            if crossDown:
                if self.atrRatio[0] > self.params.generalSellAtrRatio:  # 如果ATR比率大于卖出条件
                    self.order = self.close()  # 平仓
                    return
        else:  # 如果没有持仓
            # Buy logic: Bull Market - 买入逻辑：如果发生牛市条件
            if self.data.close[0] > self.smaBullBear[0]:  # 如果当前价格大于牛熊线
                if self.chance_CrossUp:  # 如果存在向上的交叉信号
                    if self.atrRatio[0] < self.params.generalBuyAtrRatio:  # 如果ATR比率小于买入条件
                        if self.rsi[0] < self.params.bull_generalBuyRsi:  # 如果RSI小于指定的值
                            self.chance_CrossUp = False
                            self.order = self.buy(size=int(self.broker.getcash() / self.data.close[0]))  # 根据现金量买入
                            return

    def notify_order(self, order):  # 订单通知方法
        if order.status in [order.Completed, order.Canceled, order.Rejected]:  # 如果订单已完成、已取消或已拒绝
            self.order = None  # 重置订单状态

# 创建一个回测引擎
cerebro = bt.Cerebro()
cerebro.addstrategy(QQQStrategy)  # 将策略添加到引擎中

cerebro.broker.setcash(100000)  # 设置初始资金为100,000

# Load data from CSV - 从CSV文件加载数据
file_path = '/stock_data/QQQ.csv'  # 数据文件路径
if os.path.exists(file_path):  # 检查文件是否存在
    dataframe = pd.read_csv(file_path, parse_dates=['Date'])  # 读取CSV文件
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])  # 转换日期列
    dataframe.set_index('Date', inplace=True)  # 将日期列设为索引

    # Create a data feed - 创建数据源
    data = bt.feeds.PandasData(
        dataname=dataframe,
        datetime=None,
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1  # 如果没有OpenInterest列，使用-1
    )

    cerebro.adddata(data)  # 将数据源添加到引擎中
else:
    print(f"File not found: {file_path}. Skipping data load.")  # 如果文件未找到，输出错误信息

# 运行回测
cerebro.run()
# 绘制回测图形
cerebro.plot()
