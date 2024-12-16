import matplotlib.pyplot as plt
import mplfinance as mpf

from qqq.v2.data_treat import getDf

#预测未来第几天
future_days=30
df = getDf('C:\py_project\LSTM\stock_data\\QQQ.csv',future_days)

# 计算总行数的 80% 位置，即 20% 数据的起始行
split_index = int(len(df) * 0.3)
# 获取最新的 20% 数据
latest_20_percent = df.iloc[split_index:]

# # 将 'Date' 列设置为索引
# latest_20_percent.set_index('Date', inplace=True)
# # 计算移动平均线
# latest_20_percent['MA5'] = latest_20_percent['Close'].rolling(window=5).mean()
# latest_20_percent['MA20'] = latest_20_percent['Close'].rolling(window=20).mean()
# # 设置移动平均线的图形样式
# ma5 = mpf.make_addplot(latest_20_percent['MA5'], color='blue', width=1)
# ma20 = mpf.make_addplot(latest_20_percent['MA20'], color='red', width=1)
# # 使用mplfinance绘制K线图
# mpf.plot(latest_20_percent, type='candle', style='charles', title='K-line Chart', ylabel='Price', volume=True,addplot=[ma5, ma20])

# 创建一个包含两部分的绘图：主图和附图
fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# 主图：收盘价
ax1.plot(latest_20_percent['Date'], latest_20_percent['Close'], label='Close Price', color='blue')
ax1.set_title('Stock Closing Price')
ax1.set_ylabel('Close Price')
ax1.legend(loc='upper left')

# 附图：RSI
ax2.plot(latest_20_percent['Date'], latest_20_percent['RSI'], label='RSI', color='orange')
ax2.axhline(70, color='red', linestyle='--')  # 超买线
ax2.axhline(30, color='green', linestyle='--')  # 超卖线
ax2.set_title('RSI Indicator')
ax2.set_ylabel('RSI')
ax2.legend(loc='upper left')

# 绘制 ATR (附图)
ax3.plot(latest_20_percent['Date'], latest_20_percent['MONTH_RSI'], label='ATR_RATIO', color='red')
ax3.set_title('ATR_RATIO')
ax3.set_ylabel('ATR_RATIO Value')
ax3.legend(loc='upper left')

# 自动调整布局以避免标签重叠
plt.tight_layout()

# 显示图表
plt.show()