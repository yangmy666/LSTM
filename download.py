import yfinance as yf

# 下载QQQ的历史数据，从2010年1月1日到2020年1月1日
data = yf.download('QQQ', start='1999-03-10', end='2024-12-10')

# 查看前几行数据，确保下载成功
print(data.head())

# 保存数据到指定的路径（如 D:/Data/QQQ_data.csv）
file_path = 'C:\py_project\LSTM\stock_data\QQQ.csv'
data.to_csv(file_path)

print("ok")

