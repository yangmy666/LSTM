import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

start_date_param='1999-01-22'
filename='IWM'

# 指定文件路径
file_path = r'C:\py_project\LSTM\stock_data\\'+filename+'.csv'

# 读取现有的 CSV 文件
try:
    existing_data = pd.read_csv(file_path, parse_dates=['Date'])
    print("读取现有数据成功！")
except FileNotFoundError:
    print("文件不存在，将从头下载数据。")
    existing_data = pd.DataFrame()

# 获取文件中最新的数据日期
if not existing_data.empty:
    latest_date = existing_data['Date'].max()
    print(f"文件中最新日期为: {latest_date}")

    # 从最新日期的下一天开始下载数据
    start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
else:
    # 如果文件不存在或为空，从头下载
    start_date = start_date_param

end_date = datetime.now().strftime('%Y-%m-%d')

# 下载新的数据
print(f"开始从 {start_date} 到 {end_date} 下载数据...")
data = yf.download(filename, start=start_date, end=end_date)

# 打印下载的数据列名，检查列名
print("下载的数据列名：", data.columns)

if not data.empty:
    # 检查是否是 MultiIndex，展平列名
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]  # 去除第二层索引

    # 重置索引并处理日期列
    data.reset_index(inplace=True)

    # 打印重置后的数据列名
    print("重置索引后的数据列名：", data.columns)

    # 只保留需要的列
    if all(col in data.columns for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']):
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    else:
        print("警告: 数据中缺少必要的列。")

    # 如果已有数据，合并新旧数据
    if not existing_data.empty:
        # 确保列名一致
        existing_data = existing_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        combined_data = pd.concat([existing_data, data], ignore_index=True)
        combined_data.drop_duplicates(subset=['Date'], inplace=True)
    else:
        combined_data = data

    # 保存合并后的数据到文件
    combined_data.to_csv(file_path, index=False)
    print(f"数据已更新并保存到: {file_path}")
else:
    print("没有新的数据需要更新！")

print("完成！")
