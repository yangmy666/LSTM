import numpy as np
import pandas as pd
import pandas_ta as ta

# 生成特征列的函数，根据num_prev_days动态生成Prev_*列
def generate_feature_columns(features, num_prev_days):
    all_feature_columns = []

    # 遍历所有特征列
    for feature in features:
        # 对于每个没有 'Prev_' 前缀的特征列，生成多个Prev_*列
        for i in range(1, num_prev_days + 1):
            all_feature_columns.append(f'Prev_{i}_{feature}')  # 生成 Prev_1_*，Prev_2_*，... 列
    return all_feature_columns

def getDf(dataPath, future_days, num_prev_days):
    # 读取数据
    df = pd.read_csv(dataPath)

    # 假设 'Date' 列是字符串类型，需要转换为日期类型
    df['Date'] = pd.to_datetime(df['Date'])
    df_min = df['Date'].min()
    df['DateTime'] = (df['Date'] - df_min) / np.timedelta64(1, 'D')
    df['Month'] = df['Date'].dt.month

    ### **趋势类 (MA Indicators)**
    df["SMA_14"] = ta.sma(df["Close"], length=14)
    df["SMA_125"] = ta.sma(df["Close"], length=125)
    df["SMA_186"] = ta.sma(df["Close"], length=186)
    df['Bull_Bear'] = (df['Close'] > df['SMA_186']).astype(int)
    df["EMA_7"] = ta.ema(df["Close"], length=7)
    df["EMA_14"] = ta.ema(df["Close"], length=14)
    df['EMA_28'] = ta.ema(df['Close'], length=28)
    df['EMA_56'] = ta.ema(df['Close'], length=56)
    df['EMA_112'] = ta.ema(df['Close'], length=112)
    df['EMA_224'] = ta.ema(df['Close'], length=224)
    df["WMA"] = ta.wma(df["Close"], length=14)
    df["HMA"] = ta.hma(df["Close"], length=14)
    df["RMA"] = ta.rma(df["Close"], length=14)
    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["ADX"] = adx_df["ADX_14"]
    df["DI_PLUS"] = adx_df["DMP_14"]
    df["DI_MINUS"] = adx_df["DMN_14"]
    df["KAMA"] = ta.kama(df["Close"], length=10)
    macd = df.ta.macd(close='Close', fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['SIGNAL'] = macd['MACDh_12_26_9']
    df['HIST'] = macd['MACDs_12_26_9']

    ### **动量类 (Momentum Indicators)**
    df["RSI"] = ta.rsi(df["Close"], length=14)
    weekly_df = df.groupby(pd.Grouper(key='Date', freq='W')).last()
    weekly_df['WEEK_RSI'] = ta.rsi(weekly_df['Close'], length=14)
    df['WEEK_RSI'] = df['Date'].apply(
        lambda x: weekly_df.loc[weekly_df.index <= x, 'WEEK_RSI'].iloc[-1] if not weekly_df.loc[
            weekly_df.index <= x, 'WEEK_RSI'].empty else None
    )
    df['WEEK_RSI'] = df['WEEK_RSI'].ffill()
    monthly_df = df.groupby(pd.Grouper(key='Date', freq='M')).last()
    monthly_df['MONTH_RSI'] = ta.rsi(monthly_df['Close'], length=14)
    df['MONTH_RSI'] = df['Date'].apply(
        lambda x: monthly_df.loc[monthly_df.index <= x, 'MONTH_RSI'].iloc[-1] if not monthly_df.loc[
            monthly_df.index <= x, 'MONTH_RSI'].empty else None
    )
    df['MONTH_RSI'] = df['MONTH_RSI'].ffill()

    df["WILLR"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)
    df["CCI"] = ta.cci(df["High"], df["Low"], df["Close"], length=14)
    df["MOM"] = ta.mom(df["Close"], length=10)
    df["ROC"] = ta.roc(df["Close"], length=10)

    ### **均值回归类 (Mean Reversion Indicators)**
    bbands_df = ta.bbands(df["Close"])
    df["BB_LOWER"], df["BB_MIDDLE"], df["BB_UPPER"], df["BB_WIDTH"], df["BB_PERCENT"] = \
        bbands_df["BBL_5_2.0"], bbands_df["BBM_5_2.0"], bbands_df["BBU_5_2.0"], bbands_df["BBB_5_2.0"], bbands_df[
            "BBP_5_2.0"]
    kc_df = ta.kc(df["High"], df["Low"], df["Close"])
    df["KC_LOWER"], df["KC_MIDDLE"], df["KC_UPPER"] = kc_df["KCLe_20_2"], kc_df["KCBe_20_2"], kc_df["KCUe_20_2"]
    donchian_df = ta.donchian(df["High"], df["Low"], length=20)
    df["DC_LOWER"], df["DC_MIDDLE"], df["DC_UPPER"] = \
        donchian_df["DCL_20_20"], donchian_df["DCM_20_20"], donchian_df["DCU_20_20"]

    ### **波动性类 (Volatility Indicators)**
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df['ATR_RATIO'] = df['ATR'] / df['Close']
    df["HVOL"] = ta.pvol(df["Close"], df["Volume"], length=20)

    ### **成交量类 (Volume Indicators)**
    df["VWMA"] = ta.vwma(df["Close"], df["Volume"], length=14)
    df["OBV"] = ta.obv(df["Close"], df["Volume"])
    df["CMF"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=20)
    df["AD"] = ta.ad(df["High"], df["Low"], df["Close"], df["Volume"])
    df["VOL_SMA_14"] = ta.sma(df["Volume"], length=14)
    df["VOL_SMA_125"] = ta.sma(df["Volume"], length=125)
    df["VOL_EMA_7"] = ta.ema(df["Volume"], length=7)
    df["VOL_EMA_14"] = ta.ema(df["Volume"], length=14)
    df['VOL_EMA_28'] = ta.ema(df['Volume'], length=28)
    df['VOL_EMA_56'] = ta.ema(df['Volume'], length=56)
    df['VOL_EMA_112'] = ta.ema(df['Volume'], length=112)
    df['VOL_EMA_224'] = ta.ema(df['Volume'], length=224)

    ### **统计类 (Statistical Indicators)**
    df["SKEW"] = ta.skew(df["Close"], length=10)
    df["KURT"] = df["Close"].kurtosis()
    df["ZSCORE"] = ta.zscore(df["Close"], length=20)

    # Prev_需要排除的列名（比如日期列等）
    exclude_columns = ['Date', 'DateTime']  # 根据实际情况修改

    # 创建一个空字典，用于存储所有Prev_列
    prev_columns = {}

    # 遍历 num_prev_days 来生成 Prev_* 列
    for i in range(1, num_prev_days + 1):
        for column in df.columns:
            if column not in exclude_columns and not column.startswith('Prev_'):
                prev_columns[f'Prev_{i}_{column}'] = df[column].shift(future_days + (i - 1))

    # 使用 pd.concat 一次性将所有 Prev_ 列添加到原 DataFrame
    df = pd.concat([df, pd.DataFrame(prev_columns)], axis=1)

    df = df.sort_values(by='Date')

    # 删除空值（由于第一行没有前一天的数据）
    df = df.dropna()

    # 获取最后一行的日期
    last_date = df['Date'].iloc[-1]

    for _ in range(future_days):
        # 判断是否是周五，若是，则加3天；否则加1天
        if last_date.weekday() == 4:  # 如果是周五
            next_date = last_date + pd.Timedelta(days=3)
        else:
            next_date = last_date + pd.Timedelta(days=1)

        # 创建新行的数据字典
        new_row = {
            'Date': next_date,  # 下一日期
            'Month': next_date.month,
            'DateTime': (next_date - df_min) / np.timedelta64(1, 'D'),
        }

        # 为新行生成 Prev_* 列，但只生成需要的列，并避免重复生成
        for i in range(1, num_prev_days + 1):
            for column in df.columns:
                if column not in exclude_columns and not column.startswith('Prev_'):
                    new_row[f'Prev_{i}_{column}'] = df[column].iloc[-(future_days + (i - 1))]

        # 创建新行数据框并添加到原数据框中
        new_row_df = pd.DataFrame([new_row])
        new_row_df = new_row_df.dropna(axis=1, how='all')  # 删除全为空的列
        df = pd.concat([df, new_row_df], ignore_index=True)

        # 更新 last_date 为当前添加行的日期
        last_date = next_date

    # 设置显示选项
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', None)  # 防止换行
    pd.set_option('display.max_rows', None)  # 如果你想显示所有行

    # 打印最后 10 行
    print(df.tail(future_days+1))

    return df





