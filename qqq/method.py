import numpy as np
import pandas as pd
import pandas_ta as ta


def getDf(dataPath):
    # 读取数据
    df = pd.read_csv(dataPath)
    # 假设 'Date' 列是字符串类型，需要转换为日期类型
    df['Date'] = pd.to_datetime(df['Date'])
    df['DateTime'] = (df['Date'] - df['Date'].min()) / np.timedelta64(1, 'D')
    df['Month'] = df['Date'].dt.month
    # df['SMA_125'] = ta.sma(df['Close'], length=125)
    # df['SMA_186'] = ta.sma(df['Close'], length=186)
    # df['Rise']=(df['Close']-df['Close'].shift(1))/df['Close'].shift(1)
    # df['Vol_diff']=df['Volume']-df['Volume'].shift(1)
    # df['Entity_diff']=(df['Close']-df['Open'])/df['Open']
    # df['High_diff']=df['High']-df['Low']
    ### **趋势类 (MA Indicators)**
    df["SMA"] = ta.sma(df["Close"], length=14)  # 简单移动平均线
    df["EMA"] = ta.ema(df["Close"], length=14)  # 指数移动平均线
    df["WMA"] = ta.wma(df["Close"], length=14)  # 加权移动平均线
    df["HMA"] = ta.hma(df["Close"], length=14)  # Hull 移动平均线
    df["RMA"] = ta.rma(df["Close"], length=14)  # 指数平滑移动平均线
    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)  # 平均方向指数
    df["ADX"] = adx_df["ADX_14"]  # 平均方向指数
    df["DI_PLUS"] = adx_df["DMP_14"]  # 正方向指标 (+DI)
    df["DI_MINUS"] = adx_df["DMN_14"]  # 负方向指标 (-DI)
    df["KAMA"] = ta.kama(df["Close"], length=10)  # 自适应均线
    # df["MACD"], df["SIGNAL"], df["HIST"] = ta.macd(df["Close"])  # MACD 指标
    ### **动量类 (Momentum Indicators)**
    df["RSI"] = ta.rsi(df["Close"], length=14)  # 相对强弱指数
    # df["STOCH_K"], df["STOCH_D"] = ta.stoch(df["High"], df["Low"], df["Close"])  # 随机指标
    df["WILLR"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)  # 威廉指标
    df["CCI"] = ta.cci(df["High"], df["Low"], df["Close"], length=14)  # 商品通道指数
    df["MOM"] = ta.mom(df["Close"], length=10)  # 动量指标
    df["ROC"] = ta.roc(df["Close"], length=10)  # 价格变化率
    ### **均值回归类 (Mean Reversion Indicators)**
    bbands_df = ta.bbands(df["Close"])  # 布林带
    df["BB_LOWER"], df["BB_MIDDLE"], df["BB_UPPER"], df["BB_WIDTH"], df["BB_PERCENT"] = \
        bbands_df["BBL_5_2.0"], bbands_df["BBM_5_2.0"], bbands_df["BBU_5_2.0"], bbands_df["BBB_5_2.0"], bbands_df[
            "BBP_5_2.0"]
    kc_df = ta.kc(df["High"], df["Low"], df["Close"])  # 凯尔特通道
    df["KC_LOWER"], df["KC_MIDDLE"], df["KC_UPPER"] = kc_df["KCLe_20_2"], kc_df["KCBe_20_2"], kc_df["KCUe_20_2"]
    donchian_df = ta.donchian(df["High"], df["Low"], length=20)  # 唐奇安通道
    df["DC_LOWER"], df["DC_MIDDLE"], df["DC_UPPER"] = \
        donchian_df["DCL_20_20"], donchian_df["DCM_20_20"], donchian_df["DCU_20_20"]
    ### **波动性类 (Volatility Indicators)**
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)  # 平均真实波幅
    df['ATR_RATIO'] = df['ATR'] / df['Close']
    df["HVOL"] = ta.pvol(df["Close"], df["Volume"], length=20)  # 历史波动率
    ### **成交量类 (Volume Indicators)**
    df["VWMA"] = ta.vwma(df["Close"], df["Volume"], length=14)  # 成交量加权移动平均线
    df["OBV"] = ta.obv(df["Close"], df["Volume"])  # 平衡成交量
    df["CMF"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=20)  # 钱德动量摆动
    df["AD"] = ta.ad(df["High"], df["Low"], df["Close"], df["Volume"])  # 累积/分布线
    ### **统计类 (Statistical Indicators)**
    df["SKEW"] = ta.skew(df["Close"], length=10)  # 偏度
    df["KURT"] = df["Close"].kurtosis()  # 峰度
    df["ZSCORE"] = ta.zscore(df["Close"], length=20)  # Z 分数

    # 计算所有Prev_*列
    prev_columns = {
        'Prev_Month': df['Month'].shift(1),
        'Prev_Open': df['Open'].shift(1),
        'Prev_Close': df['Close'].shift(1),
        'Prev_High': df['High'].shift(1),
        'Prev_Low': df['Low'].shift(1),
        'Prev_Volume': df['Volume'].shift(1),
        # 'Prev_Rise': df['Rise'].shift(1),
        # 'Prev_SMA_125': df['SMA_125'].shift(1),
        # 'Prev_SMA_186': df['SMA_186'].shift(1),
        # 'Prev_Vol_diff': df['Vol_diff'].shift(1),
        # 'Prev_Entity_diff': df['Entity_diff'].shift(1),
        # 'Prev_High_diff': df['High_diff'].shift(1),
        'Prev_SMA': df['SMA'].shift(1),
        'Prev_EMA': df['EMA'].shift(1),
        'Prev_WMA': df['WMA'].shift(1),
        'Prev_HMA': df['HMA'].shift(1),
        'Prev_RMA': df['RMA'].shift(1),
        'Prev_ADX': df['ADX'].shift(1),
        'Prev_DI_PLUS': df['DI_PLUS'].shift(1),
        'Prev_DI_MINUS': df['DI_MINUS'].shift(1),
        'Prev_KAMA': df['KAMA'].shift(1),
        'Prev_RSI': df['RSI'].shift(1),
        'Prev_WILLR': df['WILLR'].shift(1),
        'Prev_CCI': df['CCI'].shift(1),
        'Prev_MOM': df['MOM'].shift(1),
        'Prev_ROC': df['ROC'].shift(1),
        'Prev_BB_LOWER': df['BB_LOWER'].shift(1),
        'Prev_BB_MIDDLE': df['BB_MIDDLE'].shift(1),
        'Prev_BB_UPPER': df['BB_UPPER'].shift(1),
        'Prev_BB_WIDTH': df['BB_WIDTH'].shift(1),
        'Prev_BB_PERCENT': df['BB_PERCENT'].shift(1),
        'Prev_KC_LOWER': df['KC_LOWER'].shift(1),
        'Prev_KC_MIDDLE': df['KC_MIDDLE'].shift(1),
        'Prev_KC_UPPER': df['KC_UPPER'].shift(1),
        'Prev_DC_LOWER': df['DC_LOWER'].shift(1),
        'Prev_DC_MIDDLE': df['DC_MIDDLE'].shift(1),
        'Prev_DC_UPPER': df['DC_UPPER'].shift(1),
        'Prev_ATR': df['ATR'].shift(1),
        'Prev_ATR_RATIO': df['ATR_RATIO'].shift(1),
        'Prev_HVOL': df['HVOL'].shift(1),
        'Prev_VWMA': df['VWMA'].shift(1),
        'Prev_OBV': df['OBV'].shift(1),
        'Prev_CMF': df['CMF'].shift(1),
        'Prev_AD': df['AD'].shift(1),
        'Prev_SKEW': df['SKEW'].shift(1),
        'Prev_KURT': df['KURT'].shift(1),
        'Prev_ZSCORE': df['ZSCORE'].shift(1),
    }

    # 使用concat一次性合并所有新的Prev_*列
    prev_df = pd.DataFrame(prev_columns)

    # 将新的列合并到原始df
    df = pd.concat([df, prev_df], axis=1)

    # 按日期排序
    df = df.sort_values(by='Date')

    # 删除空值（由于第一行没有前一天的数据）
    df = df.dropna()

    # 获取最后一行的日期
    last_date = df['Date'].iloc[-1]

    # 判断是否是周五，若是，则加3天；否则加1天
    if last_date.weekday() == 4:  # 如果是周五
        next_date = last_date + pd.Timedelta(days=3)
    else:
        next_date = last_date + pd.Timedelta(days=1)

    # 创建新行的数据字典
    new_row = {
        'Date': next_date,  # 下一日期
        'Month': None,
        'Open': None,
        'Close': None,
        'High': None,
        'Low': None,
        'Volume': None,
        # 'Rise': None,
        # 'SMA_125': None,
        # 'SMA_186': None,
        # 'Vol_diff': None,
        # 'Entity_diff': None,
        # 'High_diff': None,
        'SMA': None,
        'EMA': None,
        'WMA': None,
        'HMA': None,
        'RMA': None,
        'ADX': None,
        'DI_PLUS': None,
        'DI_MINUS': None,
        'KAMA': None,
        'RSI': None,
        'WILLR': None,
        'CCI': None,
        'MOM': None,
        'ROC': None,
        'BB_LOWER': None,
        'BB_MIDDLE': None,
        'BB_UPPER': None,
        'BB_WIDTH': None,
        'BB_PERCENT': None,
        'KC_LOWER': None,
        'KC_MIDDLE': None,
        'KC_UPPER': None,
        'DC_LOWER': None,
        'DC_MIDDLE': None,
        'DC_UPPER': None,
        'ATR': None,
        'ATR_RATIO': None,
        'HVOL': None,
        'VWMA': None,
        'OBV': None,
        'CMF': None,
        'AD': None,
        'SKEW': None,
        'KURT': None,
        'ZSCORE': None,

        'Prev_Month': df['Month'].iloc[-1],  # 上一行的 Month
        'Prev_Open': df['Open'].iloc[-1],  # 上一行的 Open
        'Prev_Close': df['Close'].iloc[-1],  # 上一行的 Close
        'Prev_High': df['High'].iloc[-1],  # 上一行的 High
        'Prev_Low': df['Low'].iloc[-1],  # 上一行的 Low
        'Prev_Volume': df['Volume'].iloc[-1],  # 上一行的 Volume
        # 'Prev_Rise': df['Rise'].iloc[-1],  # 上一行的 Rise
        # 'Prev_SMA_125': df['SMA_125'].iloc[-1],  # 上一行的 SMA_125
        # 'Prev_SMA_186': df['SMA_186'].iloc[-1],  # 上一行的 SMA_186
        # 'Prev_Vol_diff': df['Vol_diff'].iloc[-1],
        # 'Prev_Entity_diff': df['Entity_diff'].iloc[-1],
        # 'Prev_High_diff': df['High_diff'].iloc[-1],
        'Prev_SMA': df['SMA'].iloc[-1],  # 上一行的 SMA
        'Prev_EMA': df['EMA'].iloc[-1],  # 上一行的 EMA
        'Prev_WMA': df['WMA'].iloc[-1],  # 上一行的 WMA
        'Prev_HMA': df['HMA'].iloc[-1],  # 上一行的 HMA
        'Prev_RMA': df['RMA'].iloc[-1],  # 上一行的 RMA
        'Prev_ADX': df['ADX'].iloc[-1],  # 上一行的 ADX
        'Prev_DI_PLUS': df['DI_PLUS'].iloc[-1],  # 上一行的 DI_PLUS
        'Prev_DI_MINUS': df['DI_MINUS'].iloc[-1],  # 上一行的 DI_MINUS
        'Prev_KAMA': df['KAMA'].iloc[-1],  # 上一行的 KAMA
        'Prev_RSI': df['RSI'].iloc[-1],
        'Prev_WILLR': df['WILLR'].iloc[-1],  # 上一行的 WILLR
        'Prev_CCI': df['CCI'].iloc[-1],  # 上一行的 CCI
        'Prev_MOM': df['MOM'].iloc[-1],  # 上一行的 MOM
        'Prev_ROC': df['ROC'].iloc[-1],  # 上一行的 ROC
        'Prev_BB_LOWER': df['BB_LOWER'].iloc[-1],  # 上一行的 BB_LOWER
        'Prev_BB_MIDDLE': df['BB_MIDDLE'].iloc[-1],  # 上一行的 BB_MIDDLE
        'Prev_BB_UPPER': df['BB_UPPER'].iloc[-1],  # 上一行的 BB_UPPER
        'Prev_BB_WIDTH': df['BB_WIDTH'].iloc[-1],  # 上一行的 BB_WIDTH
        'Prev_BB_PERCENT': df['BB_PERCENT'].iloc[-1],  # 上一行的 BB_PERCENT
        'Prev_KC_LOWER': df['KC_LOWER'].iloc[-1],  # 上一行的 KC_LOWER
        'Prev_KC_MIDDLE': df['KC_MIDDLE'].iloc[-1],  # 上一行的 KC_MIDDLE
        'Prev_KC_UPPER': df['KC_UPPER'].iloc[-1],  # 上一行的 KC_UPPER
        'Prev_DC_LOWER': df['DC_LOWER'].iloc[-1],  # 上一行的 DC_LOWER
        'Prev_DC_MIDDLE': df['DC_MIDDLE'].iloc[-1],  # 上一行的 DC_MIDDLE
        'Prev_DC_UPPER': df['DC_UPPER'].iloc[-1],  # 上一行的 DC_UPPER
        'Prev_ATR': df['ATR'].iloc[-1],  # 上一行的 ATR
        'Prev_ATR_RATIO': df['ATR_RATIO'].iloc[-1],  # 上一行的 ATR_RATIO
        'Prev_HVOL': df['HVOL'].iloc[-1],  # 上一行的 HVOL
        'Prev_VWMA': df['VWMA'].iloc[-1],  # 上一行的 VWMA
        'Prev_OBV': df['OBV'].iloc[-1],  # 上一行的 OBV
        'Prev_CMF': df['CMF'].iloc[-1],  # 上一行的 CMF
        'Prev_AD': df['AD'].iloc[-1],  # 上一行的 AD
        'Prev_SKEW': df['SKEW'].iloc[-1],  # 上一行的 SKEW
        'Prev_KURT': df['KURT'].iloc[-1],  # 上一行的 KURT
        'Prev_ZSCORE': df['ZSCORE'].iloc[-1],  # 上一行的 ZSCORE
    }
    # 使用pd.concat将新行添加到DataFrame中
    new_row_df = pd.DataFrame([new_row])
    # 删除new_row_df中的所有空或全为NA的列
    new_row_df = new_row_df.dropna(axis=1, how='all')
    df = pd.concat([df, new_row_df], ignore_index=True)

    # 设置显示选项
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', None)  # 防止换行
    pd.set_option('display.max_rows', None)  # 如果你想显示所有行
    # 打印数据框
    print(df.tail(10))
    return df