import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from config import config

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    #_data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data

def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.datadate.factorize()[0]
    return data
# 根据原始的股票交易数据，计算出调整后的收盘价以及开盘价、最高价、最低价和成交量，并返回处理后的数据框
def calcualte_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    datadate: 日期（通常是整数格式，比如YYYYMMDD）。
    tic: 股票代码（Ticker Symbol）。
    prccd: 收盘价。
    ajexdi: 拆股因子（Adjustment Factor for Dividends and Splits）。
    prcod: 开盘价。
    prchd: 最高价。
    prcld: 最低价。
    cshtrd: 成交量。
    """
    data = df.copy() # 复制数据
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x) # 将拆股因子为0的值替换为1

    data['adjcp'] = data['prccd'] / data['ajexdi'] # 计算调整后的收盘价
    data['open'] = data['prcod'] / data['ajexdi'] # 计算调整后的开盘价
    data['high'] = data['prchd'] / data['ajexdi'] # 计算调整后的最高价
    data['low'] = data['prcld'] / data['ajexdi'] # 计算调整后的最低价
    data['volume'] = data['cshtrd'] # 成交量

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']] # 选择需要的列
    data = data.sort_values(['tic', 'datadate'], ignore_index=True) # 按股票代码和日期排序
    return data # 返回数据
# 计算和添加一些常见的 技术指标，如 MACD（指数平滑异同移动平均线）、RSI（相对强弱指数）、CCI（商品通道指数）和 ADX（平均趋向指数），并将其加入到股票数据中
def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())  # 将数据转换为 StockDataFrame 类型

    stock['close'] = stock['adjcp']  # 将调整后的收盘价 adjcp 设置为 StockDataFrame 对象的 close 列，因为许多技术指标是基于收盘价计算的
    unique_ticker = stock.tic.unique() #  获取唯一的股票代码
    # 创建了空的 DataFrame，用于存储计算得到的各项技术指标（MACD、RSI、CCI 和 ADX）的结果
    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    #temp = stock[stock.tic == unique_ticker[0]]['macd']
    # 逐只股票计算技术指标
    for i in range(len(unique_ticker)):
        ## macd 
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)

    # 将计算的技术指标添加到原始数据框中
    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df # 返回一个包含原始股票数据和新计算的技术指标（macd, rsi, cci, adx）的 pandas DataFrame 对象


#  数据预处理流水线
def preprocess_data():
    """data preprocessing pipeline"""

    df = load_dataset(file_name=config.TRAINING_DATA_FILE) # 读取数据
    # get data after 2009
    df = df[df.datadate>=20090000] # 选择2009年之后的数据
    # calcualte adjusted price
    df_preprocess = calcualte_price(df) # 计算调整后的价格
    # add technical indicators using stockstats
    df_final=add_technical_indicator(df_preprocess) # 添加技术指标
    # fill the missing values at the beginning
    df_final.fillna(method='bfill',inplace=True) # 用后向填充法（bfill）填补缺失值,将每个缺失值替换为它后面最近的非缺失值
    return df_final
# 添加震荡指数
# 震荡指数是一个衡量市场波动性和异常情况的指标，它基于某些市场资产（如道琼斯30指数）的历史价格波动。
def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df) # 计算震荡指数
    df = df.merge(turbulence_index, on='datadate') # 将震荡指数合并到数据中
    df = df.sort_values(['datadate','tic']).reset_index(drop=True) # 按日期和股票代码排序
    return df # 返回数据


#  计算震荡指数
def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets
    
    df_price_pivot=df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()
    # start after a year
    start = 252
    turbulence_index = [0]*start
    #turbulence_index = [0]
    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                #avoid large outlier because of the calculation just begins
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)
    
    
    turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index










