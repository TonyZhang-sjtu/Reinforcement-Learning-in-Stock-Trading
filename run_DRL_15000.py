# common library
import pandas as pd
import numpy as np
import time
from stable_baselines.common.vec_env import DummyVecEnv

# preprocessor
from preprocessing.preprocessors import *
# config
from config.config import *
# model
from model.models_15000 import *


def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    data = preprocess_data() # 数据预处理
    '''
    data 将包含以下字段：
        datadate：日期（通常是整数格式，比如YYYYMMDD）。
        tic：股票代码（Ticker Symbol）。
        adjcp：调整后的收盘价。
        open：调整后的开盘价。
        high：调整后的最高价。
        low：调整后的最低价。
        volume：成交量。
        macd：MACD指标。
        rsi：RSI指标。
        cci：CCI指标。
        adx：ADX指标。
    '''
    data = add_turbulence(data) # 添加震荡指数

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    # 选择验证和交易的时间段
    # 从数据中提取出2015年10月1日至2020年7月7日的日期范围作为交易日期（可能用于回测和训练模型时的时间窗口）
    unique_trade_date = data[(data.datadate > 20151001)&(data.datadate <= 20200707)].datadate.unique()
    
    # rebalance_window is the number of months to retrain the model
    # validation_window is the numebr of months to validation the model and select for trading
    rebalance_window = 63 # 63个交易日，约3个月
    validation_window = 63 # 63个交易日，约3个月
    
    ## Ensemble Strategy
    # 执行一个集成策略，该策略包括了多个模型，每个模型都是一个DRL模型
    run_ensemble_strategy(df=data,  # 数据
                          unique_trade_date= unique_trade_date, # 交易日期
                          rebalance_window = rebalance_window, # 重新平衡窗口
                          validation_window=validation_window)  # 验证窗口

    #_logger.info(f"saving model version: {_version}")

if __name__ == "__main__":
    run_model()
