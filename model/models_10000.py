# common library
import pandas as pd
import numpy as np
import time
import gym

# RL models from stable-baselines
from stable_baselines import SAC
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import TD3
from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from preprocessing.preprocessors import *
from config import config

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade


def train_A2C(env_train, model_name, timesteps=50000):
    """A2C model"""

    start = time.time() # 记录开始时间
    model = A2C('MlpPolicy', env_train, verbose=0) # 创建一个 A2C 模型
    model.learn(total_timesteps=timesteps) # 在训练环境上训练模型
    end = time.time() # 记录结束时间

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}") # 保存模型
    print('Training time (A2C): ', (end - start) / 60, ' minutes') # 打印训练时间
    return model # 返回模型


def train_DDPG(env_train, model_name, timesteps=50000):
    """DDPG model"""

    start = time.time()  # 记录开始时间
    model = DDPG('MlpPolicy', env_train) # 创建一个 DDPG 模型
    model.learn(total_timesteps=timesteps) # 在训练环境上训练模型
    end = time.time() # 记录结束时间

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', (end - start) / 60, ' minutes')
    return model


def train_PPO(env_train, model_name, timesteps=50000):
    """PPO model"""
    start = time.time() # 记录开始时间
    model = PPO2('MlpPolicy', env_train) # 创建一个 PPO 模型
    model.learn(total_timesteps=timesteps) # 在训练环境上训练模型
    end = time.time() # 记录结束时间

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model


def train_SAC(env_train, model_name, timesteps=50000):
    """SAC model"""
    start = time.time() # 记录开始时间
    model = SAC('MlpPolicy', env_train, verbose=0) # 创建一个 SAC 模型
    model.learn(total_timesteps=timesteps) # 在训练环境上训练模型
    end = time.time() # 记录结束时间

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}") # 保存模型
    print('Training time (SAC): ', (end - start) / 60, ' minutes') # 打印训练时间
    return model # 返回模型


def train_TD3(env_train, model_name, timesteps=50000):
    """TD3 model"""
    start = time.time() # 记录开始时间
    model = TD3('MlpPolicy', env_train, verbose=0) # 创建一个 SAC 模型
    model.learn(total_timesteps=timesteps) # 在训练环境上训练模型
    end = time.time() # 记录结束时间

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}") # 保存模型
    print('Training time (TD3): ', (end - start) / 60, ' minutes') # 打印训练时间
    return model # 返回模型


#  对未来的交易数据进行预测，并在每个时间步根据模型的动作进行交易决策
def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial):
    '''
    df：包含历史市场数据的 DataFrame。
    model：训练好的强化学习模型（PPO, A2C, DDPG等），用于进行预测。
    name：模型的名称，通常用于日志记录或文件保存。
    last_state：上一个时间周期的模型状态，强化学习的状态通常是由环境提供的，可以是市场的某种状态表示。
    iter_num：当前迭代的周期编号，用于从 unique_trade_date 中提取相应的时间范围。
    unique_trade_date：唯一的交易日期列表，用于确定每个交易周期的开始和结束日期。
    rebalance_window：重平衡窗口，用于确定每次交易周期的长度。
    turbulence_threshold：市场波动性阈值，用于调整策略的风险管理。
    initial：指示是否是初始状态，用于模型训练中的初始条件设置。
    '''
    ### make a prediction based on trained model### 

    ## trading env
    # 根据 unique_trade_date 和 rebalance_window 参数，从数据 df 中分割出当前周期的交易数据（例如，交易日期范围从 unique_trade_date[iter_num - rebalance_window] 到 unique_trade_date[iter_num]
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    # 创建一个用于交易的环境（StockEnvTrade），这个环境会基于当前的交易数据进行模拟，
    # 并使用当前的波动性阈值（turbulence_threshold）、初始状态（initial）、上一个状态（last_state）以及当前迭代的编号（iter_num）进行初始化
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset() # 重置交易环境
    # 执行交易步骤,对于每一个交易日（trade_data 的每个唯一日期），执行一次交易步骤
    for i in range(len(trade_data.index.unique())):
        # 使用模型对当前的观察值 obs_trade 进行预测，输出 action，即模型的交易决策（如买、卖或持有），同时 _states 是模型的内部状态（可以在一些模型中使用）
        action, _states = model.predict(obs_trade) 
        # 在环境中执行该动作，返回下一个观察值 obs_trade、奖励 rewards、是否结束 dones 和其他信息 info
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2): # 如果是倒数第二个交易日
            # print(env_test.render())
            last_state = env_trade.render() # 获取当前环境的状态
    # 保存最后状态到文件
    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results_10000/last_state_{}_{}.csv'.format(name, i), index=False)
    return last_state # 返回最后的状态


def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    # # 读取与当前迭代相关的验证结果文件
    df_total_value = pd.read_csv('results_10000/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train'] # 重命名列名
    #  计算的是每一行与前一行的百分比变化，即每日的收益率。通过 df_total_value['daily_return'] 新增一列来存储这些日收益率
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    # 计算夏普比率
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe # 年化的夏普比率，用来衡量验证期间策略的风险调整后收益

# 根据历史数据训练三个强化学习模型（PPO、A2C、DDPG），
# 然后根据它们的 Sharpe Ratio（夏普比率） 选择最佳的模型进行交易决策
def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = [] # 记录上一个模型的最后状态

    ppo_sharpe_list = [] # 记录PPO模型的夏普比率
    ddpg_sharpe_list = [] # 记录DDPG模型的夏普比率
    a2c_sharpe_list = [] # 记录A2C模型的夏普比率
    sac_sharpe_list = [] # 记录SAC模型的夏普比率
    td3_sharpe_list = [] # 记录TD3模型的夏普比率

    model_use = [] # 记录最终使用的模型

    # based on the analysis of the in-sample data
    # turbulence_threshold = 140
    # 波动性（Turbulence Index）调整
    # 用来获取历史数据（2009年到2015年）中的波动性数据。
    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate']) # 去重
    # 是历史数据中的 90% 分位数，表示市场的波动性阈值
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time() # 记录开始时间
    #  从2015年10月1日开始，每隔63个交易日（约3个月）进行一次模型训练和验证
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        # 用于标识是否为初始状态（第一次训练时为 True）
        if i - rebalance_window - validation_window == 0: # 如果是第一次
            # inital state
            initial = True
        else: # 如果不是第一次
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        # 用来获取过去126个交易日的波动性数据
        historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window-63]))]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate']) 
        #  计算该历史窗口内的波动性均值
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)   
        # 根据均值与历史数据的比较，来决定是否降低市场风险
        # 如果当前市场波动性较高，设置波动性阈值为 90% 分位数；否则，将阈值调高到 99% 分位数，降低风险
        if historical_turbulence_mean > insample_turbulence_threshold: # 如果历史数据的均值大于90%分位数
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile, 
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold 
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold # 设置波动性阈值
        else:  # 如果历史数据的均值小于90%分位数
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk 
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 0.99)
        print("turbulence_threshold: ", turbulence_threshold) # 打印波动性阈值

        ############## Environment Setup starts ##############
        ## training env
        # 从2009年开始，到当前时间之前的数据作为训练数据
        train = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)]) # 创建一个训练环境，用于模型训练

        ## validation env 创建了一个用于验证的环境
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window]) # 从数据中提取验证集
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)]) # 创建一个验证环境
        obs_val = env_val.reset() # 重置验证环境
        ############## Environment Setup ends ##############
        
        # 每次循环中，会分别训练 A2C、PPO 和 DDPG 三个模型，并在验证集上计算它们的 Sharpe Ratio（夏普比率）
        # 额外加入SAC和TD3两个模型
        ############## Training and Validation starts ##############
        print("======Model training from: ", 20090000, "to ",
              unique_trade_date[i - rebalance_window - validation_window]) # 打印训练时间范围
        # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))


        print("==============Model Training===========")
        print("======A2C Training========")
        # 训练 A2C 模型
        model_a2c = train_A2C(env_train, model_name="A2C_10k_dow_{}".format(i), timesteps=10000)
        print("======A2C Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        # 在验证集上验证 A2C 模型
        DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_a2c = get_validation_sharpe(i) # 计算 A2C 模型的夏普比率
        print("A2C Sharpe Ratio: ", sharpe_a2c) # 打印 A2C 模型的夏普比率


        # print("======PPO Training========") # 训练 PPO 模型
        # # 训练 PPO 模型
        # model_ppo = train_PPO(env_train, model_name="PPO_100k_dow_{}".format(i), timesteps=80000) # 训练 PPO 模型
        # print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
        #       unique_trade_date[i - rebalance_window]) # 在验证集上验证 PPO 模型
        # DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        # sharpe_ppo = get_validation_sharpe(i) # 计算 PPO 模型的夏普比率
        # print("PPO Sharpe Ratio: ", sharpe_ppo) # 打印 PPO 模型的夏普比率


        # print("======DDPG Training========")
        # model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=5000) # 训练 DDPG 模型
        # print("======DDPG Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
        #       unique_trade_date[i - rebalance_window])
        # DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
        # sharpe_ddpg = get_validation_sharpe(i) # 计算 DDPG 模型的夏普比率


        # print("======SAC Training========")
        # model_sac = train_SAC(env_train, model_name="SAC_10k_dow_{}".format(i), timesteps=5000) # 训练 DDPG 模型
        # print("======SAC Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
        #       unique_trade_date[i - rebalance_window])
        # DRL_validation(model=model_sac, test_data=validation, test_env=env_val, test_obs=obs_val)
        # sharpe_sac = get_validation_sharpe(i) # 计算 SAC 模型的夏普比率


        # print("======TD3 Training========")
        # model_td3 = train_TD3(env_train, model_name="TD3_10k_dow_{}".format(i), timesteps=5000) # 训练 DDPG 模型
        # print("======TD3 Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
        #       unique_trade_date[i - rebalance_window])
        # DRL_validation(model=model_td3, test_data=validation, test_env=env_val, test_obs=obs_val)
        # sharpe_td3 = get_validation_sharpe(i) # 计算 SAC 模型的夏普比率


        # ppo_sharpe_list.append(sharpe_ppo) # 记录 PPO 模型的夏普比率
        a2c_sharpe_list.append(sharpe_a2c) # 记录 A2C 模型的夏普比率
        # ddpg_sharpe_list.append(sharpe_ddpg) # 记录 DDPG 模型的夏普比率
        # sac_sharpe_list.append(sharpe_sac) # 记录 A2C 模型的夏普比率
        # td3_sharpe_list.append(sharpe_td3) # 记录 DDPG 模型的夏普比率

        model_ensemble = model_a2c
        model_use.append('A2C')

        # 根据五种模型的 Sharpe Ratio，选择表现最好的模型作为当前周期的交易模型
        # Model Selection based on sharpe ratio

        # if (sharpe_ppo >= sharpe_a2c) and (sharpe_ppo >= sharpe_ddpg) and (sharpe_ppo >= sharpe_sac): # 如果 PPO 的夏普比率最高 
        #     model_ensemble = model_ppo 
        #     model_use.append('PPO')
        # elif (sharpe_a2c > sharpe_ddpg) and (sharpe_ppo >= sharpe_sac):
        #     model_ensemble = model_a2c
        #     model_use.append('A2C')
        # elif (sharpe_ddpg > sharpe_sac):
        #     model_ensemble = model_ddpg
        #     model_use.append('DDPG')
        # else:
        #     model_ensemble = model_sac
        #     model_use.append('SAC')


        # if (sharpe_ppo >= sharpe_a2c) and (sharpe_ppo >= sharpe_ddpg) and (sharpe_ppo >= sharpe_sac) and (sharpe_ppo >= sharpe_td3): # 如果 PPO 的夏普比率最高 
        #     model_ensemble = model_ppo 
        #     model_use.append('PPO')
        # elif (sharpe_a2c > sharpe_ddpg) and (sharpe_ppo >= sharpe_sac) and (sharpe_ppo >= sharpe_td3):
        #     model_ensemble = model_a2c
        #     model_use.append('A2C')
        # elif (sharpe_ddpg > sharpe_sac) and (sharpe_ddpg > sharpe_td3):
        #     model_ensemble = model_ddpg
        #     model_use.append('DDPG')
        # else:
        #     model_ensemble = model_sac
        #     model_use.append('SAC')
        # elif (sharpe_sac > sharpe_td3):
        #     model_ensemble = model_sac
        #     model_use.append('SAC')
        # else:
        #     model_ensemble = model_td3
        #     model_use.append('TD3')
        ############## Training and Validation ends ##############    

        ############## Trading starts ##############    
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        print("Used Model: ", model_ensemble)
        # 在确定了最佳模型后，使用该模型进行实际交易决策
        last_state_ensemble = DRL_prediction(df=df, model=model_ensemble, name="ensemble",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        # print("============Trading Done============")
        ############## Trading ends ##############    

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
