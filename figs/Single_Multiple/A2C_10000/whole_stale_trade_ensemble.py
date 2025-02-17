import pandas as pd
import matplotlib.pyplot as plt

# 用于拼接的起始和结束数字
start_num = 126
end_num = 1197
step = 63

# 用于保存拼接后的数据
all_data = []

# 读取并拼接每个 CSV 文件
for num in range(start_num, end_num + 1, step):
    file_name = f"account_value_trade_ensemble_{num}.csv"
    
    # 跳过第一行并读取数据，header=None 保证不会将第一行作为列名
    data = pd.read_csv(file_name, header=None, skiprows=1)
    
    # 假设第一列是天数，第二列是资金总数
    days = data.iloc[:, 0]  # 获取第一列（天数）
    account_value = data.iloc[:, 1]  # 获取第二列（资金总数）
    
    # 修改天数列，使其连续
    days = days + (num - start_num)  # 使天数列连续
    
    # 拼接修改后的天数与资金数据
    combined_data = pd.DataFrame({ 'day': days, 'account_value': account_value })
    all_data.append(combined_data)

# 将所有数据拼接成一个完整的 DataFrame
full_data = pd.concat(all_data, ignore_index=True)

# 可视化资金变化情况
plt.figure(figsize=(10, 6))
plt.plot(full_data['day'], full_data['account_value'], label='Account Value', color='b')
plt.xlabel('Days')
plt.ylabel('Account Value')
plt.title('Account Value Over Time')
plt.grid(True)
plt.legend()
plt.show()

# 保存合并后的数据为一个新的 CSV 文件
full_data.to_csv("combined_account_value.csv", index=False)
