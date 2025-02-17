import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_files():
    # 获取当前目录下的所有csv文件
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]

    # 创建一个图像
    plt.figure(figsize=(10, 6))

    # 遍历所有csv文件，绘制曲线
    for csv_file in csv_files:
        # 去掉.csv扩展名
        file_name = os.path.splitext(csv_file)[0]
        
        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 假设CSV文件有"day"和"account_value"列
        x = df['day']
        y = df['account_value']

        # 绘制曲线，使用不同的颜色，自动分配颜色
        plt.plot(x, y, label=file_name)  # 使用去掉扩展名的文件名作为图例

    # 设置图像的标签和标题
    plt.xlabel('Day')
    plt.ylabel('Account Value')
    plt.title('Account Value vs. Day')

    # 显示图例
    plt.legend()

    # 显示图像
    plt.show()

# 调用函数绘制所有csv文件的曲线
plot_csv_files()
