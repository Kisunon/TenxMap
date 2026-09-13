import numpy as np
import matplotlib.pyplot as plt

def draw_losses(data_path):
    data = np.load(data_path)
    if len(data.shape) > 1:
        for i in range(data.shape[1]):
            data_draw = data[:, i].T
            # 绘制折线图
            plt.plot(data_draw)

            # 添加标题和标签
            plt.title(f'plot_{data_draw[-1]}')
            plt.xlabel('step')
            plt.ylabel('value')

            # 显示图形
            plt.show()
    else:
        data_draw = data.T
        # 绘制折线图
        plt.plot(data_draw)

        # 添加标题和标签
        plt.title(f'plot_{data_draw[-1]}')
        plt.xlabel('step')
        plt.ylabel('value')

        # 显示图形
    plt.show()
