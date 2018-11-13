'''
BP神经网络拟合非线性曲线
激活函数：sigmoid
损失函数：quadratic_cost
'''
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
#  计算损失函数：平方损失函数
def quadratic_cost(y_, y):
    return np.sum(1 / 2 * np.square(y_ - y))
# 归一化数据
def normalize(data):
    data_min, data_max = data.min(), data.max()
    data = (data - data_min) / (data_max - data_min)
    return data
# 反归一化数据
def d_normalize(norm_data, data):
    data_min, data_max = data.min(), data.max()
    return norm_data * (data_max - data_min) + data_min
# 前向传播算法
def prediction(l0, W, B):
    Layers = [l0]
    for i in range(len(W)):
        Layers.append(sigmoid(x=Layers[-1].dot(W[i]) + B[i]))
    return Layers
# 反向传播算法：根据损失函数优化各个隐藏层的权重
def optimizer(Layers, W, B, y, learn_rate):
    # 计算最后一层误差
    l_error_arr = [(y - Layers[-1]) * d_sigmoid(x=Layers[-1])]
    # 计算每层神经元的误差
    for i in range(len(W) - 1, 0, -1):
        l_error_arr.append(l_error_arr[-1].dot(W[i].T) * d_sigmoid(x=Layers[i]))
    j = 0  # l_delta_arr = [err3, err2, err1]
    # 倒叙更新优化每层神经元的权重
    for i in range(len(W) - 1, -1, -1):
        W[i] += learn_rate * Layers[i].T.dot(l_error_arr[j])  # W3 += h2 * err3
        B[i] += learn_rate * l_error_arr[j]  # B3 += err3
        j += 1
# 训练BP神经网络
def train_BP(X, y, W, B, learn_rate=0.01, decay=0.5):
    norm_X, norm_y = normalize(data=X), normalize(data=y)  # 归一化处理
    end_loss = 0.068  # 结束训练的最小误差
    step_arr, loss_arr = [], []  # 记录单位时刻的误差
    step = 1
    while True:
        learn_rate_decay = learn_rate * 1.0 / (1.0 + decay * step)  # 计算衰减学习率
        Layers = prediction(l0=norm_X, W=W, B=B)  # 正向传播算法
        optimizer(Layers=Layers, W=W, B=B, y=y, learn_rate=learn_rate_decay)  # 反向传播
        cur_loss = quadratic_cost(y_=Layers[-1], y=norm_y)  # 计算当前误差
        if step % 1000 == 0:
            step_arr.append(step)
            loss_arr.append(cur_loss)
            print('经过{}次迭代，当前误差为{}'.format(step, cur_loss))
        if cur_loss < end_loss:
            prediction_ys = d_normalize(norm_data=Layers[-1], data=y)  # 反归一化结果
            print('经过{}次迭代，最终误差为：{}'.format(step, cur_loss))
            draw_fit_curve(origin_xs=X, origin_ys=y, prediction_ys=prediction_ys, step_arr=step_arr, loss_arr=loss_arr)
            break
        step += 1
# 可视化多项式曲线拟合结果
def draw_fit_curve(origin_xs, origin_ys, prediction_ys, step_arr, loss_arr):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(origin_xs, origin_ys, color='m', linestyle='', marker='.', label='原数据')
    ax1.plot(origin_xs, prediction_ys, color='#009688', label='拟合曲线')
    plt.title(s='BP神经网络拟合非线性曲线')
    ax2 = fig.add_subplot(122)
    ax2.plot(step_arr, loss_arr, color='red', label='误差曲线')
    plt.title(s='BP神经网络误差下降曲线')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    np.random.seed(1)
    inpit_n_row = 100  # 输入层神经元节点行数
    input_n_col = 1  # 输入层神经元节点列数
    hidden_n_1 = 24  # 第一个隐藏层节点数
    hidden_n_2 = 3  # 第二个隐藏层节点数
    # X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    # y = np.array([[0], [1], [1], [0], [0], [1], [1], [0]])
    X = np.arange(-5, 5, 0.1)[:, np.newaxis]  # 输入层矩阵
    y = 5 * np.sin(X) + 2 * np.random.random()  # 输出层矩阵
    # 第一个隐藏层权重矩阵
    W1 = np.random.randn(input_n_col, hidden_n_1) / np.sqrt(inpit_n_row)
    b1 = np.zeros((inpit_n_row, hidden_n_1))
    # 第二个隐藏层权重矩阵
    W2 = np.random.randn(hidden_n_1, hidden_n_2) / np.sqrt(inpit_n_row)
    b2 = np.zeros((inpit_n_row, hidden_n_2))
    # 输出层权重矩阵
    W3 = np.random.randn(hidden_n_2, 1) / np.sqrt(inpit_n_row)
    b3 = np.zeros((inpit_n_row, 1))
    W, B = [W1, W2, W3], [b1, b2, b3]
    train_BP(X=X, y=y, W=W, B=B, learn_rate=0.15, decay=0.5)  # 开始训练BP神经网络
