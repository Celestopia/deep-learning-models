"""
本文件定义了一些工具函数，包括训练模型、绘图等，将一些复用次数多的代码封装在函数内，方便调用。
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import tqdm


def train(MODEL, train_loader, val_loader, optimizer,
            loss_func=nn.MSELoss(),
            metric_func=nn.L1Loss(),
            num_epochs=10
            # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ):
    # 训练模型
    fit_history=[]
    total_time=0.0 # 总训练时间
    for epoch in tqdm.tqdm(range(num_epochs)):
        t1=time.time() # 该轮开始时间
        train_loss, train_metric = 0.0, 0.0 # 本轮的训练loss和metric
        val_loss, val_metric = 0.0, 0.0 # 本轮的验证loss和metric

        # 训练
        MODEL.train() # 切换到训练模式
        for inputs, targets in train_loader: # 分批次遍历训练集
            optimizer.zero_grad() # 清空梯度
            outputs = MODEL(inputs)
            loss = loss_func(outputs, targets)
            metric = metric_func(outputs, targets)
            loss.backward() # 反向传播
            optimizer.step() # 更新权重
            train_loss+=loss.item()
            train_metric+=metric.item()

        # 验证
        MODEL.eval() # 切换到验证模式
        with torch.no_grad(): # 关闭梯度计算
            for inputs, targets in val_loader: # 分批次遍历验证集
                outputs = MODEL(inputs)
                loss = loss_func(outputs, targets)
                metric=metric_func(outputs, targets)
                val_loss+=loss.item()
                val_metric+=metric.item()

        # 计算各指标的平均值
        average_train_loss=train_loss/len(train_loader) # 本轮的平均训练loss
        average_train_metric=train_metric/len(train_loader) # 本轮的平均训练metric
        average_val_loss=val_loss/len(val_loader) # 本轮的平均验证loss
        average_val_metric=val_metric/len(val_loader) # 本轮的平均验证metric

        # 记录本轮各指标值
        fit_history.append((average_train_loss, average_train_metric, average_val_loss, average_val_metric))

        t2=time.time() # 该轮结束时间
        total_time+=(t2-t1) # 累计训练时间

        # 输出过程信息
        message=f'Epoch [{str(epoch + 1).center(4, " ")}/{num_epochs}], Time: {(t2-t1):.4f}s'
        message+=f', Loss: {average_train_loss:.4f}'
        message+=f', Metric: {average_train_metric:.4f}'
        message+=f', Val Loss: {average_val_loss:.4f}'
        message+=f', Val Metric: {average_val_metric:.4f}'
        print(message)
    print(f'Total Time: {total_time:.4f}s')
    return np.array(fit_history) # shape: (num_epochs, 4)


def plot_fit_history(fit_history, save_path=None):
    '''
    Parameters:
    - fit_history: numpy array, shape: (num_epochs, 4)
    Return:
    - None
    '''
    assert isinstance(fit_history, np.ndarray), "fit_history should be a numpy array"
    assert fit_history.ndim == 2, "fit_history should have 2 axes"
    assert fit_history.shape[1] == 4, "fit_history should have 4 columns"
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(fit_history[:,0], label='train_loss')
    plt.plot(fit_history[:,2], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(fit_history[:,1], label='train_metric')
    plt.plot(fit_history[:,3], label='val_metric')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.suptitle("Training History")
    plt.legend()
    plt.tight_layout() # 调整子图间距，防止重叠
    # plt.savefig(save_path, dpi=300, bbox_inches='tight') # 保存图片
    plt.show()


def plot_predictions(MODEL, X_grouped, Y_grouped, var_names, mat_paths,
                    iii=6,
                    figsize=(16,12)
                    ):
    '''
    Plot the predictions of the model on a given mat file.
    Parameters:
    - MODEL: torch.nn.Module, the trained model
    - X_grouped: list of (list of (input_length,len(var_names)) numpy array), the input data grouped by mat file
    - Y_grouped: list of (list of (output_length,len(var_names)) numpy array), the output data grouped by mat file
    - var_names: list of strings, the names of the variables
    - mat_paths: list of strings, the paths of the mat files
    - iii: int, the index of the mat file to be plotted
    - figsize: tuple of int, the size of the figure
    Return:
    - None
    '''
    # iii=43 # 要在某个mat文件上做预测,选中mat编号(iii)（建议落在测试集对应的编号范围内）

    X_to_predict=[] # 作为输入的真实数据
    Y_to_predict=[] # 待预测的真实数据

    for i in range(len(X_grouped[iii])):
        X_to_predict.append(X_grouped[iii][i])
    X_to_predict=np.array(X_to_predict) # X_to_predict: numpy array. Shape: (num_batches, input_len, input_channels)

    for i in range(len(Y_grouped[iii])):
        Y_to_predict.append(Y_grouped[iii][i])
    Y_to_predict=np.array(Y_to_predict) # Y_to_predict: numpy array. Shape: (num_batches, output_len, output_channels):

    Y_predicted=MODEL(torch.Tensor(X_to_predict)).detach().numpy() # 根据X_to_predict预测到的数据

    output_channels=Y_predicted.shape[2]
    Y_predicted_flatten=Y_predicted.reshape(-1,output_channels)
    Y_to_predict_flatten=Y_to_predict.reshape(-1,output_channels)
    loss=np.mean((Y_predicted_flatten-Y_to_predict_flatten)**2) # 计算预测误差

    plt.figure(figsize=figsize) # figsize is specified in the function parameter
    plt.suptitle('Time Series Prediction on {}\n Loss: {:.4f}'.format(mat_paths[iii][-30:], loss))
    for var_name in var_names:
        var_idx=var_names.index(var_name)
        plt.subplot(4, 4, var_idx+1)
        plt.plot(Y_predicted_flatten[:,var_idx], alpha=0.9, c='red')
        plt.plot(Y_to_predict_flatten[:,var_idx], alpha=0.6, c='blue')
        plt.legend(['predict', 'true'], loc='upper right')
        plt.title(var_name)
    plt.tight_layout(h_pad=2)
    #plt.savefig("", bbox_inches='tight')
    plt.show()

'''
def plot_BNN_predictions(Y_mean, Y_std, Y_to_predict, var_names, mat_paths, iii, output_channels):
    Y_mean=Y_mean.reshape(-1,output_channels)
    Y_std=Y_std.reshape(-1,output_channels)
    Y_to_predict=Y_to_predict.reshape(-1,output_channels)


    plt.figure(figsize=(16,12))
    plt.suptitle('Time Series Prediction on {}'.format(mat_paths[iii][-30:]))
    for var_name in var_names:
        var_idx=var_names.index(var_name)
        plt.subplot(4, 4, var_idx+1)
        plt.plot(range(Y_to_predict[:,var_idx].shape[0]), Y_mean[:,var_idx])
        plt.fill_between(range(Y_to_predict[:,var_idx].shape[0]), Y_mean[:,var_idx]-Y_std[:,var_idx], Y_mean[:,var_idx]+Y_std[:,var_idx], alpha=0.2)
        plt.scatter(range(Y_to_predict[:,var_idx].shape[0]), Y_to_predict[:,var_idx], s=1.5)

        plt.legend(['predict', 'uncertainty', 'true'], loc='upper right')
        plt.title(var_name)
    plt.tight_layout(h_pad=2)
    #plt.savefig("D:\\_SRT\\Dataset\\Result\\20240321\\CNN_prediction_fig1.png", bbox_inches='tight')
    plt.show()
'''