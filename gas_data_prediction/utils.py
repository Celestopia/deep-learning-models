"""
本文件定义了一些工具函数，包括训练模型、绘图等，将一些复用次数多的代码封装在函数内，方便调用。
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import tqdm


class FitHistory:
    '''
    Fit history class to record the training history of a model.
    '''
    def __init__(self):
        self.num_epochs=0
        self.epoch_time=[]
        self.train_loss=[]
        self.train_metric=[]
        self.val_loss=[]
        self.val_metric=[]
        self.metadata=None # 用于保存额外信息

    def update(self, epoch_time, train_loss, train_metric, val_loss, val_metric):
        '''
        Parameters:
        - epoch_time: list. The time of training each epoch.
        - train_loss: list. The loss of training each epoch.
        - train_metric: list. The metric of training each epoch.
        - val_loss: list. The loss of validation each epoch.
        - val_metric: list. The metric of validation each epoch.
        '''
        self.num_epochs+=len(epoch_time)
        self.epoch_time.extend(epoch_time)
        self.train_loss.extend(train_loss)
        self.train_metric.extend(train_metric)
        self.val_loss.extend(val_loss)
        self.val_metric.extend(val_metric)

    def plot(self, figsize=(8,4)):
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, label='train_loss')
        plt.plot(self.val_loss, label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_metric, label='train_metric')
        plt.plot(self.val_metric, label='val_metric')
        plt.xlabel('Epochs')
        plt.ylabel('Metric')
        plt.suptitle("Training History")
        plt.legend()
        plt.tight_layout() # 调整子图间距，防止重叠
        # plt.savefig(save_path, dpi=300, bbox_inches='tight') # 保存图片
        plt.show()
    
    def summary(self):
        print(f'Number of epochs:  {self.num_epochs}')
        print(f'Training time:     {np.sum(self.epoch_time):.4f}s')
        print(f'Training loss:     {self.train_loss[-1]:.4f}')
        print(f'Training metric:   {self.train_metric[-1]:.4f}')
        print(f'Validation loss:   {self.val_loss[-1]:.4f}')
        print(f'Validation metric: {self.val_metric[-1]:.4f}')


def train(MODEL, train_loader, val_loader, optimizer,
            loss_func=nn.MSELoss(),
            metric_func=nn.L1Loss(),
            num_epochs=10,
            device='cpu',
            verbose=1
            ):
    if not hasattr(MODEL, 'label_len'): # 如果模型不含有label_len属性，说明前向传播过程不需要解码器输入
        epoch_time_list=[]
        train_loss_list=[]
        train_metric_list=[]
        val_loss_list=[]
        val_metric_list=[]
        total_time=0.0 # 总训练时间

        for epoch in tqdm.tqdm(range(num_epochs)):
            t1=time.time() # 该轮开始时间
            train_loss, train_metric = 0.0, 0.0 # 本轮的训练loss和metric
            val_loss, val_metric = 0.0, 0.0 # 本轮的验证loss和metric

            # 训练
            MODEL.train() # 切换到训练模式
            for inputs, targets in train_loader: # 分批次遍历训练集
                inputs, targets = inputs.to(device), targets.to(device) # 将数据转移到GPU（如果可用）
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
                    inputs, targets = inputs.to(device), targets.to(device) # 将数据转移到GPU（如果可用）
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

            # 计算训练用时
            t2=time.time() # 该轮结束时间
            total_time+=(t2-t1) # 累计训练时间

            # 记录本轮各指标值
            epoch_time_list.append(t2-t1)
            train_loss_list.append(average_train_loss)
            train_metric_list.append(average_train_metric)
            val_loss_list.append(average_val_loss)
            val_metric_list.append(average_val_metric)

            # 输出过程信息
            if verbose==1:
                message =f'Epoch [{str(epoch + 1).center(4, " ")}/{num_epochs}], Time: {(t2-t1):.4f}s'
                message+=f', Loss: {average_train_loss:.4f}'
                message+=f', Metric: {average_train_metric:.4f}'
                message+=f', Val Loss: {average_val_loss:.4f}'
                message+=f', Val Metric: {average_val_metric:.4f}'
                print(message)
        print(f'Total Time: {total_time:.4f}s')

        return (epoch_time_list, train_loss_list, train_metric_list, val_loss_list, val_metric_list)

    elif hasattr(MODEL, 'label_len') and MODEL.label_len > 0: # 如果模型含有label_len属性，说明前向传播过程需要解码器输入，训练过程考虑label
        label_len=MODEL.label_len
        output_len=MODEL.output_len
        pred_len=output_len-label_len

        epoch_time_list=[]
        train_loss_list=[]
        train_metric_list=[]
        val_loss_list=[]
        val_metric_list=[]
        total_time=0.0 # 总训练时间

        for epoch in tqdm.tqdm(range(num_epochs)):
            t1 = time.time() # 该轮开始时间
            train_loss, train_metric = 0.0, 0.0 # 本轮的训练loss和metric
            val_loss, val_metric = 0.0, 0.0 # 本轮的验证loss和metric

            # 训练
            MODEL.train() # 切换到训练模式
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device) # 将数据转移到GPU（如果可用）
                optimizer.zero_grad() # 清空梯度
                # decoder input
                dec_inp = torch.zeros_like(targets[:, -pred_len:, :]).float().to(device)
                dec_inp = torch.cat([targets[:, :label_len, :], dec_inp], dim=1).float().to(device)
                # encoder - decoder
                outputs = MODEL(inputs, dec_inp)
                outputs = outputs[:, -pred_len:, :].to(device) # 取待预测时间范围内的数据
                targets = targets[:, -pred_len:, :].to(device) # 取待预测时间范围内的数据
                loss = loss_func(outputs, targets)
                metric = metric_func(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss+=loss.item()
                train_metric+=metric.item()
            
            # 验证
            MODEL.eval() # 切换到验证模式
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device) # 将数据转移到GPU（如果可用）
                    dec_inp = torch.zeros_like(targets[:, -pred_len:, :]).float().to(device)
                    dec_inp = torch.cat([targets[:, :label_len, :], dec_inp], dim=1).float().to(device)
                    outputs = MODEL(inputs, dec_inp)
                    outputs = outputs[:, -pred_len:, :].to(device) # 取待预测时间范围内的数据
                    targets = targets[:, -pred_len:, :].to(device) # 取待预测时间范围内的数据
                    loss = loss_func(outputs, targets)
                    metric = metric_func(outputs, targets)
                    val_loss+=loss.item()
                    val_metric+=metric.item()
            average_train_loss=train_loss/len(train_loader)
            average_train_metric=train_metric/len(train_loader)
            average_val_loss=val_loss/len(val_loader)
            average_val_metric=val_metric/len(val_loader)

            # 计算训练用时
            t2=time.time()
            total_time+=(t2-t1)

            # 记录本轮各指标值
            epoch_time_list.append(t2-t1)
            train_loss_list.append(average_train_loss)
            train_metric_list.append(average_train_metric)
            val_loss_list.append(average_val_loss)
            val_metric_list.append(average_val_metric)
            
            # 输出过程信息
            if verbose==1:
                message =f'Epoch [{str(epoch + 1).center(4, " ")}/{num_epochs}], Time: {(t2-t1):.4f}s'
                message+=f', Loss: {average_train_loss:.4f}'
                message+=f', Metric: {average_train_metric:.4f}'
                message+=f', Val Loss: {average_val_loss:.4f}'
                message+=f', Val Metric: {average_val_metric:.4f}'
                print(message)
        print(f'Total Time: {total_time:.4f}s')

        return (epoch_time_list, train_loss_list, train_metric_list, val_loss_list, val_metric_list)


def plot_predictions(MODEL, X_grouped, Y_grouped, var_names, mat_paths,
                    iii=6,
                    figsize=(16,12),
                    device='cpu'
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

    if hasattr(MODEL, 'label_len') and MODEL.label_len > 0: # 如果模型含有label_len属性，说明前向传播过程需要解码器输入
        label_len=MODEL.label_len
        output_len=MODEL.output_len
        pred_len=output_len-label_len
        dec_inp = torch.zeros_like(torch.Tensor(Y_to_predict[:, -pred_len:, :])).float().to(device)
        dec_inp = torch.cat([torch.Tensor(Y_to_predict[:, :label_len, :]).to(device), dec_inp], dim=1).float().to(device)
        Y_to_predict=Y_to_predict[:, -pred_len:, :] # 取待预测时间范围内的数据
        Y_predicted=MODEL(torch.Tensor(X_to_predict).to(device), dec_inp).cpu().detach().numpy() # 根据X_to_predict预测到的数据
    else: # 如果模型不含有label_len属性，说明前向传播过程不需要解码器输入
        Y_predicted=MODEL(torch.Tensor(X_to_predict).to(device)).cpu().detach().numpy() # 根据X_to_predict预测到的数据

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


def plot_predictions(MODEL, X_grouped, Y_grouped, var_names, mat_paths,
                    iii=6,
                    figsize=(16,12),
                    device='cpu'
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

    if hasattr(MODEL, 'label_len') and MODEL.label_len > 0: # 如果模型含有label_len属性，说明前向传播过程需要解码器输入
        label_len=MODEL.label_len
        output_len=MODEL.output_len
        pred_len=output_len-label_len
        dec_inp = torch.zeros_like(torch.Tensor(Y_to_predict[:, -pred_len:, :])).float().to(device)
        dec_inp = torch.cat([torch.Tensor(Y_to_predict[:, :label_len, :]).to(device), dec_inp], dim=1).float().to(device)
        Y_to_predict=Y_to_predict[:, -pred_len:, :] # 取待预测时间范围内的数据
        Y_predicted=MODEL(torch.Tensor(X_to_predict).to(device), dec_inp).cpu().detach().numpy() # 根据X_to_predict预测到的数据
    else: # 如果模型不含有label_len属性，说明前向传播过程不需要解码器输入
        Y_predicted=MODEL(torch.Tensor(X_to_predict).to(device)).cpu().detach().numpy() # 根据X_to_predict预测到的数据

    output_channels=Y_predicted.shape[2]
    Y_predicted_flatten=Y_predicted.reshape(-1,output_channels)
    Y_to_predict_flatten=Y_to_predict.reshape(-1,output_channels)
    loss=np.mean((Y_predicted_flatten-Y_to_predict_flatten)**2) # 计算预测误差

    plt.figure(figsize=figsize) # figsize is specified in the function parameter
    import os
    plt.suptitle('Time Series Prediction on {}\n Loss: {:.4f}'.format(os.path.basename(mat_paths[iii]), loss))
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

