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
    plt.bbox_inches='tight' # 确保保存时图片不丢失边界区域
    if save_path is not None:
        plt.savefig(save_path, dpi=300) # 保存图片
    plt.show()


class StandardScaler:
    '''
    Standardize the input along the specified axis.
    e.g. axis=0 means the mean and variance are calculated along the first dimension of the input.
    '''
    def __init__(self, axis):
        self.mean = None
        self.std = None
        self.axis = axis

    def fit(self, X):
        self.mean = np.mean(X, axis=self.axis)
        self.std = np.std(X, axis=self.axis)

    def transform(self, X):
        mean_mat=np.stack([np.mean(X, axis=self.axis) for _ in range(X.shape[self.axis])],axis=self.axis)
        std_mat=np.stack([np.std(X, axis=self.axis) for _ in range(X.shape[self.axis])],axis=self.axis)
        return (X - mean_mat) / std_mat

    def inverse_transform(self, X):
        mean_mat=np.stack([np.mean(X, axis=self.axis) for _ in range(X.shape[self.axis])],axis=self.axis)
        std_mat=np.stack([np.std(X, axis=self.axis) for _ in range(X.shape[self.axis])],axis=self.axis)
        return X*std_mat+mean_mat