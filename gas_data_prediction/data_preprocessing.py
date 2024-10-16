"""
本文件定义了几个用于数据预处理的操作
"""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def get_XY(data_paths, input_len, output_len,
            label_len=0,
            var_names=['% O2', 'ppm CO', '% CO2', 'ppm NO', 'ppm NO2', '°C 烟温', 'ppm NOx', 'ppm SO2', '°C 环温', 'l/min 泵流量'],
            indices=None,
            verbose=1
            ):
    '''
    Get organized dataset X, Y and grouped dataset X_grouped, Y_grouped (numpy arrays) from mat files.

    Parameters:
    - data_paths: list of str. 所有数据文件的路径构成的列表
    - input_len: int. Input sequence length. 输入序列长度
    - output_len: int. Output sequence length. 输出序列长度
    - label_len: int. Label (start token) sequence length. 标签序列长度，即解码器启动序列的长度。默认为0，即不使用标签，此时输入序列与目标序列在时间轴上无重合。
    - var_names: list of str. Variable names. 要处理的变量，即实际用到的变量
    - indices: list of int. Indices of mat files to be used.
    - verbose: int.

    Return:
    - X: np.ndarray. shape=(N, input_len, len(var_names))
    - Y: np.ndarray. shape=(N, output_len, len(var_names))
    - X_grouped: list of (list of (input_length,len(var_names)) numpy array).
    - Y_grouped: list of (list of (output_length,len(var_names)) numpy array).
    '''
    assert label_len <= input_len, "label_len should be less than or equal to input_len"
    assert label_len < output_len, "label_len should be less than output_len"
    pred_len = output_len - label_len # Prediction sequence length # Note that when label_len==0, pred_len==output_len

    indices=[i for i in range(len(data_paths))] if indices is None else indices # 若未指定indices，则默认使用全部数据文件

    DATA=[] # 所用到的整个数据库
    count=0 # 已处理文件数量
    for i in indices: # 遍历数据文件
        # 每轮处理一个文件
        data_this_file=pd.read_excel(data_paths[i])[var_names].apply( # 读取指定变量（var_names）的数据
            lambda x:pd.to_numeric(x, errors='coerce') # 对于非数字值，用nan填充
            ).fillna(0 # 对于nan值，用0填充
                     ).to_numpy(dtype=float) # 将DataFrame数据转化为float型numpy数组
        DATA.append(data_this_file) # 把该数据文件的数据添加至数据库中
        count+=1
        if verbose==1 and count%10==0:
            print(f'GenerateX: {count}th mat completed.')
    # DATA: list of numpy arrays of shape (data_len, len(var_names))
    '''
    标准归一化
    '''
    var_sum=np.zeros((len(var_names)))
    var_mean=np.zeros((len(var_names)))
    var_sum2=np.zeros((len(var_names)))
    var_std_dev=np.zeros((len(var_names)))
    count=0 # 总时间长度
    # 计算各变量均值
    for data in DATA:
        # data: numpy array. Shape: (data_len, len(var_names))
        for i in range(data.shape[0]):
            var_sum+=data[i]
            count+=1
    var_mean=var_sum/count

    # 计算各变量方差
    for data in DATA:
        # data: numpy array. Shape: (data_len, len(var_names))
        for i in range(data.shape[0]):
            var_sum2+=(data[i]-var_mean)**2
    var_std_dev=np.sqrt(var_sum2/count)
    print(var_std_dev)

    # 转化数据
    for data in DATA: # data: numpy array. Shape: (data_len, len(var_names))
        for i in range(data.shape[0]):
            data[i]=(data[i]-var_mean)/var_std_dev
    

    '''
    根据采样出的数据库构建用于训练的数据集
    （确保屏幕宽度足够以显示完整）
                     |<----------input_len------------------->|
    Input sequence:  ├────────────────────────────────────────┤
    Label sequence:                          ├────────────────┤
    Target sequence:                         ├───────────────────────────────────────────────┤
                                             |<--label_len--->|<----------pred_len---------->|
                                             |<-----------------output_len------------------>|
    Time axis------------------------------------------------------------------------------------------->
    '''
    # X_grouped: list of(list of (input_length,len(var_names)) numpy array).
    X_grouped=[]
    Y_grouped=[]
    for DATA_i in DATA: # DATA_i: numpy array. Shape: (data_len, len(var_names))
        DATA_i_length=DATA_i.shape[0] # The (time series) length of the current mat data
        X_i=[]
        Y_i=[]
        for i in range(0, DATA_i_length-input_len-pred_len, pred_len):
            X_i.append(DATA_i[i:i+input_len,:])
            Y_i.append(DATA_i[i+input_len-label_len:i+input_len+pred_len,:]) # When label_len==0, X_i and Y_i don't intersect, and pred_len==output_len
        X_grouped.append(X_i)
        Y_grouped.append(Y_i)

    if verbose==1:
        print('len(X_grouped):', len(X_grouped))
        print('len(Y_grouped):', len(Y_grouped))

    # Part each time series into subseries with equal shape, in convenience of neural network training.
    X=[]
    Y=[]
    for X_i in X_grouped:
        for X_ij in X_i:
            X.append(X_ij)
    for Y_i in Y_grouped:
        for Y_ij in Y_i:
            Y.append(Y_ij)
    X=np.array(X).astype("float32") # X shape: (num_samples, input_len, len(var_names))
    Y=np.array(Y).astype("float32") # Y shape: (num_samples, output_len, len(var_names))
    
    if verbose==1:
        print('X shape: ', X.shape)
        print('Y shape: ', Y.shape)
    return X, Y, X_grouped, Y_grouped


def get_XY_loaders(X, Y,
                    batch_size=32,
                    verbose=1
                    ):
    '''
    Get data loaders for training, validation, and testing, from dataset in np.ndarray format.

    Parameters:
    - X: numpy array. Shape: (num_samples, input_len, input_channels)
    - Y: numpy array. Shape: (num_samples, output_len, output_channels)
    - batch_size: int.
    - verbose: int. Whether to print messages. If 1, print messages.
    Return:
    - train_loader, val_loader, test_loader
    '''
    assert type(X)==np.ndarray and type(Y)==np.ndarray, 'X and Y must be numpy arrays.'
    assert X.shape[0]==Y.shape[0], 'X and Y must have the same amount of samples.'

    # Customized dataset class # 自定义数据集类
    class TimeSeriesDataset(Dataset):
        def __init__(self, X, Y):
            self.X = X # shape: (num_samples, input_len, input_channels)
            self.Y = Y # shape: (num_samples, output_len, output_channels)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    # 构建数据集
    X,Y=X.astype("float32"), Y.astype("float32")
    num_samples=X.shape[0]
    input_len=X.shape[1]
    input_channels=X.shape[2]
    output_len=Y.shape[1]
    output_channels=Y.shape[2]

    train_ratio=0.7 # 训练集占比
    val_ratio=0.1 # 验证集占比
    test_ratio=0.2 # 测试集占比
    assert train_ratio+val_ratio+test_ratio<=1.0

    num_train=int(train_ratio*num_samples)
    num_val=int(val_ratio*num_samples)
    num_test=int(test_ratio*num_samples)
    assert num_train+num_val+num_test<=num_samples

    # 随机打乱数据集
    indices=list(range(num_samples))
    import random
    random.shuffle(indices)
    train_indices=indices[:num_train]
    val_indices=indices[num_train:num_train+num_val]
    test_indices=indices[num_train+num_val:num_train+num_val+num_test]

    train_dataset = TimeSeriesDataset(X[train_indices], Y[train_indices])
    val_dataset = TimeSeriesDataset(X[val_indices], Y[val_indices])
    test_dataset = TimeSeriesDataset(X[test_indices], Y[test_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if verbose==1:
        print(f"Train dataset size: X: ({num_train}, {input_len}, {input_channels}); Y: ({num_train}, {output_len}, {output_channels})")
        print(f"Val dataset size: X: ({num_val}, {input_len}, {input_channels}); Y: ({num_val}, {output_len}, {output_channels})")
        print(f"Test dataset size: X: ({num_test}, {input_len}, {input_channels}); Y: ({num_test}, {output_len}, {output_channels})")

    return train_loader, val_loader, test_loader