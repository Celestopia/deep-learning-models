import numpy as np
# 生成随机多元时间序列数据集
def load_test_data(id='0', input_dim=(1000,64,8), output_dim=(1000,16,4), noise=0.3, **kwargs):
    assert input_dim[0]==output_dim[0]
    if id=='0':
        '''
        生成num_samples对训练样本，每个输入的形状为(input_len,input_channels)，每个输出的形状为(output_len,output_channels)。
        默认生成1000对时间序列，每个输入时间序列的形状为(50,8)，输出时间序列的形状为(10,3)。
        '''
        num_samples, input_len, input_channels = input_dim
        _, output_len, output_channels = output_dim
        seq1=np.linspace(1,1000,input_len)
        seq2=np.linspace(1,1000,output_len)
        xs=[]
        ys=[]
        for i in range(input_channels):
            xs.append(np.sin(i*seq1)+np.cos((i+1)*seq1))
        for i in range(output_channels):
            ys.append(np.cos(2*i*seq2)+np.sin((2*i+1)*seq2))
        X_ = np.stack(xs, axis=1)
        Y_ = np.stack(ys, axis=1)
        
        X = np.stack([X_] * num_samples)
        Y = np.stack([Y_] * num_samples)
        Y = Y+noise*np.random.randn(*Y.shape) # 加入噪声，标准差为noise（默认为0.3）

        return X.astype(np.float32), Y.astype(np.float32) # X: (num_samples, input_len, input_channels); Y: (num_samples, output_len, output_channels)
    
    elif id=='1':
        '''
        默认生成1000个时间序列，每个时间序列的形状为(500,8)
        X: (num, time_steps, num_features)
        '''
        num_samples, time_steps, num_features = input_dim
        seq=np.linspace(1,1000,time_steps)
        xs=[]
        for i in range(num_features):
            xs.append(np.sin(i*seq)+np.cos((i+1)*seq))
        X = np.stack(xs, axis=1)
        X = np.stack([X] * num_samples)
        X = X+np.random.normal(0,noise,X.shape) # 加入噪声，标准差为noise（默认为0.3）
        
        return X.astype(np.float32)
    
    elif id=='2':
        '''
        生成num_samples对训练样本，每个输入的形状为(input_len,input_channels)，每个输出的形状为(output_len,output_channels)。
        默认生成1000对时间序列，每个输入时间序列的形状为(50,8)，输出时间序列的形状为(10,3)。
        与0的唯一区别是此处X也加了扰动。
        '''
        num_samples, input_len, input_channels = input_dim
        _, output_len, output_channels = output_dim
        seq1=np.linspace(1,1000,input_len)
        seq2=np.linspace(1,1000,output_len)
        xs=[]
        ys=[]
        for i in range(input_channels):
            xs.append(np.sin(i*seq1)+np.cos((i+1)*seq1))
        for i in range(output_channels):
            ys.append(np.cos(2*i*seq2)+np.sin((2*i+1)*seq2))
        X_ = np.stack(xs, axis=1)
        Y_ = np.stack(ys, axis=1)
        
        X = np.stack([X_] * num_samples)
        Y = np.stack([Y_] * num_samples)
        X = X+0.1*np.random.randn(*X.shape) # 加入噪声，标准差为noise（默认为0.1）
        Y = Y+noise*np.random.randn(*Y.shape) # 加入噪声，标准差为noise（默认为0.3）

        return X.astype(np.float32), Y.astype(np.float32) # X: (num_samples, input_len, input_channels); Y: (num_samples, output_len, output_channels)

def get_XY_loaders(X, Y,
                    batch_size=32,
                    train_ratio=0.7,
                    val_ratio=0.1,
                    test_ratio=0.2,
                    verbose=1
                    ):
    '''
    Get data loaders for training, validation, and testing.
    X: (num_samples, input_len, input_channels)
    Y: (num_samples, output_len, output_channels)
    type(X): numpy.ndarray
    type(Y): numpy.ndarray
    '''
    import random
    from torch.utils.data import DataLoader, Dataset

    assert type(X)==np.ndarray and type(Y)==np.ndarray
    assert X.shape[0]==Y.shape[0]
    assert X.ndim==3 and Y.ndim==3
    assert train_ratio+val_ratio+test_ratio<=1.0

    # 自定义数据集类
    class TimeSeriesDataset(Dataset):
        def __init__(self, X, Y):
            self.X = X # shape: (num_samples, input_len, input_channels)
            self.Y = Y # shape: (num_samples, output_len, output_channels)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    # 构建数据集
    if X.dtype!=np.float32:
        X=X.astype("float32")
    if Y.dtype!=np.float32:
        Y.astype("float32")
    
    num_samples=X.shape[0]
    input_len=X.shape[1]
    input_channels=X.shape[2]
    output_len=Y.shape[1]
    output_channels=Y.shape[2]

    num_train=int(train_ratio*num_samples)
    num_val=int(val_ratio*num_samples)
    num_test=int(test_ratio*num_samples)
    assert num_train+num_val+num_test<=num_samples
    indices=list(range(num_samples))
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
