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