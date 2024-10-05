"""
本文件定义了几个用于数据预处理的操作，包括获取mat文件路径、读取mat文件并返回numpy数组、由数组获取dataloader等。
"""
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader



def get_mat_paths(data_dir="D:\\SRT_Dataset\\Data_Download"):
    '''
    Get the paths of all.mat files in the given directory.

    Parameters:
    - data_dir: str. 数据集所在文件夹路径
    Return:
    - mat_paths: list of str. 所有mat文件的路径构成的列表

    Directory Structure:
    Data_Download
    ├── Tail_652_1
    │   ├── 652200101092009.mat
    │   ├── 652200101092046.mat
    │   ├──...
    ├── Tail_652_2
    │   ├── 652200107282140.mat
    │   ├── 652200107290614.mat
    │   ├──...
    ├──...
    ├──...
    ├──...
    ├── Tail_652_8
    │   ├── 652200306252041.mat
    │   ├── 652200306260449.mat
    │   ├──...
    '''
    # 获取文件夹下的所有文件名称
    folder_names = os.listdir(data_dir) # ['Tail_652_1','Tail_652_2',...,'Tail_652_8']

    # 获取所有.mat文件路径
    mat_paths=[]
    for folder_name in folder_names:
         folder_path=os.path.join(data_dir,folder_name) # mat所在文件夹的存储路径
         mat_names=os.listdir(folder_path)
         for mat_name in mat_names:
             mat_paths.append(os.path.join(folder_path,mat_name))
    return mat_paths



def get_XY(mat_paths, input_len, output_len,
            var_names=['ALT', 'ALTR', "TAS", 'GS', 'AOA1', 'AOA2', 'PTCH', 'WS', "WD", 'SAT', 'TAT', 'PI', 'PT'],
            indices=[i for i in range(0,100)],
            verbose=1
            ):
    '''
    Get organized dataset X, Y and grouped dataset X_grouped, Y_grouped (numpy arrays) from mat files.

    Parameters:
    - mat_paths: list of str. 所有mat文件的路径构成的列表
    - input_len: int. Input sequence length. 输入序列长度
    - output_len: int. Output sequence length. 输出序列长度
    - var_names: list of str. Variable names. 要处理的变量，即实际用到的变量
    - indices: list of int. Indices of mat files to be used.
    - verbose: int.

    Return:
    - X: np.ndarray. shape=(N, input_len, len(var_names))
    - Y: np.ndarray. shape=(N, output_len, len(var_names))
    - X_grouped: list of (list of (input_length,len(var_names)) numpy array).
    - Y_grouped: list of (list of (output_length,len(var_names)) numpy array).
 
    可能用到的变量：
    WS: WIND SPEED
    TAS: TRUE AIRSPEED LSP
    GS: GROUND SPEED LSP

    WD: WIND DIRECTION TRUE
    PTCH: PITCH ANGLE LSP
    AOA1: ANGLE OF ATTACK 1
    AOA2: ANGLE OF ATTACK 2

    FPAC: FLIGHT PATH ACCELERATION
    LATG: LATERAL ACCELERATION
    LONG: LONGITUDINAL ACCELERATION
    VRTG: VERTICAL ACCELERATION
    CTAC: CROSS TRACK ACCELERATION

    ALT: PRESSURE ALTITUDE LSP
    ALTR: ALTITUDE RATE
    '''

    import scipy.io
    #var_names=['WS','WD','TAS', 'ALTR'] # 要处理的变量
    #indices=[i for i in range(0,400)] # 要处理的mat文件的序号
    # 以下为数据清洗后得到的正常mat文件的序号
    #indices=[6, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 40, 41, 42, 44, 46, 47, 48, 49, 50, 53, 54, 57, 58, 66, 67, 68, 69, 70, 80, 81, 84, 85, 86, 87, 89, 92, 93, 95, 96, 97, 98, 99, 105, 108, 110, 113, 114, 115, 116, 117, 120, 121, 122, 123, 124, 125, 127, 128, 129, 131, 137, 138, 142, 143, 144, 146, 147, 148, 149, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 168, 170, 171, 173, 174, 175, 177, 179, 181, 182, 194, 195, 196, 197, 202, 203, 204, 206, 207, 208, 211, 212, 213, 216, 217, 218, 219, 220, 221, 222, 223, 227, 229, 231, 232, 233, 239, 240, 243, 244, 245, 246, 249, 250, 252, 253, 255, 256, 257, 258, 260, 261, 262, 264, 270, 271, 272, 273, 276, 277, 279, 281, 282, 283, 288, 292, 293, 294, 295, 296, 297, 298, 301, 302, 304, 305, 306, 307, 308, 309, 310, 312, 313, 314, 316, 319, 320, 323, 324, 325, 330, 331, 332, 333, 339, 340, 343, 344, 348, 371, 378, 379, 380, 381, 382, 385, 386, 387, 388, 390, 391, 394, 395, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 413, 414, 415, 416, 417, 418, 420, 421, 423, 424, 426, 432, 433, 434, 435, 444, 445, 446, 447, 451, 452, 453, 454, 459, 460, 461, 462, 463, 465, 466, 469, 470, 471, 472, 483, 484, 485, 486, 490, 491, 493, 494, 495, 499, 500, 502, 503, 504, 507, 509, 511, 513, 514, 515, 516, 517, 523, 524, 525, 528, 529, 530, 531, 532, 533, 536, 537, 538, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 556, 557, 558, 560, 561, 562, 563, 564, 568, 570, 575, 577, 578, 579, 580, 581, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 599, 600, 601, 604, 605, 606, 607, 610, 611, 612, 614, 615, 619, 620, 621, 623, 624, 626, 627, 628, 631, 632, 633, 642, 644, 647, 650, 651, 652, 654, 658, 659, 660, 661, 662, 663, 664, 665, 667, 668, 669, 671, 672, 673, 675, 677, 679, 680, 681, 683, 684, 685, 687, 689, 690, 691, 692, 693, 695, 697, 698, 699, 703, 704, 705, 706, 707, 708, 709, 710, 711, 713, 714, 715, 718, 720, 721, 722, 729, 732, 733, 734, 735, 737, 739, 743, 744, 745, 746, 749, 750, 751, 752, 754, 756, 759, 760, 761, 763, 764, 766, 771, 772, 773, 774, 775, 776, 778, 780, 784, 785, 787, 789, 795, 796, 797, 803, 806, 807, 812, 813, 815, 816, 817, 818, 820, 822, 826, 827, 828, 829, 833, 836, 837, 841, 842, 843, 845, 846, 847, 855, 856, 858, 859, 860, 861, 862, 868, 869, 870, 872, 876, 877, 886, 887, 888, 889, 890, 891, 892, 896, 897, 898, 899, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 912, 913, 914, 915, 916, 917, 918, 920, 924, 925, 926, 928, 933, 935, 937, 938, 940, 941, 943, 945, 947, 948, 950, 951, 955, 959, 960, 961, 962, 965, 967, 968, 969, 974, 978, 982, 983, 984, 986, 987, 991, 992]

    DATA=[] # 所用到的整个数据库
    count=0 # 已处理mat文件数量
    for i in indices: # 遍历mat文件
        # 每轮处理一个mat
        X_this_mat=[] # 待采样的本mat中的数据
        mat_data=scipy.io.loadmat(mat_paths[i]) # dict. 读取到的mat数据
        mat_data_len=mat_data['WSHR'][0][0][0].shape[0] # 本mat文件的时间步长度
        for var_name in var_names: # 遍历各变量, 每次处理本mat文件中一个变量的数据
            var_data=mat_data[var_name][0][0][0] # 读取到的本mat中某变量的数据. numpy array. Shape: (var_len, 1)
            var_rate=mat_data[var_name][0][0][1][0][0] # 该变量采样率
            var_this_mat=var_data[::var_rate, 0]
            X_this_mat.append(var_this_mat) # 把该变量的时间序列数据添加至X_this_mat(本mat的数据)中
        #X_this_mat: shape: (len(var_names), mat_data_len)
        X_this_mat=np.array(X_this_mat) # 将数据转换为numpy数组
        DATA.append(X_this_mat.T) # 把该mat的数据添加至数据库中
        count+=1
        if verbose==1 and count%10==0:
            print(f'GenerateX: {count}th mat completed.')
    # DATA: list of numpy arrays of shape (mat_data_len, len(var_names))
    '''
    标准归一化
    '''
    var_sum=np.zeros((len(var_names)))
    var_mean=np.zeros((len(var_names)))
    var_sum2=np.zeros((len(var_names)))
    var_std_dev=np.zeros((len(var_names)))
    count=0 # 总时间长度

    # 计算各变量均值
    for mat_data in DATA:
        # mat_data: numpy array. Shape: (mat_data_len, len(var_names))
        for i in range(mat_data.shape[0]):
            var_sum+=mat_data[i]
            count+=1
    var_mean=var_sum/count

    # 计算各变量方差
    for mat_data in DATA:
        # mat_data: numpy array. Shape: (mat_data_len, len(var_names))
        for i in range(mat_data.shape[0]):
            var_sum2+=(mat_data[i]-var_mean)**2
    var_std_dev=np.sqrt(var_sum2/count)

    # 转化数据
    for mat_data in DATA:
        # mat_data: numpy array. Shape: (mat_data_len, len(var_names))
        for i in range(mat_data.shape[0]):
            mat_data[i]=(mat_data[i]-var_mean)/var_std_dev

    '''
    根据采样出的数据库构建用于训练的数据集
    X_grouped: list of(list of (input_length,len(var_names)) numpy array).
    '''
    X_grouped=[]
    Y_grouped=[]
    for DATA_i in DATA: # DATA_i: numpy array. Shape: (mat_data_len, len(var_names))
        DATA_i_length=DATA_i.shape[0]
        X_i=[]
        Y_i=[]
        for i in range(0, DATA_i_length-input_len-output_len, output_len):
            X_i.append(DATA_i[i:i+input_len,:])
            Y_i.append(DATA_i[i+input_len:i+input_len+output_len,:])
        X_grouped.append(X_i)
        Y_grouped.append(Y_i)

    if verbose==1:
        print('len(X_grouped):', len(X_grouped))
        print('len(Y_grouped):', len(Y_grouped))

    # Construct data set (in numpy arrays)
    X=[]
    Y=[]
    for X_i in X_grouped:
        for X_ij in X_i:
            X.append(X_ij)
    for Y_i in Y_grouped:
        for Y_ij in Y_i:
            Y.append(Y_ij)
    X=np.array(X).astype("float32")
    Y=np.array(Y).astype("float32")

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
    assert X.shape[0]==Y.shape[0], 'X and Y must have the same amountof samples.'
    import random

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


#----------------------------------------------------------------------------------------------------------------------

# 测试中，不完善
def get_XY_with_positional_encoding(mat_paths, input_len, output_len,
            var_names=['ALT', 'ALTR', "TAS", 'GS', 'AOA1', 'AOA2', 'PTCH', 'WS', "WD", 'SAT', 'TAT', 'PI', 'PT'],
            indices=[i for i in range(0,100)],
            verbose=1
            ): # 获取数据集numpy数组
    '''
    Parameters:
    - mat_paths: list of str.
    - input_len: int. Input sequence length.
    - output_len: int. Output sequence length.
    - var_names: list of str. Variable names.
    - indices: list of int. Indices of mat files to be used.
    - verbose: int.

    Return:
    - X: np.ndarray. shape=(N, input_len, len(var_names))
    - Y: np.ndarray. shape=(N, output_len, len(var_names))
    - X_grouped: list of(list of (input_len,len(var_names)) numpy array).
    - Y_grouped: list of(list of (output_len,len(var_names)) numpy array).


    mat_paths: list of str. 所有mat文件的路径构成的列表
    input_len: int. 输入序列长度
    output_len: int. 输出序列长度
    var_names: list of str. 要处理的变量，即实际用到的变量

    可能用到的变量：
    WS: WIND SPEED
    TAS: TRUE AIRSPEED LSP
    GS: GROUND SPEED LSP

    WD: WIND DIRECTION TRUE
    PTCH: PITCH ANGLE LSP
    AOA1: ANGLE OF ATTACK 1
    AOA2: ANGLE OF ATTACK 2

    FPAC: FLIGHT PATH ACCELERATION
    LATG: LATERAL ACCELERATION
    LONG: LONGITUDINAL ACCELERATION
    VRTG: VERTICAL ACCELERATION
    CTAC: CROSS TRACK ACCELERATION

    ALT: PRESSURE ALTITUDE LSP
    ALTR: ALTITUDE RATE
    '''

    import scipy.io
    #var_names=['WS','WD','TAS', 'ALTR'] # 要处理的变量
    #indices=[i for i in range(0,400)] # 要处理的mat文件的序号
    # 以下为数据清洗后得到的正常mat文件的序号
    #indices=[6, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 40, 41, 42, 44, 46, 47, 48, 49, 50, 53, 54, 57, 58, 66, 67, 68, 69, 70, 80, 81, 84, 85, 86, 87, 89, 92, 93, 95, 96, 97, 98, 99, 105, 108, 110, 113, 114, 115, 116, 117, 120, 121, 122, 123, 124, 125, 127, 128, 129, 131, 137, 138, 142, 143, 144, 146, 147, 148, 149, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 168, 170, 171, 173, 174, 175, 177, 179, 181, 182, 194, 195, 196, 197, 202, 203, 204, 206, 207, 208, 211, 212, 213, 216, 217, 218, 219, 220, 221, 222, 223, 227, 229, 231, 232, 233, 239, 240, 243, 244, 245, 246, 249, 250, 252, 253, 255, 256, 257, 258, 260, 261, 262, 264, 270, 271, 272, 273, 276, 277, 279, 281, 282, 283, 288, 292, 293, 294, 295, 296, 297, 298, 301, 302, 304, 305, 306, 307, 308, 309, 310, 312, 313, 314, 316, 319, 320, 323, 324, 325, 330, 331, 332, 333, 339, 340, 343, 344, 348, 371, 378, 379, 380, 381, 382, 385, 386, 387, 388, 390, 391, 394, 395, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 413, 414, 415, 416, 417, 418, 420, 421, 423, 424, 426, 432, 433, 434, 435, 444, 445, 446, 447, 451, 452, 453, 454, 459, 460, 461, 462, 463, 465, 466, 469, 470, 471, 472, 483, 484, 485, 486, 490, 491, 493, 494, 495, 499, 500, 502, 503, 504, 507, 509, 511, 513, 514, 515, 516, 517, 523, 524, 525, 528, 529, 530, 531, 532, 533, 536, 537, 538, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 556, 557, 558, 560, 561, 562, 563, 564, 568, 570, 575, 577, 578, 579, 580, 581, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 599, 600, 601, 604, 605, 606, 607, 610, 611, 612, 614, 615, 619, 620, 621, 623, 624, 626, 627, 628, 631, 632, 633, 642, 644, 647, 650, 651, 652, 654, 658, 659, 660, 661, 662, 663, 664, 665, 667, 668, 669, 671, 672, 673, 675, 677, 679, 680, 681, 683, 684, 685, 687, 689, 690, 691, 692, 693, 695, 697, 698, 699, 703, 704, 705, 706, 707, 708, 709, 710, 711, 713, 714, 715, 718, 720, 721, 722, 729, 732, 733, 734, 735, 737, 739, 743, 744, 745, 746, 749, 750, 751, 752, 754, 756, 759, 760, 761, 763, 764, 766, 771, 772, 773, 774, 775, 776, 778, 780, 784, 785, 787, 789, 795, 796, 797, 803, 806, 807, 812, 813, 815, 816, 817, 818, 820, 822, 826, 827, 828, 829, 833, 836, 837, 841, 842, 843, 845, 846, 847, 855, 856, 858, 859, 860, 861, 862, 868, 869, 870, 872, 876, 877, 886, 887, 888, 889, 890, 891, 892, 896, 897, 898, 899, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 912, 913, 914, 915, 916, 917, 918, 920, 924, 925, 926, 928, 933, 935, 937, 938, 940, 941, 943, 945, 947, 948, 950, 951, 955, 959, 960, 961, 962, 965, 967, 968, 969, 974, 978, 982, 983, 984, 986, 987, 991, 992]

    DATA=[] # 所用到的整个数据库
    count=0 # 已处理mat文件数量
    for i in indices: # 遍历mat文件
        # 每轮处理一个mat
        X_this_mat=[] # 待采样的本mat中的数据
        mat_data=scipy.io.loadmat(mat_paths[i]) # dict. 读取到的mat数据
        mat_data_len=mat_data['WSHR'][0][0][0].shape[0] # 本mat文件的时间步长度
        for var_name in var_names: # 遍历各变量, 每次处理本mat文件中一个变量的数据
            var_data=mat_data[var_name][0][0][0] # 读取到的本mat中某变量的数据. numpy array. Shape: (var_len, 1)
            var_rate=mat_data[var_name][0][0][1][0][0] # 该变量采样率
            var_this_mat=var_data[::var_rate, 0]
            X_this_mat.append(var_this_mat) # 把该变量的时间序列数据添加至X_this_mat(本mat的数据)中
        #X_this_mat: shape: (len(var_names), mat_data_len)
        X_this_mat=np.array(X_this_mat) # 将数据转换为numpy数组
        DATA.append(X_this_mat.T) # 把该mat的数据添加至数据库中
        count+=1
        if verbose==1 and count%10==0:
            print(f'GenerateX: {count}th mat completed.')
    # DATA: list of numpy arrays of shape (mat_data_len, len(var_names))
    if verbose==1:
        print('len(DATA): ', len(DATA))
        print('DATA[0].shape: ', DATA[0].shape)

    '''
    标准归一化
    '''
    var_sum=np.zeros((len(var_names)))
    var_mean=np.zeros((len(var_names)))
    var_sum2=np.zeros((len(var_names)))
    var_std_dev=np.zeros((len(var_names)))
    count=0 # 总时间长度

    # 计算各变量均值
    for mat_data in DATA:
        # mat_data: numpy array. Shape: (mat_data_len, len(var_names))
        for i in range(mat_data.shape[0]):
            var_sum+=mat_data[i]
            count+=1
    var_mean=var_sum/count

    # 计算各变量方差
    for mat_data in DATA:
        # mat_data: numpy array. Shape: (mat_data_len, len(var_names))
        for i in range(mat_data.shape[0]):
            var_sum2+=(mat_data[i]-var_mean)**2
    var_std_dev=np.sqrt(var_sum2/count)

    # 转化数据
    for mat_data in DATA:
        # mat_data: numpy array. Shape: (mat_data_len, len(var_names))
        for i in range(mat_data.shape[0]):
            mat_data[i]=(mat_data[i]-var_mean)/var_std_dev

    '''
    根据采样出的数据库构建用于训练的数据集
    X_grouped: list of(list of (input_length,len(var_names)) numpy array).
    '''

    def get_positional_encoding(max_len, d_model):
        """
        生成位置编码矩阵
        max_len: 最大序列长度
        d_model: 模型的维度
        return: 位置编码矩阵，形状为 (max_len, d_model)
        """
        position = np.arange(max_len)[:, np.newaxis]  # shape: (max_len, 1)
        d_model_=(d_model+1)//2*2 # 向上取偶数
        pos_enc = np.zeros((max_len, d_model_))  # 初始化位置编码矩阵
        div_term = np.exp(np.arange(0, d_model_, 2) * -(np.log(10000.0) / d_model_))
        pos_enc[:, 0::2] = np.sin(position * div_term) # even dimensions
        pos_enc[:, 1::2] = np.cos(position * div_term) # odd dimensions
        return pos_enc

    X_grouped=[]
    Y_grouped=[]
    X_mark_grouped=[]
    Y_mark_grouped=[]
    for DATA_i in DATA: # DATA_i: numpy array. Shape: (mat_data_len, len(var_names))
        DATA_i_length=DATA_i.shape[0]
        X_i=[]
        Y_i=[]
        X_mark_i=get_positional_encoding(DATA_i_length, len(var_names))
        Y_mark_i=get_positional_encoding(DATA_i_length, len(var_names))
        for i in range(0, DATA_i_length-input_len-output_len, output_len):
            X_i.append(DATA_i[i:i+input_len,:])
            Y_i.append(DATA_i[i+input_len:i+input_len+output_len,:])
        X_grouped.append(X_i)
        Y_grouped.append(Y_i)
        X_mark_grouped.append(X_mark_i)
        Y_mark_grouped.append(Y_mark_i)

    if verbose==1:
        print('len(X_grouped):', len(X_grouped))
        print('len(Y_grouped):', len(Y_grouped))
        print('len(X_mark_grouped):', len(X_mark_grouped))
        print('len(Y_mark_grouped):', len(Y_mark_grouped))

    # 构建数据集（numpy）
    X=[]
    Y=[]
    X_mark=[]
    Y_mark=[]
    for X_i in X_grouped:
        for X_ij in X_i:
            X.append(X_ij)
    for Y_i in Y_grouped:
        for Y_ij in Y_i:
            Y.append(Y_ij)
    for X_mark_i in X_mark_grouped:
        for X_mark_ij in X_mark_i:
            X_mark.append(X_mark_ij)
    for Y_mark_i in Y_mark_grouped:
        for Y_mark_ij in Y_mark_i:
            Y_mark.append(Y_mark_ij)
    X=np.array(X).astype("float32")
    Y=np.array(Y).astype("float32")
    X_mark=np.array(X_mark).astype("float32")
    Y_mark=np.array(Y_mark).astype("float32")

    if verbose==1:
        print('X shape: ', X.shape)
        print('Y shape: ', Y.shape)
    return X, Y, X_mark, Y_mark, X_grouped, Y_grouped, X_mark_grouped, Y_mark_grouped


# 测试中，不完善
def get_XY_loaders_with_positional_encoding(X, Y, X_mark, Y_mark, 
                    batch_size=32,
                    verbose=1
                    ): # 构建数据集loader
    '''
    Parameters:
    - X: numpy array. Shape: (num_samples, input_len, input_channels)
    - Y: numpy array. Shape: (num_samples, output_len, output_channels)
    - X_mark: numpy array. Shape: (num_samples, input_len, input_channels)
    - Y_mark: numpy array. Shape: (num_samples, output_len, output_channels)
    - batch_size: int.
    - verbose: int.
    Return:
    - train_loader, val_loader, test_loader
    '''
    assert X.shape[0]==Y.shape[0]==X_mark.shape[0]==Y_mark.shape[0]
    assert X.shape[1]==X_mark.shape[1]
    assert Y.shape[1]==Y_mark.shape[1]
    assert X.shape[2]==X_mark.shape[2]
    assert Y.shape[2]==Y_mark.shape[2]
    
    # 自定义数据集类
    class TimeSeriesDataset(Dataset):
        def __init__(self, X, Y, X_mark, Y_mark):
            self.X = X # shape: (num_samples, input_len, input_channels)
            self.Y = Y # shape: (num_samples, output_len, output_channels)
            self.X_mark = X_mark # shape: (num_samples, input_len, input_channels)
            self.Y_mark = Y_mark # shape: (num_samples, output_len, output_channels)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx], self.X_mark[idx], self.Y_mark[idx]

    # 构建数据集
    X,Y,X_mark, Y_mark=X.astype("float32"), Y.astype("float32"), X_mark.astype("float32"), Y_mark.astype("float32")
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

    train_dataset = TimeSeriesDataset(X[train_indices], Y[train_indices], X_mark[train_indices], Y_mark[train_indices])
    val_dataset = TimeSeriesDataset(X[val_indices], Y[val_indices], X_mark[val_indices], Y_mark[val_indices])
    test_dataset = TimeSeriesDataset(X[test_indices], Y[test_indices], X_mark[test_indices], Y_mark[test_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if verbose==1:
        print(f"Train dataset size: X: ({num_train}, {input_len}, {input_channels}); Y: ({num_train}, {output_len}, {output_channels})")
        print(f"Val dataset size: X: ({num_val}, {input_len}, {input_channels}); Y: ({num_val}, {output_len}, {output_channels})")
        print(f"Test dataset size: X: ({num_test}, {input_len}, {input_channels}); Y: ({num_test}, {output_len}, {output_channels})")

    return train_loader, val_loader, test_loader

