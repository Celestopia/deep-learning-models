# 飞行数据预测

`flight_data_prediction`子文件夹实现了对飞机运行数据的预测，运行示例参考example.ipynb文件。可以在笔记本文件中调整学习率、批次大小、输入长度、预测长度等参数，也可以选择不同的预测模型。

唯一需要更改的是`data_preprocessing.get_mat_paths()`函数的路径参数，该参数指定了数据文件所在的文件夹路径，该文件夹结构如下：
```
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
```
例如，如果你的数据文件夹的绝对路径是`D:\Data\Data_Download`，则可以将`data_preprocessing.get_mat_paths()`函数的路径参数设置为`D:\\Data\\Data_Download`，亦即`mat_paths=data_preprocessing.get_mat_paths('D:\\Data\\Data_Download')`。

注意本项目并非可复用性高的成熟项目，其中函数的实现主要针对于本特定数据集，仅供测试用。


