import torch
import torch.nn as nn


# Base class for time series neural network models
class TimeSeriesNN(nn.Module):
    '''
    (batch_size, input_len, input_channels) -> (batch_size, output_len, output_channels)
    The overall architecture is direct multistep (DMS), rather than iterated multi-step (IMS). That is, directly predicting T future time steps (T>1).
    Usually input_channels == output_channels, but for generalizability, we set them separately.
    '''
    def __init__(self, input_len, output_len, input_channels, output_channels):
        super(TimeSeriesNN, self).__init__()
        self.input_len = input_len
        self.input_channels = input_channels
        self.output_len = output_len
        self.output_channels = output_channels

    # placeholder for forward function, to be implemented by subclasses
    def forward(self, x): # x: (batch_size, input_len, input_channels)
        pass
    
    # evaluate the model on a given dataset
    def evaluate(self, data,
                loss=nn.functional.mse_loss,
                mode='data_loader'
                ):
        if mode == 'numpy': # If mode is 'numpy', data should be a tuple of numpy arrays
            '''
            data: (inputs, targets)
            - inputs: (batch_size, input_len, input_channels)
            - targets: (batch_size, output_len, output_channels)
            '''
            inputs, targets = data
            assert inputs.ndim==3, 'inputs should be a 3D numpy array'
            assert targets.ndim==3, 'targets should be a 3D numpy array'
            assert inputs.shape[0]==targets.shape[0], 'inputs and targets should have the same batch size'
            assert inputs.shape[1:]==(self.input_len, self.input_channels), 'inputs should have shape (batch_size, input_len, input_channels)'
            assert targets.shape[1:]==(self.output_len, self.output_channels), 'targets should have shape (batch_size, output_len, output_channels)'
            inputs, targets = torch.from_numpy(inputs).float(), torch.from_numpy(targets).float()
            self.eval()  # switch to evaluation mode
            with torch.no_grad():
                outputs = self(inputs)
                result = loss(outputs, targets).item()
                return result
        
        elif mode == 'data_loader': # If mode is 'data_loader', data should be a DataLoader object
            '''
            data: DataLoader object
            '''
            import tqdm
            data_loader = data
            self.eval()  # switch to evaluation mode
            total_loss = 0.0
            with torch.no_grad():
                for inputs, targets in tqdm.tqdm(data_loader):
                    outputs = self(inputs)
                    total_loss += loss(outputs, targets).item() * inputs.size(0)
            return total_loss / len(data_loader.dataset)