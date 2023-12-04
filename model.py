import torch
import torch.nn as nn

class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(WaveNetBlock, self).__init__()
        
        # dilated convolutional layer
        self.dilated_conv = nn.Conv1d(in_channels, out_channels, kernel_size = 2, dilation=dilation)

        # residual convolutional layer
        self.residual_conv = nn.Conv1d(out_channels, out_channels, kernel_size = 1)

        # skip connection convolutional layer
        self.skipConn_conv = nn.Conv1d(out_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        print(self.dilated_conv(x).shape)
        # compute dilated convolution
        dilation = torch.tanh(self.dilated_conv(x))
        gated_output = torch.sigmoid(self.dilated_conv(x))
        output_final = dilation * gated_output
        # compute residual and skip convolution
        residual = self.residual_conv(output_final)
        skip = self.skipConn_conv(output_final)

        # Return the sum of the input and residual for the next block,
        # and the skip connection for aggregation in the main WaveNet model
        return residual + x[:, :, :-residual.shape[2]], skip

class WaveNet(nn.Module):
    def __init__(self, in_channels, residual_channels, skip_channels, num_blocks, how_much):
        super(WaveNet, self).__init__()

        # input layer
        self.start_convLayer = nn.Conv1d(in_channels, residual_channels, kernel_size = 2)

        # get number of blocks
        self.blocks = nn.ModuleList([
            WaveNetBlock(residual_channels, skip_channels, 2 ** i) for i in range(num_blocks)
        ])

        # process skip connections
        self.end_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.end_conv2 = nn.Conv1d(skip_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Initial convolution
        x = self.start_convLayer(x)

        # Summing skip connections from all blocks
        skip_connections = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_connections += skip

        # Applying ReLU to the sum of skip connections
        x = torch.relu(skip_connections)

        # Two final convolutional layers
        x = torch.relu(self.end_conv1(x))
        x = self.end_conv2(x)[:how_much]

        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        
        return x