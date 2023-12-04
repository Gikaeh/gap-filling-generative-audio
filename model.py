import torch
import torch.nn as nn

class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, mel_spectrogram_length, out_channels, dilation):
        super(WaveNetBlock, self).__init__()
        
        # dilated convolutional layer
        self.dilated_conv_before = nn.Conv1d(in_channels, out_channels, kernel_size = 2, dilation=dilation)
        self.dilated_conv_after = nn.Conv1d(in_channels, out_channels, kernel_size = 2, dilation=dilation)
        self.dilated_conv_mel = nn.Conv1d(mel_spectrogram_length, out_channels, kernel_size = 2, dilation=dilation) 

        # residual convolutional layer
        self.residual_before_conv = nn.Conv1d(out_channels, out_channels, kernel_size = 1)
        self.residual_after_conv = nn.Conv1d(out_channels, out_channels, kernel_size = 1)

        # skip connection convolutional layer
        self.skipConn_conv = nn.Conv1d(out_channels, out_channels, kernel_size = 1)

    def forward(self, before, after, mel_spectrogram):
        # compute dilated convolution
        dilation = torch.tanh(self.dilated_conv_before(before) + self.dilated_conv_after(after) + self.dilated_conv_mel(mel))
        gated_output = torch.sigmoid(self.dilated_conv_before(before) + self.dilated_conv_after(after) + self.dilated_conv_mel(mel))
        output_final = dilation * gated_output
        # compute residual and skip convolution
        residual_before = self.residual_before_conv(output_final)
        residual_after = self.residual_after_conv(output_final)
        skip = self.skipConn_conv(output_final)

        # Return the sum of the input and residual for the next block,
        # and the skip connection for aggregation in the main WaveNet model
        return residual_before + before[:, :, :-residual_before.shape[2]], residual_after + after[:, :, :-residual_after.shape[2]], skip

class WaveNet(nn.Module):
    def __init__(self, in_channels, residual_channels, mel_spectrogram_length, skip_channels, num_blocks, output_length):
        super(WaveNet, self).__init__()

        # input layer
        self.start_convLayer_before = nn.Conv1d(in_channels, residual_channels, kernel_size = 2)
        self.start_convLayer_after = nn.Conv1d(in_channels, residual_channels, kernel_size = 2)

        # get number of blocks
        self.blocks = nn.ModuleList([
            WaveNetBlock(residual_channels, mel_spectrogram_length, skip_channels, 2 ** i) for i in range(num_blocks)
        ])

        # process skip connections
        self.end_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.end_conv2 = nn.Conv1d(skip_channels, in_channels, kernel_size=1)
        self.output_length = output_length

    def forward(self, before, after, mel_spectrogram):
        # Initial convolution
        before = self.start_convLayer_before(before)
        after = self.start_convLayer_after(after)

        # Summing skip connections from all blocks
        skip_connections = 0
        for block in self.blocks:
            before, after, skip = block(before, after, mel_spectrogram)
            skip_connections += skip

        # Applying ReLU to the sum of skip connections
        x = torch.relu(skip_connections)

        # Two final convolutional layers
        x = torch.relu(self.end_conv1(x))
        x = self.end_conv2(x)[:self.output_length]

        return x

