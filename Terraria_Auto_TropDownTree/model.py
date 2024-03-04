import torch
import torch.nn as nn

class TwoStreamCNN(nn.Module):
    def __init__(self, num_classes):
        super(TwoStreamCNN, self).__init__()
        # Spatial stream
        self.spatial_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.spatial_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.spatial_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.spatial_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.spatial_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.spatial_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.spatial_fc1 = nn.Linear(256 * 15 * 20, 512)

        # Temporal stream
        self.temporal_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.temporal_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.temporal_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.temporal_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.temporal_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.temporal_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.temporal_fc1 = nn.Linear(256 * 15 * 20, 512)

        # Fusion
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)  # Assuming 10 possible key presses

    def forward(self, spatial_input, temporal_input):
        # Spatial stream
        spatial_out = self.spatial_conv1(spatial_input)
        spatial_out = self.spatial_pool1(spatial_out)
        spatial_out = self.spatial_conv2(spatial_out)
        spatial_out = self.spatial_pool2(spatial_out)
        spatial_out = self.spatial_conv3(spatial_out)
        spatial_out = self.spatial_pool3(spatial_out)
        spatial_out = spatial_out.view(spatial_out.size(0), -1)
        spatial_out = self.spatial_fc1(spatial_out)

        # Temporal stream
        temporal_out = self.temporal_conv1(temporal_input)
        temporal_out = self.temporal_pool1(temporal_out)
        temporal_out = self.temporal_conv2(temporal_out)
        temporal_out = self.temporal_pool2(temporal_out)
        temporal_out = self.temporal_conv3(temporal_out)
        temporal_out = self.temporal_pool3(temporal_out)
        temporal_out = temporal_out.view(temporal_out.size(0), -1)
        temporal_out = self.temporal_fc1(temporal_out)

        # Fusion
        fusion_out = torch.cat((spatial_out, temporal_out), dim=1)
        fusion_out = self.fc2(fusion_out)
        fusion_out = self.fc3(fusion_out)
        output = self.fc4(fusion_out)

        return output