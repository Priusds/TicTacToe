import torch
import torch.nn as nn


class ResNet(nn.Module):
    
    def __init__(self, n_channels_in, n_features):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(n_channels_in, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(n_features)

        
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(n_features)

        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        
        y = torch.cat((x,y), dim=1)
        return self.relu(y)


class Net(nn.Module):
    def __init__(self, n_features, n_residual_blocks, n_actions, h=3,w=3):
        super(Net, self).__init__()
        self.n_features = n_features
        self.n_residual_blocks = n_residual_blocks

        self.conv = nn.Sequential(
            nn.Conv2d(3, n_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_features),
            nn.ReLU()
        ) # output shape : (N, n_channels_conv_block, w, h)

        self.residuals = nn.ModuleList(
            [ResNet(n_features*(i+1), n_features) for i in range(n_residual_blocks)]
        )
        self.n_features_after_residuals = n_features*(n_residual_blocks+1)
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(self.n_features_after_residuals, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(h*w*2,n_actions)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(self.n_features_after_residuals, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(h*w, n_features),
            nn.ReLU(),
            nn.Linear(n_features,1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)

        for i in range(self.n_residual_blocks):
            x = self.residuals[i](x)

        policy = self.policyHead(x)

        value = self.valueHead(x)


        return torch.cat((policy, value), dim=1)

    
    def trainable_params(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))