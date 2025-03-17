
import torch
import torch.nn as nn

class neural_network_model(nn.Module):
    def __init__(self , input_channels = 13 , action_dim = 10):
        
        super(neural_network_model , self).__init__()

        self.fc1 = nn.Linear(input_channels * 19 * 13 , 256)

        self.out = nn.Linear(256 , action_dim)

    def forward(self, x):
        if x.dim() == 4: 
            x = x.permute(0, 3, 1, 2) 
        elif x.dim() == 3: 
            x = x.permute(2, 0, 1).unsqueeze(0)  
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        
        return self.out(x) 
    
# This net keeps track of visited states
class exploration_network(nn.Module):
    def __init__(self, input_channels=13):
        super(exploration_network, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        nn.init.dirac_(self.conv1.weight)
        
    def forward(self, x):
        if x.dim() == 4: 
            x = x.permute(0, 3, 1, 2) 
        elif x.dim() == 3: 
            x = x.permute(2, 0, 1).unsqueeze(0) 
        return self.conv1(x).squeeze(1) 