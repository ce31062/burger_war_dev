import torch
import torch.nn as nn
import torch.nn.functional as F

print("net.pyを実行する")
class Net(nn.Module):
    def __init__(self, output_size):
        """
        Args:
            output_size (int): size of output
        """
        super(Net, self).__init__()

        '''
        # input state
        self.state = {
            "lidar": self.lidar_ranges,     # (1, 1, 360)
            "map": map,                     # (1, 2, 16, 16)
            "image": self.image             # (1, 3, 480, 640)
        }
        '''
        # OpenAI: Emergent Tool Use from Multi-Agent Interaction
        # https://openai.com/blog/emergent-tool-use/
        # https://pira-nino.hatenablog.com/entry/introduce_openai_hide-and-seek

        # Core network
        self.block_lidar = nn.Sequential(
            # Input size: (1, 1, 360)
            nn.Conv1d(1, 16, 3, padding=2, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 64, 3, padding=2, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, 3, padding=2, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3),
            nn.AvgPool1d(kernel_size=30)  # Global Average Pooling
            # Output size: (1, 128, 1)
        )
        self.block_map = nn.Sequential(
            # Input size: (1, 2, 16, 16)
            nn.Conv2d(2, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AvgPool2d(kernel_size=4)  # Global Average Pooling
            # Output size: (1, 128, 1, 1)
        )
        self.block_image = nn.Sequential(
            # Input size: (1, 3, 480, 640)
            nn.Conv2d(3, 8, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AvgPool2d(kernel_size=(15, 20))  # Global Average Pooling
            # Output size: (1, 128, 1, 1)
        )
        self.block_merge = nn.Sequential(
            # Input size: (1, 3, 128)
            nn.Conv1d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.AvgPool1d(kernel_size=32)  # Global Average Pooling
            # Output size: (1, 64, 1)
        )
        
        # Dueling network
        self.fc_adv = nn.Linear(64, output_size)
        self.fc_val = nn.Linear(64, 1)

    def forward(self, lidar, map, image):
        # Core network
        ## Process each input
        x = self.block_lidar(lidar)
        y = self.block_map(map)
        z = self.block_image(image)

        ## Merge intermediate results
        x = torch.flatten(x, 1)    # (1, 128)
        y = torch.flatten(y, 1)    # (1, 128)
        z = torch.flatten(z, 1)    # (1, 128)
        w = torch.stack([x, y, z], dim=1)    # (1, 3, 128)
        w = self.block_merge(w)    # (1, 64, 1)
        w = torch.flatten(w, 1)    # (1, 64)

        # Dueling network
        adv = self.fc_adv(w)
        val = self.fc_val(w).expand(-1, adv.size(1))
        w = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return w

if __name__ == '__main__':
    def example(net, device_name):
        # Prepare sample datasets
        device = torch.device(device_name)
        net = net.to(device)
        lidar = torch.randn(data_sizes[0]).to(device)
        map = torch.randn(data_sizes[1]).to(device)
        image = torch.randn(data_sizes[2]).to(device)

        # Run
        import time
        print('[{}] Processing...'.format(device))
        start_time = time.time()
        val = net(lidar, map, image)
        elapsed_time = time.time() - start_time
        print('[{}] Done. {:.3f}[ms]'.format(device, elapsed_time))
        print(val[0])

    net = Net(5)

    # Summarize
    #from torchinfo import summary
    data_sizes = [
        (1, 1, 360),
        (1, 2, 16, 16),
        (1, 3, 480, 640)
    ]
    #summary(net, data_sizes)
    
    # Test run
    for _ in range(3):
        example(net, 'cpu')
    if torch.cuda.is_available():
        example(net, 'cuda:0')
    else:
        print('* CUDA not available.')
