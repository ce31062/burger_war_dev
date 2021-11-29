import torch
import torch.nn as nn
import torch.nn.functional as F

print("net2.pyを実行する")
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
            nn.Conv1d(1, 16, 3, padding=2, padding_mode='circular'),    # (1, 16, 360)
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=6),                                # (1, 16, 60)
            nn.Conv1d(16, 32, 3, padding=2, padding_mode='circular'),   # (1, 32, 60)
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3),                                # (1, 32, 20)
            nn.Conv1d(32, 64, 3, padding=2, padding_mode='circular'),   # (1, 64, 20)
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64*20, 32)
        )

        self.block_map = nn.Sequential(
            # Input size: (1, 4, 16, 16)
            nn.Conv2d(4, 16, 3, 1),         # (1, 16, 14, 14)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),    # (1, 16, 7, 7)
            nn.Conv2d(16, 32, 3, 1),        # (1, 32, 5, 5)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32*5*5, 32)
        )

        self.block_image = nn.Sequential(
            # Input size: (1, 3, 480, 640)
            nn.Conv2d(3, 16, kernel_size=5, stride=2),  # (1, 16, 238, 318)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),                # (1, 16, 79, 106)
            nn.Conv2d(16, 32, kernel_size=5, stride=2), # (1, 32, 38, 51)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                # (1, 32, 19, 25)
            nn.Conv2d(32, 32, kernel_size=5, stride=2), # (1, 32, 8, 11)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32*8*11, 32)
        )
        
        # head
        self.fc = nn.Linear(64, output_size)

    def forward(self, lidar, map, image):
        # Core network
        ## Process each input
        x = self.block_lidar(lidar)
        y = self.block_map(map)
        # z = self.block_image(image)

        ## Merge intermediate results
        #w = torch.cat([x, y, z], dim=1)    # (1, 96)
        w = torch.cat([x, y], dim=1)    # (1, 64)

        w = self.fc(w)

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
        (1, 4, 16, 16),
        (1, 3, 480, 640)
    ]
    #summary(net, data_sizes)
    
    # Test run
    example(net, 'cpu')
    example(net, 'cpu')
    example(net, 'cpu')
    if torch.cuda.is_available():
        example(net, 'cuda:0')
    else:
        print('* CUDA not available.')
