import torch
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, ReLU, BatchNorm2d, AvgPool2d, Dropout, Linear, Module
torch.manual_seed(1234)


class Bottleneck(Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bottle_neck = Sequential(
            BatchNorm2d(in_channels),
            ReLU(),
            Conv2d(in_channels, 4*growth_rate, kernel_size=1, bias=False),
            BatchNorm2d(4*growth_rate),
            ReLU(),
            Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )
    def forward(self, x):
        return torch.cat([self.bottle_neck(x), x], 1)

class DenseBlock(Module):
    def __init__(self, num_BotNeck, in_channels, growth_rate) -> None:
        super().__init__()
        self.layers = Sequential()
        for i in range(num_BotNeck):
            self.layers.append(Bottleneck(in_channels=in_channels+(i*growth_rate), growth_rate=growth_rate))

    def forward(self, x):
        return self.layers(x)

class Transition(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = Sequential(
            BatchNorm2d(in_channels),
            ReLU(),
            Conv2d(in_channels, out_channels, 1, bias=False),
            AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

class MyCNN(Module):
    def __init__(
            self,
            nblocks: list, 
            growth_rate: int, 
        ):
        super().__init__()
        in_channels = 2*growth_rate
        self.layers = Sequential(
            Conv2d(1, in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            BatchNorm2d(in_channels),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        for i, num_bottle_necks in enumerate(nblocks):
            self.layers.append(DenseBlock(num_bottle_necks, in_channels, growth_rate))
            in_channels += num_bottle_necks * growth_rate
            if i != len(nblocks) - 1:
                self.layers.append(Transition(in_channels, in_channels // 2))
                in_channels = in_channels // 2

        self.layers.append(BatchNorm2d(in_channels))
        self.layers.append(ReLU())
        self.layers.append(AvgPool2d(3))
        self.layers.append(Flatten())
        self.layers.append(Dropout(0.25))
        self.layers.append(Linear(in_channels, 20))

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        x = self.layers(input_images)
        return x

model_parameters={}
model_parameters['densenet121'] = [6,12,24,16]
model_parameters['densenet169'] = [6,12,32,32]
model_parameters['densenet201'] = [6,12,48,32] #!
model_parameters['densenet264'] = [6,12,64,48]

model = MyCNN(
        nblocks=model_parameters['densenet201'], 
        growth_rate=32
    )
# if __name__ == "__main__":
#     x = torch.rand((1,1,100,100))
#     #print(x)
#     print(model(x))