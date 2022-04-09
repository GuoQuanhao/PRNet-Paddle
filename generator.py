#PyTorch lib
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
#Tools lib


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)


#Set iteration time
ITERATION = 4

#Model

ITERATION = 4

#Model
class Generator_lstm(nn.Layer):
    def __init__(self, recurrent_iter=4):
        super(Generator_lstm, self).__init__()
        self.iteration = recurrent_iter

        self.conv0 = nn.Sequential(
            nn.Conv2D(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2D(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2D(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2D(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2D(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2D(32, 3, 3, 1, 1),
            )


    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        x = input
        h = paddle.zeros([batch_size, 32, row, col])
        c = paddle.zeros([batch_size, 32, row, col])

        x_list = []
        for i in range(self.iteration):
            x = paddle.concat((input, x), 1)
            x = self.conv0(x)
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = paddle.concat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * F.tanh(c)
            x = self.conv(h)
            x_list.append(x)

        return x, x_list

