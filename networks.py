#PyTorch lib
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
#Tools lib
 

class PReNet(nn.Layer):
    def __init__(self, recurrent_iter=6):
        super(PReNet, self).__init__()
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
        batch_size, row, col = input.shape[0], input.shape[2], input.shape[3]

        x = input
        h = paddle.zeros([batch_size, 32, row, col])
        c = paddle.zeros([batch_size, 32, row, col])

        x_list = []
        for i in range(self.iteration):
            x = paddle.concat((input, x), 1)
            x = self.conv0(x)

            x = paddle.concat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * paddle.tanh(c)

            x = h
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
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list


class PReNet_LSTM(nn.Layer):
    def __init__(self, recurrent_iter=6):
        super(PReNet_LSTM, self).__init__()
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
        batch_size, row, col = input.shape[0], input.shape[2], input.shape[3]

        x = input
        h = paddle.zeros([batch_size, 32, row, col])
        c = paddle.zeros([batch_size, 32, row, col])

        x_list = []
        for i in range(self.iteration):
            x = paddle.concat((input, x), 1)
            x = self.conv0(x)

            x = paddle.concat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * paddle.tanh(c)

            x = h
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
            x = self.conv(x)

            x_list.append(x)

        return x, x_list


class PReNet_GRU(nn.Layer):
    def __init__(self, recurrent_iter=6):
        super(PReNet_GRU, self).__init__()
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
        self.conv_z = nn.Sequential(
            nn.Conv2D(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_b = nn.Sequential(
            nn.Conv2D(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2D(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        # self.conv_o = nn.Sequential(
        #     nn.Conv2D(32 + 32, 32, 3, 1, 1),
        #     nn.Sigmoid()
        #     )
        self.conv = nn.Sequential(
            nn.Conv2D(32, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.shape[0], input.shape[2], input.shape[3]

        x = input
        h = paddle.zeros([batch_size, 32, row, col])

        x_list = []
        for i in range(self.iteration):
            x = paddle.concat((input, x), 1)
            x = self.conv0(x)

            x1 = paddle.concat((x, h), 1)
            z = self.conv_z(x1)
            b = self.conv_b(x1)
            s = b * h
            s = paddle.concat((s, x), 1)
            g = self.conv_g(s)
            h = (1 - z) * h + z * g

            x = h
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

            x = self.conv(x)
            x_list.append(x)

        return x, x_list


class PReNet_x(nn.Layer):
    def __init__(self, recurrent_iter=6):
        super(PReNet_x, self).__init__()
        self.iteration = recurrent_iter

        self.conv0 = nn.Sequential(
            nn.Conv2D(3, 32, 3, 1, 1),
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
        batch_size, row, col = input.shape[0], input.shape[2], input.shape[3]

        x = input
        h = paddle.zeros([batch_size, 32, row, col])
        c = paddle.zeros([batch_size, 32, row, col])

        x_list = []
        for i in range(self.iteration):
            #x = paddle.concat((input, x), 1)
            x = self.conv0(x)

            x = paddle.concat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * paddle.tanh(c)

            x = h
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

            x = self.conv(x)
            x_list.append(x)

        return x, x_list


class PReNet_r(nn.Layer):
    def __init__(self, recurrent_iter=6):
        super(PReNet_r, self).__init__()
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
        batch_size, row, col = input.shape[0], input.shape[2], input.shape[3]
        x = input
        h = paddle.zeros([batch_size, 32, row, col])
        c = paddle.zeros([batch_size, 32, row, col])

        x_list = []
        for i in range(self.iteration):
            x = paddle.concat((input, x), 1)
            x = self.conv0(x)

            x = paddle.concat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * paddle.tanh(c)

            x = h
            for j in range(5):
                resx = x
                x = F.relu(self.res_conv1(x) + resx)

            x = self.conv(x)
            x = input + x
            x_list.append(x)

        return x, x_list


## PRN
class PRN(nn.Layer):
    def __init__(self, recurrent_iter=6):
        super(PRN, self).__init__()
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
        self.conv = nn.Sequential(
            nn.Conv2D(32, 3, 3, 1, 1),
        )

    def forward(self, input):

        x = input

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
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list


class PRN_r(nn.Layer):
    def __init__(self, recurrent_iter=6):
        super(PRN_r, self).__init__()
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

        self.conv = nn.Sequential(
            nn.Conv2D(32, 3, 3, 1, 1),
            )

    def forward(self, input):

        x = input

        x_list = []
        for i in range(self.iteration):
            x = paddle.concat((input, x), 1)
            x = self.conv0(x)

            for j in range(5):
                resx = x
                x = F.relu(self.res_conv1(x) + resx)

            x = self.conv(x)
            x = input + x
            x_list.append(x)

        return x, x_list
