import math
import torch
import torch.nn as nn
import framework.config as config
import torch.nn.functional as F


def move_data_to_gpu(x, cuda, half=False):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")
    if cuda:
        x = x.cuda()
        if half:
            x = x.half()
    return x



def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)



class TinyCNN(nn.Module):
    def __init__(self, event_class, batchnormal):

        super(TinyCNN, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)

        out_channels = 32
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        out_channels = 64
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=out_channels,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        out_channels = 128
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=out_channels,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        units = 512
        self.fc1 = nn.Linear(128*16, units, bias=True)
        self.fc_final = nn.Linear(units, 1, bias=True)

        self.fc1_event = nn.Linear(128*16, units, bias=True)
        self.fc_final_event = nn.Linear(units, event_class, bias=True)

        self.my_init_weight()

    def my_init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_final)
        init_layer(self.fc1_event)
        init_layer(self.fc_final_event)

    def forward(self, input):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (16, 481, 64)

        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        # print(x.size())  # torch.Size([64, 1, 480, 64])
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(3, 3))
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())  # torch.Size([64, 32, 95, 12])

        x = F.relu_(self.bn2(self.conv2(x)))
        # print(x.size())  # torch.Size([64, 64, 91, 8])
        x = F.max_pool2d(x, kernel_size=(3, 3))
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())  # torch.Size([64, 64, 18, 1])

        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(3, 3))
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size()) # torch.Size([64, 128, 16, 1])

        x = x.view(x.size()[0], -1)
        # print(x.size())  # torch.Size([64, 1152])

        x = F.dropout(x, p=0.3, training=self.training)

        x_embed = F.relu_(self.fc1(x))
        x_rate_linear = self.fc_final(x_embed)

        x_event = F.relu_(self.fc1_event(x))
        x_event_linear = self.fc_final_event(x_event)

        return x_rate_linear, x_event_linear



def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')

        return x



class Cnn6_source_rate(nn.Module):
    def __init__(self, event_class, batchnormal=False):

        super(Cnn6_source_rate, self).__init__()

        self.batchnormal=batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)

        self.fc_final_event = nn.Linear(2048, event_class, bias=True)

        units = 512
        self.fc_rate_512 = nn.Linear(2048, units, bias=True)
        units2 = 64
        self.fc_rate_64 = nn.Linear(units, units2, bias=True)
        self.fc_rate_final = nn.Linear(units2, 1, bias=True)

        self.init_weight()

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_final_event)

        init_layer(self.fc_rate_512)
        init_layer(self.fc_rate_64)
        init_layer(self.fc_rate_final)

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        # print(x.size())  # torch.Size([64, 1, 320, 64])
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x.size())  torch.Size([64, 64, 160, 32])

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x.size()) torch.Size([64, 128, 80, 16])

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x_scene.size())  torch.Size([64, 256, 40, 8])

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x_scene.size())  torch.Size([64, 512, 20, 4])

        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x_scene.size())  torch.Size([64, 1024, 10, 2])

        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x_scene.size())  torch.Size([64, 2048, 10, 2])

        x = torch.mean(x, dim=3)
        # print(x_scene.size())  torch.Size([64, 2048, 10])

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        # print(x_scene.size()) # torch.Size([64, 2048])

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))

        event = self.fc_final_event(x)

        x = F.relu_(self.fc_rate_512(x))
        x = F.relu_(self.fc_rate_64(x))
        rate = F.relu_(self.fc_rate_final(x))

        return rate, event



class Cnn6_source_rate_additional_loss(nn.Module):
    def __init__(self, event_class, batchnormal=False):

        super(Cnn6_source_rate_additional_loss, self).__init__()

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)

        self.fc_final_event = nn.Linear(2048, event_class, bias=True)

        self.fc_rate_event_256 = nn.Linear(event_class, 256, bias=True)
        self.fc_rate_final = nn.Linear(256, 1, bias=True)

        self.init_weight()

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_final_event)

        init_layer(self.fc_rate_event_256)
        init_layer(self.fc_rate_final)

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        # print(x.size())  # torch.Size([64, 1, 320, 64])
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x.size())  torch.Size([64, 64, 160, 32])

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x.size()) torch.Size([64, 128, 80, 16])

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x_scene.size())  torch.Size([64, 256, 40, 8])

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x_scene.size())  torch.Size([64, 512, 20, 4])

        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x_scene.size())  torch.Size([64, 1024, 10, 2])

        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x_scene.size())  torch.Size([64, 2048, 10, 2])

        x = torch.mean(x, dim=3)
        # print(x_scene.size())  torch.Size([64, 2048, 10])

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        # print(x_scene.size()) # torch.Size([64, 2048])

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))

        event = self.fc_final_event(x)

        x = F.relu_(self.fc_rate_event_256(event))
        rate = self.fc_rate_final(x)

        return rate, event



