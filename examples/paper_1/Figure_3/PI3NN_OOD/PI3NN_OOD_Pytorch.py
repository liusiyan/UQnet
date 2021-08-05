import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.FloatTensor)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print('device', device)


bias_list = [10.0, 0.0]

def target_fun(x):

   y = x**3
   y = torch.sum(y, dim=1, keepdim=True)/10 + torch.randn(x.size(0), device=device).unsqueeze(1)
   return y

class UQ_Net_mean(nn.Module):

    def __init__(self, nd):
        super(UQ_Net_mean, self).__init__()
        self.fc1 = nn.Linear(nd, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def UQ_loss(self, x, output, ydata):
        loss = torch.mean((output[:, 0]-ydata[:, 0])**2)
        return loss


class UQ_Net_std(nn.Module):

    def __init__(self, nd, bias):
        super(UQ_Net_std, self).__init__()
        self.fc1 = nn.Linear(nd, 200)
        self.fc2 = nn.Linear(200, 1)
        self.fc2.bias = torch.nn.Parameter(torch.tensor([bias]))
        # self.fc2.bias = torch.nn.Parameter(torch.tensor([50.0]))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sqrt(torch.square(self.fc2(x)) + 1e-10)
        return x

    def UQ_loss(self, x, output, ydata):
        loss = torch.mean((output[:, 0] - ydata[:, 0])**2)
        return loss



for iii, bias in enumerate(bias_list):

    # Generate dataset
    Npar = 10
    Ntrain = 5000
    Nout = 1

    xtrain = (torch.randn(Ntrain, Npar)*1.0).to(device)
    ytrain = target_fun(xtrain).to(device)

    # normalize data
    x_mean = torch.mean(xtrain, axis=0)
    x_std = torch.std(xtrain, axis=0)
    xtrain_normal = (xtrain - x_mean)/x_std

    y_mean = torch.mean(ytrain, axis=0)
    y_std = torch.std(ytrain, axis=0)
    ytrain_normal = (ytrain - y_mean)/y_std


    Nvalid = 10000
    xvalid = (torch.randn(Nvalid, Npar)*5).to(device)
    yvalid = target_fun(xvalid).to(device)

    xvalid_normal = (xvalid - x_mean) / x_std
    yvalid_normal = (yvalid - y_mean) / y_std


    criterion = nn.MSELoss()

    # The network for estimating mean
    net = UQ_Net_mean(Npar).to(device)
    net.zero_grad()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    Max_iter = 3000

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes()

    for i in range(Max_iter):

        optimizer.zero_grad()

        output = net(xtrain_normal)
        loss = criterion(output, ytrain_normal)

        if i % 1000 == 0:
            print(i, loss)

        loss.backward()
        optimizer.step()

    # Generate difference data
    diff = (ytrain_normal - net(xtrain_normal)).detach()

    mask = diff > 0
    # print(mask.size())
    y_up_data = diff[diff > 0].unsqueeze(1)
    # x_up_data = xtrain_normal[diff>0].unsqueeze(1)
    x_up_data = xtrain_normal[mask[:, 0], :]#.unsqueeze(1)

    mask = diff < 0
    y_down_data = -1.0 * diff[diff < 0].unsqueeze(1)
    x_down_data = xtrain_normal[mask[:, 0], :]#.unsqueeze(1)


    net_up = UQ_Net_std(Npar, bias).to(device)
    net_up.zero_grad()
    optimizer = optim.SGD(net_up.parameters(), lr=0.01)

    for i in range(Max_iter):
        optimizer.zero_grad()
        output = net_up(x_up_data)
        loss = criterion(output, y_up_data)

        if torch.isnan(loss):
            print(output, y_up_data)
            exit()

        if i % 1000 == 0:
            print(i, loss)

        loss.backward()
        optimizer.step()

    net_down = UQ_Net_std(Npar, bias).to(device)
    net_down.zero_grad()
    optimizer = optim.SGD(net_down.parameters(), lr=0.01)

    for i in range(Max_iter):
        optimizer.zero_grad()
        output = net_down(x_down_data)
        # loss = net_up.UQ_loss(x_down_data, output, y_down_data)
        loss = criterion(output, y_down_data)

        if torch.isnan(loss):
            print(output, y_down_data)
            exit()

        if i % 1000 == 0:
            print(i, loss)

        loss.backward()
        optimizer.step()

    output = net(xvalid_normal)
    output_up = net_up(xvalid_normal)
    output_down = net_down(xvalid_normal)

    # err = torch.abs(output - yvalid_normal)
    # PI = output_up + output_down
    # PI = output_up - output_down

    # in_mask = torch.sqrt(torch.sum(xvalid**2.0, 1, keepdim=True))<4.0
    # out_mask = torch.sqrt(torch.sum(xvalid**2.0, 1, keepdim=True))>=4.0

    dist = torch.sqrt(torch.sum(xvalid**2.0, 1, keepdim=True))

    # plt.plot(dist[in_mask].detach().numpy(),PI[in_mask].detach().numpy() ,'r.')
    # plt.plot(dist[out_mask].detach().numpy(),PI[out_mask].detach().numpy() ,'b.')

    dist1 = torch.sqrt(torch.sum(xtrain**2.0, 1, keepdim=True))
    PI1 = net_up(xtrain_normal) + net_down(xtrain_normal)

    MPIW_array_train = PI1.detach().numpy()
    MPIW_train = np.mean(MPIW_array_train) + 1.8 * np.std(MPIW_array_train)

    MPIW_array = (output_up + output_down).detach().numpy()

    confidence_arr_test = [min(MPIW_train / test_width, 1.0) for test_width in MPIW_array]
    confidence_arr_train = [min(MPIW_train / train_width, 1.0) for train_width in MPIW_array_train]

    plt.plot(dist1.detach().numpy(), confidence_arr_train, 'r.')
    plt.plot(dist.detach().numpy(), confidence_arr_test, 'b.')

    ''' Save to file and plot the results '''
    confidence_arr_train = np.array(confidence_arr_train)
    confidence_arr_test = np.array(confidence_arr_test)

    PI3NN_OOD_train_np = np.hstack((dist1.reshape(-1, 1), confidence_arr_train.reshape(-1, 1)))
    PI3NN_OOD_test_np = np.hstack((dist.reshape(-1, 1), confidence_arr_test.reshape(-1, 1)))

    np.savetxt('PI3NN_OOD_train_np_bias_'+str(bias)+'.txt', PI3NN_OOD_train_np, delimiter=',')
    np.savetxt('PI3NN_OOD_test_np_bias_'+str(bias)+'.txt', PI3NN_OOD_test_np, delimiter=',')

    # plt.plot(dist1.detach().numpy(), PI1.detach().numpy(), 'r.')
    # plt.plot(dist.detach().numpy(), PI.detach().numpy(), 'b.')


    plt.xlabel('L2 distance to the mean of training data $\{x_i\}_{i=1}^N$')
    plt.ylabel('The Confidence Score')
    plt.legend(loc='lower left')

    # plt.ylim(0, 1.2)
    if iii == 0:
        plt.title('PI3NN with OOD detection')
    elif iii == 1:
        plt.title('PI3NN without OOD detection')
    plt.ylim(0, 1.2)
    plt.savefig('PI3NN_OOD_plot_bias_'+str(bias)+'_PyTorch.png')
    # plt.show()

