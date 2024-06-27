import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats


# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.FloatTensor)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print('device', device)


# ==== fix random seed for reproducibility =====
np.random.seed(12345)
torch.manual_seed(12345)


#------------------------------------------------------
#  ------- Define the three NNs ------------
#------------------------------------------------------
# ==== define the function =====
def target_fun(x):

    y = x**3
    y = (torch.sum(y, dim=1, keepdim=True)/10.0) + 1.0*torch.randn(x.size(0), device=device).unsqueeze(1)
    return y

# ==== define the mean network =====
class UQ_Net_mean(nn.Module):

    def __init__(self, nd):
        super(UQ_Net_mean, self).__init__()
        self.fc1 = nn.Linear(nd, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    def UQ_loss(self, x, output, ydata):
        loss = torch.mean((output[:, 0]-ydata[:, 0])**2)
        return loss

# ==== define the upper and lower bound network =====
class UQ_Net_std(nn.Module):

    def __init__(self, nd):
        super(UQ_Net_std, self).__init__()
        self.fc1 = nn.Linear(nd, 200)
        self.fc2 = nn.Linear(200, 1)
        self.fc2.bias = torch.nn.Parameter(torch.tensor([25.0])) ## assign a large bias value of the output layer for OOD identification

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.sqrt(torch.square(self.fc2(x)) + 0.1)
        return x

    def UQ_loss(self, x, output, ydata):
        loss = torch.mean((output[:, 0] - ydata[:, 0])**2)
        return loss



#------------------------------------------------------
#  ------- Generate the data  ------------
#------------------------------------------------------
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


Nvalid = 1000
xvalid = (torch.randn(Nvalid, Npar)+2.0).to(device)
# xvalid[:,1] = xvalid[:,1] + 4.0
yvalid = target_fun(xvalid).to(device)

xvalid_normal = (xvalid - x_mean) / x_std
yvalid_normal = (yvalid - y_mean) / y_std



#------------------------------------------------------
#  ------- Train the three NNs  ------------
#------------------------------------------------------
criterion = nn.MSELoss()

# ====== Train the mean network for estimating mean
net = UQ_Net_mean(Npar).to(device)
net.zero_grad()
optimizer = optim.SGD(net.parameters(), lr=0.01)

Max_iter = 3000

fig = plt.figure(figsize=(6, 6))
# ax = fig.add_axes()

for i in range(Max_iter):

    optimizer.zero_grad()

    output = net(xtrain_normal)
    loss = criterion(output, ytrain_normal)

    if i % 1000 == 0:
        print(i, loss)

    loss.backward()
    optimizer.step()


## ==== Calculate the difference to get training data of U and V network
diff = (ytrain_normal - net(xtrain_normal)).detach()

mask = diff > 0
# print(mask.size())
y_up_data = diff[diff > 0].unsqueeze(1)
# x_up_data = xtrain_normal[diff>0].unsqueeze(1)
x_up_data = xtrain_normal[mask[:, 0], :]#.unsqueeze(1)

mask = diff < 0
y_down_data = -1.0 * diff[diff < 0].unsqueeze(1)
x_down_data = xtrain_normal[mask[:, 0], :]#.unsqueeze(1)


# ====== Train the U and V network for estimating upper and lower bound ====
net_up = UQ_Net_std(Npar).to(device)
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

net_down = UQ_Net_std(Npar).to(device)
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


#--------------------------------------------------------------------
#  ------- Root-finding to determine alpha and beta  ------------
#--------------------------------------------------------------------
quantile = 0.9
num_outlier = int(Ntrain * (1-quantile)/2)

output = net(xtrain_normal)
output_up = net_up(xtrain_normal)
output_down = net_down(xtrain_normal)

##===== find alpha =======
c_up0 = 0.0
c_up1 = 200.0

f0 = (ytrain_normal >= output + c_up0 * output_up).sum() - num_outlier
f1 = (ytrain_normal >= output + c_up1 * output_up).sum() - num_outlier

n_iter = 1000
iter = 0
while iter <= n_iter and f0*f1<0:  ##f0 != 0 and f1 != 0:

    c_up2 = (c_up0 + c_up1)/2.0
    f2 = (ytrain_normal >= output + c_up2 * output_up).sum() - num_outlier

    if f2 == 0: 
        break
    elif f2 > 0:
        c_up0 = c_up2
        f0 = f2
    else:
        c_up1 = c_up2
        f1 = f2
    # print('{}, f0: {}, f1: {}, f2: {}'.format(iter, f0, f1, f2))

c_up = c_up2

##===== find beta =======
c_down0 = 0.0
c_down1 = 200.0

f0 = (ytrain_normal <= output - c_down0 * output_down).sum() - num_outlier
f1 = (ytrain_normal <= output - c_down1 * output_down).sum() - num_outlier

n_iter = 1000
iter = 0
while iter <= n_iter and f0*f1<0:  ##f0 != 0 and f1 != 0:

    c_down2 = (c_down0 + c_down1)/2.0
    f2 = (ytrain_normal <= output - c_down2 * output_down).sum() - num_outlier

    if f2 == 0: 
        break
    elif f2 > 0:
        c_down0 = c_down2
        f0 = f2
    else:
        c_down1 = c_down2
        f1 = f2
    # print('{}, f0: {}, f1: {}, f2: {}'.format(iter, f0, f1, f2))
c_down = c_down2

print('optimal alpha and beta: ', c_up, c_down)


#--------------------------------------------------------------------
#  ------- Save and analysis results  ------------
#--------------------------------------------------------------------
##--- 1. run PIVEN, QD, SQR, DER for this 10-Cubic function;
##--- 2. for each method, save two files 'PIW_train.dat', 'PIW_test.dat'
##--- 3. Calculate confidence score for the flight delay data.


output = net(xvalid_normal)
output_up = net_up(xvalid_normal)
output_down = net_down(xvalid_normal)


PI1 = net_up(xtrain_normal) * c_up + net_down(xtrain_normal) * c_down

MPIW_array_train = (PI1 * y_std).detach().numpy()
MPIW_array_test = (output_up * c_up * y_std + c_down * output_down * y_std).detach().numpy()

MPIW_array_train = MPIW_array_train[~np.isnan(MPIW_array_train)]
MPIW_array_test = MPIW_array_test[~np.isnan(MPIW_array_test)]

print(np.shape(MPIW_array_train), np.shape(MPIW_array_test))
np.savetxt('PI3NN_OOD_MPIW_train.dat', MPIW_array_train)
np.savetxt('PI3NN_OOD_MPIW_test.dat', MPIW_array_test)


kde_train = stats.gaussian_kde(MPIW_array_train)
kde_test = stats.gaussian_kde(MPIW_array_test)

x1 = np.linspace(MPIW_array_train.min(), MPIW_array_train.max(), 100)
p1 = kde_train(x1)

x2 = np.linspace(MPIW_array_test.min(), MPIW_array_test.max(), 100)
p2 = kde_test(x2)

plt.plot(x1,p1, label='train')
plt.plot(x2,p2, label='test')
plt.legend()
# plt.savefig('PI3NN_cubic10D_bias.png')
# plt.show()

# print('P1 (train) mean: {}, STD: {}'.format(np.mean(p1), np.std(p1)))
# print('P2 (test) mean: {}, STD: {}'.format(np.mean(p2), np.std(p2)))
