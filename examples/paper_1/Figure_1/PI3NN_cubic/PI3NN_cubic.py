import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.FloatTensor)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('device',device)

device = 'cpu'



def target_fun(x):

    noise = torch.randn(x.size(),device=device)

    # print(x.size(0))

    for i in range(x.size(0)):
        if(noise[i]>0): 
            noise[i] = noise[i] * 10.0
        else:
            noise[i] = noise[i] * 2.0

    y = x**3 + noise

    return y


class UQ_Net_mean(nn.Module):

    def __init__(self):
        super(UQ_Net_mean, self).__init__()
        self.fc1 = nn.Linear(1, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def UQ_loss(self, x, output, ydata):

        loss = torch.mean((output[:,0]-ydata[:,0])**2)

        return loss




class UQ_Net_std(nn.Module):

    def __init__(self):
        super(UQ_Net_std, self).__init__()
        self.fc1 = nn.Linear(1, 200)
        self.fc2 = nn.Linear(200, 1)
        self.fc2.bias = torch.nn.Parameter(torch.tensor([15.0]))


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sqrt(torch.square(self.fc2(x)) + 1e-10)

        return x

    def UQ_loss(self, x, output, ydata):

        loss = torch.mean((output[:,0] - ydata[:,0])**2) 

        return loss



# Target percentile
quantile = 0.9

# Generate dataset
Npar = 1
Ntrain = 1000
Nout = 1

xtrain = (torch.rand(Ntrain,Npar)*8.0-4.0).to(device)
ytrain = target_fun(xtrain).to(device)

x_mean = torch.mean(xtrain,axis=0)
x_std = torch.std(xtrain,axis=0)
xtrain_normal = (xtrain - x_mean)/x_std

y_mean = torch.mean(ytrain,axis=0)
y_std = torch.std(ytrain,axis=0)
ytrain_normal = (ytrain - y_mean)/y_std


Nvalid = 1000
xvalid = (torch.rand(Nvalid,Npar)*14.0-7.0).to(device)
yvalid = target_fun(xvalid).to(device)

xvalid_normal = (xvalid - x_mean) / x_std
yvalid_normal = (yvalid - y_mean) / y_std


criterion = nn.MSELoss()

# The network for estimating mean
net = UQ_Net_mean().to(device)
net.zero_grad()
optimizer = optim.SGD(net.parameters(), lr=0.01)


Max_iter = 4000

for i in range(Max_iter):

    optimizer.zero_grad()

    output = net(xtrain_normal)
    loss = criterion(output, ytrain_normal)
    # loss = net.UQ_loss(xtrain_normal, output, ytrain_normal)

    if i%1000==0: print(i,loss)
    
    loss.backward()
    
    optimizer.step()


# Generate difference data
diff = (ytrain_normal - net(xtrain_normal)).detach()

y_up_data = diff[diff>0].unsqueeze(1)
x_up_data = xtrain_normal[diff>0].unsqueeze(1)

y_down_data = -1.0 * diff[diff<0].unsqueeze(1)
x_down_data = xtrain_normal[diff<0].unsqueeze(1)




net_up = UQ_Net_std().to(device)
net_up.zero_grad()
optimizer = optim.SGD(net_up.parameters(), lr=0.01)

for i in range(Max_iter):

    optimizer.zero_grad()

    output = net_up(x_up_data)

    loss = criterion(output, y_up_data)
    # loss = net_up.UQ_loss(x_up_data, output, y_up_data)

    if torch.isnan(loss): 
        print(output, y_up_data)
        exit()

    if i%1000==0: print(i,loss)
    
    loss.backward()
    
    optimizer.step()



net_down = UQ_Net_std().to(device)
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

    if i%1000==0: print(i,loss)

    loss.backward()

    optimizer.step()



#------------------------------------------------------
# Determine how to move the upper and lower bounds
num_outlier = int(Ntrain * (1-quantile)/2)

output = net(xtrain_normal)
output_up = net_up(xtrain_normal)
output_down = net_down(xtrain_normal)

c_up0 = 0.0
c_up1 = 10.0

f0 = (ytrain_normal >= output + c_up0 * output_up).sum() - num_outlier
f1 = (ytrain_normal >= output + c_up1 * output_up).sum() - num_outlier

n_iter = 1000
iter = 0
while iter <= n_iter and f0 != 0 and f1 != 0:

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
    print('{}, f0: {}, f1: {}, f2: {}'.format(iter, f0, f1, f2))

c_up = c_up2


c_down0 = 0.0
c_down1 = 10.0

f0 = (ytrain_normal <= output - c_down0 * output_down).sum() - num_outlier
f1 = (ytrain_normal <= output - c_down1 * output_down).sum() - num_outlier

n_iter = 1000
iter = 0
while iter <= n_iter and f0 != 0 and f1 != 0:

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
    print('{}, f0: {}, f1: {}, f2: {}'.format(iter, f0, f1, f2))
c_down = c_down2

#------------------------------------------------------

# Plotting
x_plot = (torch.linspace(-7,7,steps=500).unsqueeze(1) - x_mean)/x_std
y_plot_mean = net(x_plot)
y_plot_up = net_up(x_plot)
y_plot_down = net_down(x_plot)

plt.figure(figsize=(5, 4), dpi=200)
plt.plot(xtrain.cpu().numpy(), ytrain.cpu().numpy(),'k.',markersize=2, alpha=0.2, zorder=0, label='Data')
plt.plot((x_plot*x_std+x_mean).cpu().numpy(), (y_plot_mean*y_std+y_mean).cpu().detach().numpy(), 'r', linewidth=2,label='Prediction')
# plt.plot((x_plot*x_std+x_mean).cpu().numpy(), ((y_plot_mean+c_up  *y_plot_up  )*y_std+y_mean).cpu().detach().numpy(), 'g--',linewidth=2)
# plt.plot((x_plot*x_std+x_mean).cpu().numpy(), ((y_plot_mean-c_down*y_plot_down)*y_std+y_mean).cpu().detach().numpy(), 'y--',linewidth=2)
plt.plot((x_plot*x_std+x_mean).cpu().numpy(), ((x_plot*x_std+x_mean)**3).cpu().numpy(),'b--',linewidth=2,label='Groud Truth')
plt.fill_between((x_plot*x_std+x_mean).cpu().numpy().squeeze(), 
            ((y_plot_mean-c_down*y_plot_down)*y_std+y_mean).cpu().detach().numpy().squeeze(),
            ((y_plot_mean+c_up  *y_plot_up  )*y_std+y_mean).cpu().detach().numpy().squeeze(),
            alpha=0.4,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Uncertainty")
plt.plot([-4, -4], [-150, 150], 'k--', alpha=0.4, linewidth=0.5, zorder=0)
plt.plot([+4, +4], [-150, 150], 'k--', alpha=0.4, linewidth=0.5, zorder=0)
plt.fill_between([-7, -4],[-150, -150], [150, 150],edgecolor=None,facecolor='k',alpha=0.1, zorder=0,label='No data')
plt.fill_between([4, 7],[-150, -150], [150, 150],edgecolor=None,facecolor='k',alpha=0.1, zorder=0)
plt.ylim(-150, 150)
plt.xlim(-7, 7)
plt.legend(loc="lower right",ncol=2)
plt.savefig('Demo.png',pad_inches=0.1)
plt.show()

x = (x_plot*x_std+x_mean).cpu().numpy()
pred = (y_plot_mean*y_std+y_mean).cpu().detach().numpy()
lower = ((y_plot_mean-c_down*y_plot_down)*y_std+y_mean).cpu().detach().numpy()
upper = ((y_plot_mean+c_up  *y_plot_up  )*y_std+y_mean).cpu().detach().numpy()
print(np.shape(x),lower.shape,upper.shape)

np.savetxt('OurMethod_cubic.dat', np.hstack((x, lower, upper, pred)))

xtrain = xtrain.cpu().numpy()
ytrain = ytrain.cpu().numpy()
print(xtrain.shape, ytrain.shape)
np.savetxt('TrainData.dat', np.hstack((xtrain,ytrain)))




# output = net(xvalid_normal)
# output_up = net_up(xvalid_normal)
# output_down = net_down(xvalid_normal)

# plt.plot((xvalid_normal*x_std+x_mean).cpu().numpy(), (output*y_std+y_mean).cpu().detach().numpy(), 'r.')
# plt.plot((xvalid_normal*x_std+x_mean).cpu().numpy(), (output*y_std+y_mean+c_up  *output_up  *y_std+y_mean).cpu().detach().numpy(), 'g+')
# plt.plot((xvalid_normal*x_std+x_mean).cpu().numpy(), (output*y_std+y_mean-c_down*output_down*y_std+y_mean).cpu().detach().numpy(), 'y+')
# # plt.plot(xtrain.cpu().numpy(), ytrain.cpu().numpy(),'b.')
# # plt.plot((xvalid_normal*x_std+x_mean).cpu().numpy(), (yvalid_normal*y_std+y_mean).cpu().numpy(),'k.')
# plt.gca().set_ylim(-150, 150)
# plt.show()

