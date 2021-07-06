import torch 
import torch.nn as nn 
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.deterministic = True

device = torch.device('cuda')

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.l = nn.Linear(10, 10)

    def forward(self, x):
        return self.l(x)

    

def test():

    # net = nn.Linear(10,10)
    net = Net()
    net = nn.parallel.DataParallel(net)
    net = net.to(device)

    criterion = nn.MSELoss()
    optimzer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    for i in range(100):

        data = torch.randn(32,10)
        gt = torch.randn(32,10)

        data = data.to(device)
        gt = gt.to(device)

        print('1')
        pred = net(data)
        print('2')
        loss = criterion(pred , gt)
        print('3')
        optimzer.zero_grad()
        print('4')
        loss.backward()
        print('5')
        optimzer.step()

        print('{} | loss {}'.format(i, loss))


if __name__ == '__main__':
    test()
