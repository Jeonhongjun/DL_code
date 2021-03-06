import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, x, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):  # torch.chunk 는 n 으로 column 단위로 spilt
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for i in range(future):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])

    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)

    # begin to train
    for i in range(15):
        print('step: ', i)

        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            loss.backward()

            return loss

        optimizer.step(closure)

        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            print(pred.shape)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()  # .detach() 현재의 계산 기록으로 부터 분리

        # draw the result
            plt.figure(figsize=(30, 10))
            plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
            plt.xlabel('x', fontsize=20)
            plt.ylabel('y', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            def draw(yi, color):
                plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
                plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)

            draw(y[0], 'r')
            draw(y[1], 'g')
            draw(y[2], 'b')
            plt.savefig('predict%d.pdf'%i)
            plt.close()
