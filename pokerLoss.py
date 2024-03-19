import torch.nn as nn
import torch

class fourLoss(nn.Module):
    def __init__(self, loss_type):
        super(fourLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, y, t):
        '''
        y: model generated output shape: (104) i.e. [card1 + card2] where each element is one_hot values
        t: target value shape: (104) i.e. [card1 + card2] where each element is one_hot values
        '''
        ce = nn.BCELoss()

        yc1, yc2 = y[:, :len(y)//2], y[:, len(y)//2:]
        tc1, tc2 = t[:, :len(y)//2], t[:, len(y)//2:]

        print(yc1.shape)
        print(yc2.shape)
        print(tc1.shape)
        print(tc2.shape)
        
        loss1 = (ce(yc1, tc1) +  ce(yc2, tc2))/2
        loss2 = (ce(yc2, tc1) +  ce(yc1, tc2))/2

        return min(loss1, loss2)


        # '''
        # y: model generated output shape: (34) i.e. [rank1 + suit1 + rank2 + suit2] where each element is a one-hot encoded representation of a card
        # t: target value shape: (34) i.e. [rank1 + suit1 + rank2 + suit2] where each element is a one-hot encoded representation of a card
        # '''
        # if self.loss_type == "loss 1":

        #     y1_r, y1_s = y[0:13], y[13:17]
        #     y2_r, y2_s = y[17:30], y[30:34]
        #     t1_r, t1_s = t[0:13], t[13:17]
        #     t2_r, t2_s = t[17:30], t[30:34]

        #     ce = nn.CrossEntropyLoss()
            
        #     loss1 = ce(y1_r, t1_r) + ce(y1_s, t1_s) + ce(y2_r, t2_r) + ce(y2_s, t2_s)
        #     loss2 = ce(y2_r, t1_r) + ce(y2_s, t1_s) + ce(y1_r, t2_r) + ce(y1_s, t2_s)

        #     return min(loss1/4, loss2/4)
    
def accuracy(y, t, k=10):
    yc1, yc2 = torch.topk(y[:52], k)[1], torch.topk(y[52:], k)[1]
    tc1, tc2 = torch.argmax(t[:52]), torch.argmax(t[52:])
    accuracy_1 = int((tc1 in yc1) and (tc2 in yc2))
    accuracy_2 = int((tc1 in yc2) and (tc2 in yc1))
    return max(accuracy_1, accuracy_2)

