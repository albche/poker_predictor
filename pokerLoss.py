import torch.nn as nn
import torch

class fourLoss(nn.Module):
    def __init__(self, loss_type):
        super(fourLoss, self).__init()

        self.loss_type = loss_type
    def forward(y, t):
        '''
        y: model generated output shape: (34) i.e. [rank1 + suit1 + rank2 + suit2] where each element is a one-hot encoded representation of a card
        t: target value shape: (34) i.e. [rank1 + suit1 + rank2 + suit2] where each element is a one-hot encoded representation of a card
        '''
        if self.loss_type == "loss 1":

            y1_r, y1_s = y[0:13], y[13:17]
            y2_r, y2_s = y[17:30], y[30:34]
            t1_r, t1_s = t[0:13], t[13:17]
            t2_r, t2_s = t[17:30], t[30:34]

            ce = nn.CrossEntropyLoss()
            
            loss1 = ce(y1_r, t1_r) + ce(y1_s, t1_s) + ce(y2_r, t2_r) + ce(y2_s, t2_s)
            loss2 = ce(y2_r, t1_r) + ce(y2_s, t1_s) + ce(y1_r, t2_r) + ce(y1_s, t2_s)

            return min(loss1, loss2)
    
#testing

# s1 = [0, 0.5, 0.5, 0]
# s2 = [0, 0, 1, 0]
# s_T1 = [0, 1, 0, 0]
# s_T2 = [0, 0, 1, 0]

# r1 = [0, 0, 0, 0.8, 0, 0.2, 0, 0, 0, 0, 0, 0, 0]
# r2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
# r_T1 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# r_T2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

# y = torch.as_tensor(r1+s1+r2+s2).type(torch.float32)
# t = torch.as_tensor(r_T1+s_T1+r_T2+s_T2).type(torch.float32)

# ce = torch.nn.CrossEntropyLoss()
# loss = ce(t, s_T1)
# print(loss)
# loss.backward()

# loss = fourLoss(y, t)
# print(loss)
# print(loss.item())
# loss.backward()

