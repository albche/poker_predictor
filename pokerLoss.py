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
        y = y.type(torch.float)
        t = t.type(torch.long)

        ce = nn.NLLLoss()

        yc1, yc2 = y[:, :52], y[:, 52:]
        tc1, tc2 = t[:, :52], t[:, 52:]
        
        loss1, loss2 = 0, 0
        for b in range(len(yc1)):
            loss1 += (ce(yc1[b], tc1[b]) +  ce(yc2[b], tc2[b]))/2
            loss2 += (ce(yc2[b], tc1[b]) +  ce(yc1[b], tc2[b]))/2
        # loss1 = (ce(yc1, tc1) +  ce(yc2, tc2))/2
        # loss2 = (ce(yc2, tc1) +  ce(yc1, tc2))/2

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
    y1, y2 = y[:, :52], y[:, 52:]
    t1, t2 = t[:, :52], t[:, 52:]

    accuracy_1, accuracy_2 = 0, 0
    for b in range(len(y1)):
        yc1, yc2 = torch.topk(y1[b], k)[1], torch.topk(y2[b], k)[1]
        tc1, tc2 = torch.argmax(t1[b]), torch.argmax(t2[b])
        accuracy_1 += int((tc1 in yc1) and (tc2 in yc2))
        accuracy_2 += int((tc1 in yc2) and (tc2 in yc1))
    return max(accuracy_1/len(y), accuracy_2/len(y))

    # y1, y2 = y[:, :52], y[:, 52:]
    # t1, t2 = t[:, :52], t[:, 52:]

    # probabilities = {}
    # for b in range(len(y1)):
    #     yc1, yc2 = torch.topk(y1[b], k), torch.topk(y2[b], k)
    #     for i in range(k):
    #         probabilities[(yc1[1][i].item(), yc2[1][i].item())] = max(yc1[0][i].item() * yc2[0][i].item(), probabilities[(yc1[1][i].item(), yc2[1][i].item())] if (yc1[1][i].item(), yc2[1][i].item()) in probabilities else 0)
    
    # sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    
    # tc1, tc2 = torch.argmax(t1[b]).item(), torch.argmax(t2[b]).item()
    # for prob in sorted_probs[:]:
    #     if prob[0] == (tc1, tc2) or prob[0] == (tc2, tc1):
    #         return 1

    # return 0