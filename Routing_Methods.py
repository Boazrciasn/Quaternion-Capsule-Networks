import torch
import torch.nn.functional as F
import torch.nn as nn
import math
eps = 1e-7

class EMRouting(nn.Module):

    def __init__(self, iterations=3 ):
        super(EMRouting, self).__init__()
        self.iterations = iterations
        self.final_lambda = 0.01
        self.register_buffer("mathpilog", torch.log(torch.FloatTensor([2 * math.pi])))

    def forward(self, V, a, Beta_u, Beta_a, R, outSize):


        for i in range(self.iterations):
            Lambda = self.final_lambda * (1 - 0.95 ** (i + 1))

            # M - Step:
            R = (R * a).unsqueeze(7)
            mu_share = (R * V).sum(dim=[3, 4, 5], keepdim=True)
            mu_denom = R.sum(dim=[3, 4, 5], keepdim=True)
            mu = mu_share / (mu_denom + eps)

            V_mu_sqr = (V - mu) ** 2
            sigma_share = (R * V_mu_sqr).sum(dim=[3, 4, 5], keepdim=True)
            sigma_sqr = sigma_share / (mu_denom + eps)

            cost_h = (Beta_u + 0.5*torch.log(sigma_sqr)) * mu_denom
            a_out = torch.sigmoid(F.normalize(Lambda * (Beta_a - cost_h.sum(dim=7)), dim=6))


            #E-Step:
            log_p1 = -0.5 * ((self.mathpilog + torch.log(sigma_sqr)).sum(dim=7)) + eps
            log_p2 = -(V_mu_sqr / ((2 * sigma_sqr) + eps)).sum(dim=7)
            log_p = log_p1 + log_p2
            R_ = torch.log(a_out) + log_p
            R = torch.softmax(R_, dim=6)

        mu = mu.squeeze()
        sigma_sqr = sigma_sqr.squeeze()
        return mu.view(outSize), \
               a_out.squeeze(), \
               sigma_sqr.view(outSize)



class LSTMRouting(nn.Module):

    def __init__(self):
        super(LSTMRouting, self).__init__()
        torch.nn.LSTM

