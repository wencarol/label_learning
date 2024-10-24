from torch import nn
import torch
import torch.nn.functional as F

def one_hot(y, num_classes: int):
    if len(y.size()) == 1:
        return nn.functional.one_hot(y, num_classes=num_classes)
    elif y.size()[1] == num_classes:
        return y
    else:
        raise ValueError("Please check the dimension of labels!")

def kl_div_loss(output):
    prob_output = torch.sigmoid(output)
    kl_div = - torch.log(prob_output)
    return kl_div

def js_div_loss(output, lam):
    prob_output = torch.sigmoid(output).clamp(min=1e-7, max=1-1e-7)
    prob_mean = (lam + prob_output * (1 - lam)).clamp(min=1e-7, max=1-1e-7)
    kl_div1 = - torch.log(prob_mean)
    kl_div2 = prob_output * (torch.log(prob_output) - torch.log(prob_mean))
    return lam * kl_div1 + (1 - lam) * kl_div2


loss_func = {'kl': kl_div_loss, 'js': js_div_loss}

class Loss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, loss_type, pi_1, pi_2, device, lam: float = 0.5,
                 gamma: float = 1, beta: float = 0, nnPU: bool = True,
                 num_classes: int = 10, reduction: str = 'mean'):
        super(Loss,self).__init__()
        if not (0 < pi_1 < 1 and 0 < pi_2 < 1) :
            raise ValueError("The class prior should be in (0, 1)")
        if loss_type not in loss_func.keys():
            raise NotImplementedError("Loss function %s is not implemented."%(loss_type))
        self.loss = loss_func[loss_type]
        self.device = device
        self.pi_1 = pi_1
        self.pi_2 = pi_2
        self.gamma = gamma
        self.lam_distribution = torch.distributions.beta.Beta(torch.tensor([lam], device=device), 
                                                              torch.tensor([lam], device=device))
        self.beta = beta
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = 0
        self.num_classes = num_classes
        self.min_count = torch.tensor(1., device=self.device)
        self.reduction = reduction
        
    
    def forward(self, outputs, targets):
        targets = one_hot(targets, self.num_classes) \
                  if len(targets.size()) == 1 else targets

        positive = (targets == self.positive).type(torch.float)
        unlabeled = (targets == self.unlabeled).type(torch.float)
        n_positive = torch.max(self.min_count, torch.sum(positive, dim=0))
        n_unlabeled = torch.max(self.min_count, torch.sum(unlabeled, dim=0))

        lam = self.lam_distribution.sample().item()
        lam = min(lam, 1-lam)
        y_positive = self.loss(outputs, lam) * positive
        y_positive_inv = self.loss(- outputs, lam) * positive
        y_unlabeled = self.loss(- outputs, lam) * unlabeled

        positive_risk = self.pi_1 * torch.sum(y_positive, dim=0) / n_positive
        negative_risk = - self.pi_2 * torch.sum(y_positive_inv, dim=0) / n_positive + \
                        torch.sum(y_unlabeled, dim=0) / n_unlabeled

        beta_tensor = self.beta * torch.ones_like(negative_risk, device=self.device).float()
        zeros = torch.zeros_like(negative_risk, device=self.device).float()

        if self.nnPU:
            negative_risk = torch.where(negative_risk < - beta_tensor, - self.gamma * negative_risk, negative_risk)
            positive_risk = torch.where(negative_risk < - beta_tensor, zeros, positive_risk)

        if self.reduction == 'sum':
            positive_risk = torch.sum(positive_risk)
            negative_risk = torch.sum(negative_risk)
        elif self.reduction == 'mean':
            positive_risk = torch.mean(positive_risk)
            negative_risk = torch.mean(negative_risk)

        alpha = torch.tensor(1.0-lam, device=self.device).clamp(min=1e-5, max=1-1e-5)
        scale = -1.0 / (alpha * torch.log(alpha))

        return scale * (positive_risk + negative_risk)