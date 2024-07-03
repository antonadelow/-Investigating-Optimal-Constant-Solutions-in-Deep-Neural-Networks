import torch
import torch.nn as nn
import torchvision.transforms as transforms

from einops import rearrange
from einops.layers.torch import Rearrange

torch.manual_seed(0)
DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    DEVICE = torch.device('cuda')

def temperature_scale(temp, logits, num_classes):
    temperature = temp.unsqueeze(0).expand(logits.size(0), num_classes)
    return logits / temperature

def get_Temperature(data, model, num_classes, lr=0.01):
    # Initialize temperature
    temperature = nn.Parameter(torch.ones(num_classes, requires_grad=True, device=DEVICE)* 1.5, requires_grad=True)
    # temperature.requires_grad = True
    # Define and train a simple model to get the temperature
    outputs_all = None
    labels_all = None
    with torch.no_grad():
        # Get the outputs of the model
        for i, (input, label) in enumerate(data):
            input_var = input.to(DEVICE)
            label = label.to(DEVICE)
            output = model(input_var)
            if i == 0:
                outputs_all = output
                labels_all = label
            else:
                outputs_all = torch.cat((outputs_all, output), dim=0)
                labels_all = torch.cat((labels_all, label), dim=0)
    # TODO:DELETE
    nll_criterion = nn.CrossEntropyLoss().to(DEVICE)
    t_cuda = temperature.to(DEVICE)
    before_temperature_nll = nll_criterion(temperature_scale(t_cuda, outputs_all, num_classes), labels_all).item()
    print('Before temperature - NLL: %.3f' % (before_temperature_nll))
    # TODO:DELETE end
    # Run optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # temperature = temperature.to(DEVICE)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=1000)
    loss_crit = nn.CrossEntropyLoss().to(DEVICE)
    def step_closure():
        optimizer.zero_grad()
        loss = loss_crit(temperature_scale(temperature, outputs_all, num_classes), labels_all)
        loss.backward()
        return loss
    
    optimizer.step(step_closure)
    # TODO:DELETE
    after_temperature_nll = nll_criterion(temperature_scale(temperature, outputs_all, num_classes), labels_all).item()
    print('After temperature - NLL: %.3f' % (after_temperature_nll))
    # TODO:DELETE end

    return temperature.detach()


def get_Temperature_cpu(data, model, num_classes, lr=0.01):
    model = model.to('cpu')
    # Initialize temperature
    temperature = nn.Parameter(torch.ones(num_classes) * 1.5)
    # Define and train a simple model to get the temperature
    outputs_all = None
    labels_all = None
    with torch.no_grad():
        # Get the outputs of the model
        for i, (input, label) in enumerate(data):
            input_var = input
            label = label
            output = model(input_var)
            if i == 0:
                outputs_all = output
                labels_all = label
            else:
                outputs_all = torch.cat((outputs_all, output), dim=0)
                labels_all = torch.cat((labels_all, label), dim=0)
    # TODO:DELETE
    nll_criterion = nn.CrossEntropyLoss()
    t_cuda = temperature
    before_temperature_nll = nll_criterion(temperature_scale(t_cuda, outputs_all, num_classes), labels_all).item()
    print('Before temperature - NLL: %.3f' % (before_temperature_nll))
    # TODO:DELETE end
    # Run optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=1000)
    temperature = temperature
    loss_crit = nn.CrossEntropyLoss()
    def step_closure():
        optimizer.zero_grad()
        loss = loss_crit(temperature_scale(temperature, outputs_all, num_classes), labels_all)
        loss.backward()
        return loss
    
    optimizer.step(step_closure)
    # TODO:DELETE
    after_temperature_nll = nll_criterion(temperature_scale(temperature, outputs_all, num_classes), labels_all).item()
    print('After temperature - NLL: %.3f' % (after_temperature_nll))
    # TODO:DELETE end

    model.to('cuda')
    return temperature.detach().numpy()


# TODO: DELETE GIT
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn, optim
from torch.nn import functional as F
class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, num_classes, lr = 0.01, max_iter = 1000):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.max_iter = max_iter
        self.temperature = nn.Parameter(torch.ones(num_classes) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), self.num_classes)
        return logits / temperature

    def set_temperature(self, loader):
        self.to(DEVICE)
        nll_criterion = nn.CrossEntropyLoss().to(DEVICE)

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in loader:
                input = input.to(DEVICE)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(DEVICE)
            labels = torch.cat(labels_list).to(DEVICE)

        before_temperature_nll = nll_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f' % (before_temperature_nll))

        optimizer = optim.LBFGS([self.temperature], lr=self.lr, max_iter=1000)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        print(self.temperature)
        print('After temperature - NLL: %.3f' % (after_temperature_nll))

        return self.temperature