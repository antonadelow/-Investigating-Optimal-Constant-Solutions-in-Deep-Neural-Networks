import os
import copy
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler

from dataset.CIFAR10C import CIFAR10C

from models.resNet import get_features
from models.resNet import ResNet, ResidualBlock
from models.coatnet import CoAtNet
from models.coatnet import *
from models.cait import CaiT

from plot_fig4 import run_plot4
from plot_fig6 import run_plot6

from models.temperature import get_Temperature

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    DEVICE = torch.device('cuda')

def run_plots(save_file, batch_size=128, use_temperature=True):
    #  First create test outputs
    fn = save_file
    if fn[-1] != "/":
        fn = fn + "/"
    # fn = data/cifar10/resnet20_s144_ep200_lr0.001_bs128_wd0.0001_mo0.9_lossreward/model.pt
    # Get seed s0
    seed = fn.split("_s")[1].split("_")[0]
    seeds = [0,77,144]
    dataset = fn.split("/")[1].split("/")[0]
    model_name = fn.split("/")[2].split("_")[0]
    noise_types = ['impulse_noise', 'shot_noise', 'defocus_blur', 'motion_blur', 'speckle_noise']
    # Dataloaders
    transform= transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    if dataset == 'cifar10':
        # CIFAR-10 dataset
        test_dataset = torchvision.datasets.CIFAR10(
            root='../datasets/pytorch_resnet_cifar10/data/',
            train=False,
            # transform=transforms.ToTensor()
            transform=transform
            )
        # CIFAR-10C dataset
        
        corrupt_dataset = lambda corruption_level, noise_type: CIFAR10C(
                path='../datasets/pytorch_resnet_cifar10/data/CIFAR-10-C/',
                corruption_type=noise_type,
                corruption_level=corruption_level,
                transform=transform
            )
    elif dataset == 'cifar100':
        # CIFAR-100 dataset
        test_dataset = torchvision.datasets.CIFAR100(
            root='../datasets/pytorch_resnet_cifar100/data/',
            train=False,
            # transform=transforms.ToTensor()
            transform=transform,
            download=True
            )
        # CIFAR-100C dataset
        corrupt_dataset = lambda corruption_level, noise_type: CIFAR10C(
                path='../datasets/pytorch_resnet_cifar100/data/CIFAR-100-C/',
                corruption_type=noise_type,
                corruption_level=corruption_level,
                transform=transform
            )

    # Data loader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    
    corrupt_loader_levels = {noise_type : [torch.utils.data.DataLoader(
                                    dataset=corrupt_dataset(i, noise_type),
                                    batch_size=batch_size,
                                    shuffle=False) 
                            for i in range(5)] for noise_type in noise_types}
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    # Create test outputs
    if "reward" in save_file:
        num_classes += 1
    
    if model_name == 'resnet20':
        model = ResNet(ResidualBlock, [3, 3, 3], num_classes=num_classes).to(DEVICE)
    elif model_name == 'coatnet':
        D, L, channels = [64, 64, 128, 256, 512], [1, 1, 2, 2, 2], 3
        model = CoAtNet((32, 32), num_classes, L, D, channels, strides = [1, 1, 2, 2, 4],dropout=0.3)
    elif model_name == 'cait':
        dropout = 0.1
        model = CaiT(image_size = 32, patch_size = 4,
                    num_classes = num_classes, dim = 328,
                    depth = 5,   # depth of transformer for patch to patch attention only
                    cls_depth=2, # depth of cross attention of CLS tokens to patch
                    heads = 6, mlp_dim = 240, dropout = dropout, #runnning for new experiment
                    emb_dropout = 0.1, layer_dropout = 0.05)

    
    for seed in seeds:
        print(f"Testing seed {seed}")
        fn_seed = fn.replace('_s0_', f'_s{seed}_')
        model_fn = fn_seed + 'model.pt'
        # model.load_state_dict(torch.load(model_fn, map_location=torch.device(DEVICE)))
        model_torch = torch.load(model_fn)
        if model_name == 'coatnet':
            model.load_state_dict(model_torch['state_dict'])
        else:
            model.load_state_dict(model_torch)
        print("Model loaded")
        model.to(DEVICE)
        test_outputs(fn_seed, noise_types, model, test_loader, corrupt_loader_levels, num_classes, use_temperature)

    # Remove ".pt" from fn
    # Obtain plots fig 3
    for loss in ['mse', 'cross_entropy']:
        print("Obtaining plots Fig3 for loss", loss)
        plot_fig3(fn, noise_types, seeds, test_loader, corrupt_loader_levels, loss, model_name, num_classes)
    # Obtain plots fig 4
    run_plot4(fn, 'impulse_noise', 128, model_name, 'cifar10')
    # Obtain plots fig 6
    run_plot6(fn, model_name, use_temperature=use_temperature)



def test_outputs(fn, noise_types, model, test_loader, corrupt_loader_levels, num_classes, use_temperature=False):
    # Check if model_noisetype_outputs.npy exists
    for noise_type in noise_types:
        if not os.path.exists(f"{fn}model_{noise_type}_outputs.npy"):
            print(f"Creating {fn}model_{noise_type}_outputs.npy")
            test_wrapper(fn, model, num_classes, noise_type, test_loader, corrupt_loader_levels[noise_type])
        # Check if model_temp_noisetype_outputs.npy exists
        if use_temperature and not os.path.exists(f"{fn}model_temp_{noise_type}_outputs.npy"):
            print(f"Creating {fn}model_temp_{noise_type}_outputs.npy")
            test_wrapper(fn, model, num_classes, noise_type, test_loader, corrupt_loader_levels[noise_type], True)

def test_wrapper(fn, model, num_classes, noise_type, test_loader, corrupt_loader_levels, use_temperature=False):
    device = DEVICE
    # Test the model
    model.eval()
    temperature = None
    if use_temperature:
        print("Using temperature")
        temperature = get_Temperature(test_loader, model, num_classes)
        # from models.temperature import ModelWithTemperature
        # temperature = ModelWithTemperature(model, num_classes).set_temperature(test_loader).cpu().detach().numpy()
    acc, out = test(fn, model, test_loader, device, temperature)
    res = [out]
    accs = [acc]
    print(len(corrupt_loader_levels))
    for loader in corrupt_loader_levels:
        temperature = None
        if use_temperature:
            print("Using temperature")
            temperature = get_Temperature(loader, model, num_classes)
            print("Temperature", temperature.cpu().detach().numpy())
            # from models.temperature import ModelWithTemperature
            # temperature = ModelWithTemperature(model, num_classes).set_temperature(loader).cpu().detach().numpy()
        acc, out = test(fn, model, loader, device, temperature)
        res.append(out)
        accs.append(acc)

    print(accs)
    save_name = fn + 'model'
    if use_temperature:
        save_name = f"{save_name}_temp"
    
    np.save(f'{save_name}_{noise_type}_outputs.npy', res)

def test(fn, model, test_loader, device, temperature=None):
    with torch.no_grad():
        correct = 0
        total = 0
        outputs_result = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).float()
            # outputs_np = outputs.cpu().detach().numpy()
            outputs_np = outputs.detach()

            if temperature is not None:
                # outputs = outputs / temperature
                # outputs_np = outputs_np / np.expand_dims(temperature, 0)
                outputs_np = outputs_np / temperature
                # Make a tensor
                # outputs = torch.from_numpy(outputs_np).to(device)

            outputs_result.append(outputs_np.cpu().numpy())

            if "reward" in fn:
                _, predicted = torch.max(outputs.data[:, :-1], 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    outputs_result = np.concatenate(outputs_result)
    return (100 * correct / total, outputs_result)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc1(x))

def normalize_dataset(x, device):
    scaler = StandardScaler()
    res = scaler.fit_transform(x.cpu())
    return torch.from_numpy(res).to(torch.float32).to(device)

def plot_fig3(fn, noise_types, seeds, test_loader, corrupt_loader_levels, loss_type, model_name, num_classes):
    device = DEVICE
    # If scores.npy exists, load it
    
    # if os.path.exists(f'../{fn}model_scores.npy'):
    path_parent = fn.split("/")[:2]
    path_parent = "/".join(path_parent)
    print("Path parent", path_parent)
    if os.path.exists(f'{path_parent}/model_scores.npy'):
        scores = np.load(f'{path_parent}/model_scores.npy')
    # if os.path.exists(f'{fn}model_scores.npy'):
        # scores = np.load(f'{fn}model_scores.npy')
    else:
        scores = [[[] for _ in range(5)] for _ in noise_types]
        test_data_features = get_features(test_loader)[:9000]
        for i, noise_type in enumerate(noise_types):
            print(f'Noise type: {noise_type}')
            for lev_corrpt in range(5):
                # Get the 1000 first images from test
                # Get the 9000 out of distribution images from corrupt_loader_levels[lev_corrpt]
                corrupt_features = get_features(corrupt_loader_levels[noise_type][lev_corrpt])
                tr_corrupt_features = corrupt_features[:9000]
                val_corrupt_features = corrupt_features[9000:]
                X = torch.concatenate((test_data_features, tr_corrupt_features)).to(device)
                # Get the labels
                Y = torch.concatenate((torch.ones(9000, dtype=torch.long), torch.zeros(9000, dtype=torch.long))).to(device)
                dataset = torch.utils.data.TensorDataset(normalize_dataset(X, device), Y)
                loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

                # Get the outputs
                # Net is a simple linear model with 2 layers, which classifies the images into 2 classes, either distribution or out of distribution
                net = Net()
                net.to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=1)
                curr_score = []
                running_loss = 0.0
                epochs = 30
                # epochs = 100
                for epoch in range(epochs):
                    # outputs = net(X)
                    for inputs, labels in loader:
                        optimizer.zero_grad()
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                    with torch.no_grad():
                        outputs = net(val_corrupt_features[:1000].to(device))

                        prob = nn.functional.log_softmax(outputs, dim=1)[:,0]
                        curr_score.append(prob.mean().item())
                        if epoch == epochs-1:
                            print(f'Corruption level: {lev_corrpt+1} Epoch %d loss: %.3f' % (epoch + 1, 
                            running_loss / (epoch+1)))
                scores[i][lev_corrpt] = np.exp(np.mean(curr_score, axis=0))
                print("Score", scores[i][lev_corrpt])
        scores = np.array(scores)
        # Save scores
        np.save(f'{fn}model_scores.npy', scores)
    # For each noise type, plot the scores with the intervals for each corruption level, all in the same plot
    ax = plt.subplot(111)
    

    for i, noise_type in enumerate(noise_types):
        # Plot the distance in MSE between the scores of the results array and the scores array
        # For each corruption level, get the mean and the standard deviation
        dists = [[] for _ in range(6)]
        for seed in seeds:
            fn_seed = fn.replace('_s0_', f'_s{seed}_')
            res = np.load(os.path.join(f'{fn_seed}model_{noise_type}_outputs.npy'), allow_pickle=True)
            # Add without corruption
            for corruption_level in range(6):
                if loss_type == 'mse':
                    # The OCS for MSE is the average of the labels of the network as defined in B.4
                    average_target = np.mean(test_loader.dataset.targets)
                    f_star = [-average_target for _ in range(num_classes)]
                    dist = np.mean((res[corruption_level] - f_star)**2)
                elif loss_type == 'cross_entropy':
                    prob_class = nn.functional.softmax(torch.from_numpy(res[corruption_level]), dim=1)
                    # For cross entropy, the OCS is the prior probability of each class, which is 0.1
                    average_target = 0.1
                    prior = np.array([average_target for _ in range(num_classes)])
                    # dist = torch.nanmean((prob_class * torch.log(prob_class/prior))).numpy()
                    # Add eps to avoid log(0)
                    dist = np.mean((prob_class * np.log((prob_class/prior)+1e-10)).numpy())
                elif loss_type == 'mll':
                    # The OCS for MLL is the average of the labels of the network as defined in B.4
                    average_target = np.mean(test_loader.dataset.targets)
                    f_star = [-average_target for _ in range(num_classes)]
                    dist = np.mean((res[corruption_level] - f_star)**2)
                dists[corruption_level].append(dist)
        # Plot error bars
        if i == 0:
            # Plot for the oracle solution, the score 
            ax.errorbar([0.5], np.mean(dists[0]), yerr=np.std(dists[0]), label="oracle", fmt='*', markersize=8, alpha=0.8)
        dists = dists[1:]
        ax.errorbar(scores[i], np.mean(dists, axis=1), yerr=np.std(dists, axis=1), label=noise_type, fmt='o', markersize=8, alpha=0.8)
        for j, score in enumerate(scores[i]):
            ax.annotate(f'l_{j}', (score, np.mean(dists[j])), xytext=(5, 5), textcoords='offset points')
        
    ax.legend()
    
    plt.xlabel('OOD Score')
    plt.ylabel('Distance to OCS')
    # plt.title(f'OOD Score vs Distance to OCS for {dataset} dataset ({loss_type} loss)')
    
    plt.savefig(f'{fn}{model_name}_scores_plot_{loss_type}.png')
    plt.clf()

    # plt.show()


if __name__ == '__main__':
    # Example usage python run_plots.py -sf data/cifar10/coatnet_s0_ep100_lr0.001_bs128_wd0.0001_mo0.9_losscross_entropy/model.pt
    # One per type of model, no need to run for each seed as it is the same model
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--save_file", type=str,
                        required=True,
                        help="Specify the name of the file to load the model")
    parser.add_argument("-no_temp", "--no_temperature", action='store_false',
                        help="Specify if you don't want to use temperature")

    args = parser.parse_args()
    # Set seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        print("Using CUDA")
        torch.cuda.manual_seed_all(seed)

    run_plots(args.save_file, use_temperature=args.no_temperature)