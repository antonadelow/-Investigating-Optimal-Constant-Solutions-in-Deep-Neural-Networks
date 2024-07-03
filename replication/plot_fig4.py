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

from einops import rearrange, repeat

from dataset.CIFAR10C import CIFAR10C

from models.resNet import get_features
from models.resNet import ResNet, ResidualBlock
from models.coatnet import CoAtNet
from models.cait import CaiT


# Recreate the plot in Figure 4 of the paper, corresponding to the code in the analyze.ipynb notebook
# It consists of 4 plots
# 1. Norm of the features for each layer of the Resnet18 model
# 2. proportion of network features that lie in the span of the following layer's features
# 3. 4. Accumulation of model constants compared to ocs for cross-entropy and mse
    
def run_plot4(save_file, noise_type, batch_size, model_name, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [0,77,144]

    fn = save_file
    if "reward" in save_file:
        output_classes = 11
    else:
        output_classes = 10
    if model_name == 'resnet20':
        model = ResNet(ResidualBlock, [3, 3, 3], num_classes=output_classes).to(device)
    elif model_name == 'coatnet':
        # If using coatnet, set optimizer to adamw and scheduler to onecycle
        D, L, channels = [64, 64, 128, 256, 512], [1, 1, 2, 2, 2], 3
        model = CoAtNet((32, 32), output_classes, L, D, channels, strides = [1, 1, 2, 2, 4], dropout=0.3)
    elif model_name == 'cait':
        model = CaiT(image_size = 32, patch_size = 4,
                    num_classes = output_classes, dim = 328,
                    depth = 5,   # depth of transformer for patch to patch attention only
                    cls_depth=2, # depth of cross attention of CLS tokens to patch
                    heads = 6, mlp_dim = 240, dropout = 0.1,
                    emb_dropout = 0.1, layer_dropout = 0.05)

    fn = save_file
    
    # model.load_state_dict(torch.load(fn))
    # model.to(device)
    models = []
    for seed in seeds:
        model_i = copy.deepcopy(model)
        # model_i.load_state_dict(torch.load(fn.replace('_s0_', f'_s{seed}_')+"model.pt"))
        model_torch = torch.load(fn.replace('_s0_', f'_s{seed}_')+"model.pt")
        if model_name == 'coatnet':
            model_i.load_state_dict(model_torch['state_dict'])
        else:
            model_i.load_state_dict(model_torch)
        model_i.to(device)
        models.append(model_i)

    transform= transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

    if dataset == 'cifar10':
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
        
    # Data loader
    # Only use the impulse noise
    # noise_types = ["impulse_noise"]
    data_loaders = [torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)]+\
                    [torch.utils.data.DataLoader(
                                    dataset=corrupt_dataset(i, "impulse_noise"),
                                    batch_size=batch_size,
                                    shuffle=True) 
                                for i in range(5)]
    # results_layers = [[] for _ in range(6)]
    # results_layers = [[[[],[],[]] for _ in range(6)] for _ in range(6)]
    # As numpy array
    results_layers = np.zeros((6, 6, 3, 10000))
    for i, model in enumerate(models):
        if model_name == 'resnet20':
            layers = [model.conv, lambda x: model.layer1(model.relu(model.bn(x))), model.layer2, model.layer3, lambda x: model.fc(model.avg_pool(x).view(x.size(0), -1))]
        elif model_name == 'coatnet':
            print("Layers for coatnet")
            # layers = [model._modules[str(0)], lambda x: model._modules[str(1)](model._modules[str(2)](x)), model._modules[str(3)], model._modules[str(4)], lambda x: model._modules[str(8)](model._modules[str(7)](model._modules[str(6)](model._modules[str(5)](x))))]
            # One per line
            layers = [model._modules[str(0)]]
            layers.append(model._modules[str(1)])
            layers.append(model._modules[str(2)])
            layers.append(model._modules[str(3)])
            layers.append(model._modules[str(4)])
            layers.append(lambda x: model._modules[str(8)](model._modules[str(7)](model._modules[str(6)](model._modules[str(5)](x)))))
        elif model_name == 'cait':
            layers = [model.to_patch_embedding, lambda x: model.dropout(model.pos_embedding[:, :x.shape[1]]+x), model.patch_transformer, lambda x: model.cls_transformer(repeat(model.cls_token, '() n d -> b n d', b = x.shape[0]), context = x), lambda x: model.mlp_head(x[:, 0])]
        print(f'Seed {seeds[i]}')
        if model_name == 'coatnet':
             for lev_corrpt in range(6):
                print(f'\tCoatNet Corruption level {lev_corrpt}')
                data_loader = data_loaders[lev_corrpt]
                for idx, (x, y) in enumerate(data_loader):
                    cur_range_start = idx*128
                    cur_range_end = min((idx+1)*128, 10000)
                    x = x.to(device)

                    out = layers[0](x)
                    results_layers[lev_corrpt][0][i][cur_range_start:cur_range_end] = torch.norm(out.reshape(out.size(0), -1), dim=1).detach().cpu().numpy()
                    out = layers[1](out)
                    results_layers[lev_corrpt][1][i][cur_range_start:cur_range_end] = torch.norm(out.reshape(out.size(0), -1), dim=1).detach().cpu().numpy()
                    out = layers[2](out)
                    results_layers[lev_corrpt][2][i][cur_range_start:cur_range_end] = torch.norm(out.reshape(out.size(0), -1), dim=1).detach().cpu().numpy()
                    out = layers[3](out)
                    results_layers[lev_corrpt][3][i][cur_range_start:cur_range_end] = torch.norm(out.reshape(out.size(0), -1), dim=1).detach().cpu().numpy()
                    out = layers[4](out)
                    results_layers[lev_corrpt][4][i][cur_range_start:cur_range_end] = torch.norm(out.reshape(out.size(0), -1), dim=1).detach().cpu().numpy()
                    out = layers[5](out)
                    results_layers[lev_corrpt][5][i][cur_range_start:cur_range_end] = torch.norm(out.reshape(out.size(0), -1), dim=1).detach().cpu().numpy()

        else:
            for lev_corrpt in range(6):
                print(f'\tCorruption level {lev_corrpt}')
                data_loader = data_loaders[lev_corrpt]
                # res = [[] for _ in range(6)]
                for idx, (x, y) in enumerate(data_loader):
                    cur_range_start = idx*batch_size
                    cur_range_end = (idx+1)*batch_size
                    x = x.to(device)
                    results_layers[lev_corrpt][0][i][cur_range_start:cur_range_end] = torch.norm(x.reshape(x.size(0), -1), dim=1).detach().cpu().numpy()
                    # out = model.conv(x)
                    out = layers[0](x)
                    results_layers[lev_corrpt][1][i][cur_range_start:cur_range_end] = torch.norm(out.reshape(out.size(0), -1), dim=1).detach().cpu().numpy()
                    # out = model.layer1(model.relu(model.bn(out)))
                    out = layers[1](out)
                    results_layers[lev_corrpt][2][i][cur_range_start:cur_range_end] = torch.norm(out.reshape(out.size(0), -1), dim=1).detach().cpu().numpy()
                    # out = model.layer2(out)
                    out = layers[2](out)
                    results_layers[lev_corrpt][3][i][cur_range_start:cur_range_end] = torch.norm(out.reshape(out.size(0), -1), dim=1).detach().cpu().numpy()
                    # out = model.layer3(out)
                    out = layers[3](out)
                    results_layers[lev_corrpt][4][i][cur_range_start:cur_range_end] = torch.norm(out.reshape(out.size(0), -1), dim=1).detach().cpu().numpy()
                    # out = model.fc(model.avg_pool(out).view(out.size(0), -1))
                    out = layers[4](out)
                    results_layers[lev_corrpt][5][i][cur_range_start:cur_range_end] = torch.norm(out.reshape(out.size(0), -1), dim=1).detach().cpu().numpy()
            
    # For each noise type, plot the norm of the features for each layer of the Resnet18 model
    ax = plt.subplot(111)

    no_noise = [np.mean(res) for res in results_layers[0]]
    no_noise_std = [np.std(np.mean(res, axis=1)) for res in results_layers[0]]
    no_noise = np.array(no_noise)
    for corruption_level in range(6):
        ratio = np.array([np.mean(res) for res in results_layers[corruption_level]])
        ratio_std = np.array([np.std(np.mean(res, axis=1)) for res in results_layers[corruption_level]])[1:]
        # ratio = np.concatenate(([no_noise[0]], ratio))
        ratio = ratio/no_noise
        ratio_std = np.abs((ratio_std/no_noise_std[1:])-1)
        # Replace nan by 0 
        ratio_std = np.concatenate(([0], ratio_std))
        # ax.plot(ratio, label=f'level_{corruption_level}')
        ax.errorbar(range(6), ratio, yerr=ratio_std, label=f'level_{corruption_level}', elinewidth=1, capsize=4)

    ax.legend()
    ax.set_xticks(range(5))
    if model_name == 'coatnet':
        ax.set_xticklabels(['S0', 'S1', 'S2', 'S3', 'S4'])
        ax.set_xlabel('Stage')
    else:
        ax.set_xticklabels(['conv1', 'layer1', 'layer2', 'layer3', 'fc'])
        ax.set_xlabel('Layer')

    # ax.set_ylim([0.6, 1.4])
    
    ax.set_ylabel('Normalized norm')
    
    plt.savefig(f'{fn}{model_name}_normalizedlayers_plot4.png')
    # plt.savefig(f'testingg.png')

    # plt.show()

    # Empty plot
    plt.clf()
    print("Layer Norm plot saved")


    # Second plot Violin plot: 
    # Uses projected_ratio_all, how many of the features in one layer lie in the span of the features in the next layer

    # Third plot: Accumulation of model constants compared to ocs for cross-entropy and mse (For the last linear layer)
    # Based on zero_input_output

    # Compute the result of using an input of zeros
    # input = torch.zeros(1, 3, 32, 32).to(device)
    if model_name == 'coatnet':
        input = torch.zeros(1, 512, 2, 2).to(device)
        constants_output = np.zeros((len(models), output_classes))
        for i, model in enumerate(models):
            if "reward" in save_file:
                x = torch.zeros((1, 64, 8, 8)).cuda()
                x = model.layer3[1](x)
                x = model.layer3[2](x)
                x = torch.nn.functional.avg_pool2d(x, x.size()[3])
                x = x.view(x.size(0), -1)
                out = model(input)
                constants_output[i] = out.detach().cpu().numpy()[0]
                print(constants_output)
                ocs = [ -3.5 for _ in range(11)]
            else:
                out = layers[-1](input)   
                constants_output[i] = torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()[0]
                ocs = [1/output_classes for _ in range(output_classes)]
    elif model_name == 'cait':
        return
    else:
        input = torch.zeros(1,64).to(device)
        constants_output = np.zeros((len(models), output_classes))
        for i, model in enumerate(models):
            if "reward" in save_file:
                # constants_output = out.detach().cpu().numpy()[0]
                x = torch.zeros((1, 64, 8, 8)).cuda()
                x = model.layer3[1](x)
                x = model.layer3[2](x)
                x = torch.nn.functional.avg_pool2d(x, x.size()[3])
                x = x.view(x.size(0), -1)
                out = model.fc(input)
                constants_output[i] = out.detach().cpu().numpy()[0]
                print(constants_output)
                ocs = [ -3.5 for _ in range(11)]
            else:
                out = model.fc(input)   
                constants_output[i] = torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()[0]
                ocs = [1/output_classes for _ in range(output_classes)]
    
    # Plot the results
    ax = plt.subplot(111)
    ax.bar(range(output_classes), constants_output.mean(axis=0), label='model')
    ax.errorbar(range(output_classes), constants_output.mean(axis=0), yerr=constants_output.std(axis=0), label='model', fmt='none', ecolor='black', elinewidth=1, capsize=2)
    # ax.plot(ocs, label='ocs')
    ax.bar(range(output_classes), ocs, label='ocs', color="none", edgecolor='black', hatch='///')
    ax.legend()
    ax.set_xticks(range(output_classes))
    ax.set_xticklabels(range(output_classes))
    ax.set_xlabel('Class')
    ax.set_ylabel('Probability')
    ax.set_ylim([0, 1])
    plt.savefig(f'{fn}{model_name}_constant_plot4.png')
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str,
                        choices=['cifar10', 'cifar100'],
                        default='cifar10',
                        help="Select which dataset to use")
    """ parser.add_argument("-n", "--noise_type", type=str,
                        choices=['impulse_noise', 'shot_noise', 'defocus_blur', 'motion_blur', 'speckle_noise'],
                        default='impulse_noise',
                        help="Select which noise type to use") """
    parser.add_argument("-sf", "--save_file", type=str,
                        required=True,
                        help="Specify the name of the file to load the model")
    parser.add_argument("-l", "--loss", type=str,
                        choices=['cross_entropy','mse', 'reward'],
                        default='cross_entropy',
                        help="Specify the loss to use")
    parser.add_argument("-m", "--model", type=str,
                        choices=['resnet20', 'cait','coatnet'],
                        default='resnet20',
                        help="Select which model to use")

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

    run_plot4(args.save_file, 'impulse_noise', 128, args.model, args.dataset)


