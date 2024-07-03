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

from dataset.CIFAR10C import CIFAR10C

from models.resNet import get_features
from models.resNet import ResNet, ResidualBlock
from models.coatnet import CoAtNet
from models.temperature import get_Temperature

def run_plot6(save_file, model_name, use_temperature=True):

    fn = save_file
    # results = np.load(os.path.join(f'{fn}_ns_{noise_type}_outputs.npy'), allow_pickle=True)
    
    noise_types = ["impulse_noise", "shot_noise", "defocus_blur", "motion_blur", "speckle_noise"]
    # noise_types = ["impulse_noise"]
    # models = ["resnet20", "coatnet", "cait"]

    def corrupt_dataset(corruption_level, noise_type):
        if corruption_level == 0:
            return np.array(torchvision.datasets.CIFAR10(
                root='../datasets/pytorch_resnet_cifar10/data/',
                train=False).targets)
        else:
            return CIFAR10C(
                path='../datasets/pytorch_resnet_cifar10/data/CIFAR-10-C/',
                corruption_type=noise_type,
                corruption_level=corruption_level-1
                ).labels
    # one_hot = np.zeros((len(labels), 11))
    # one_hot[np.arange(len(labels)), labels] = 1
    # reward_all = (4+1)*one_hot - 4
    # reward_all[:, -1] = 0
    print("Computing reward for baseline:", fn)
    reward_mean_base, abstain_mean_base = get_reward_mean(fn, noise_types, corrupt_dataset)
    fn_2 = fn.replace('_losscross_entropy', '_lossreward')
    print("Computing reward for reward:", fn_2)
    reward_mean_reward, abstain_mean_reward = get_reward_mean(fn_2, noise_types, corrupt_dataset)
    fn_2 = fn_2.replace('/model', '/model_temp')
    if use_temperature:
        print("Computing reward for temperature:", fn_2)
        reward_mean_temp, abstain_mean_temp = get_reward_mean(fn_2, noise_types, corrupt_dataset)

    # Plot the results
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    # ax.set_prop_cycle('color', plt.cm.tab10(np.linspace(0, 1, 10)))
    for i, noise in enumerate(noise_types):
        ax[i].plot(range(6), np.mean(reward_mean_base[i], axis=1), label="Baseline", c="C0")
        ax[i].errorbar(range(6), np.mean(reward_mean_base[i], axis=1), yerr=np.std(reward_mean_base[i], axis=1), label="Baseline", capsize=5, fmt="none", c="C0")
        ax[i].plot(range(6), np.mean(reward_mean_reward[i], axis=1), label="Reward", c="C1")
        ax[i].errorbar(range(6), np.mean(reward_mean_reward[i], axis=1), yerr=np.std(reward_mean_reward[i], axis=1), label="Reward", capsize=5, fmt="none", c="C1")
        if use_temperature:
            ax[i].plot(range(6), np.mean(reward_mean_temp[i], axis=1), label="Temperature", c="C2")
            ax[i].errorbar(range(6), np.mean(reward_mean_temp[i], axis=1), yerr=np.std(reward_mean_temp[i], axis=1), label="Temperature", capsize=5, fmt="none", c="C2")
        ax[i].set_title(noise)
        ax[i].set_xlabel("Corruption level")
        ax[i].set_ylabel("Reward")
        ax[i].legend()
        
    plt.tight_layout()
    plt.savefig(f"{fn}{model_name}_reward.png")
    plt.clf()

    # plot for abstain
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    for i, noise in enumerate(noise_types):
        ax[i].plot(range(6), np.mean(abstain_mean_base[i], axis=1), label="Baseline", c="C0")
        ax[i].errorbar(range(6), np.mean(abstain_mean_base[i], axis=1), yerr=np.std(abstain_mean_base[i], axis=1), label="Baseline", capsize=5, fmt="none", c="C0")
        ax[i].plot(range(6), np.mean(abstain_mean_reward[i], axis=1), label="Reward", c="C1")
        ax[i].errorbar(range(6), np.mean(abstain_mean_reward[i], axis=1), yerr=np.std(abstain_mean_reward[i], axis=1), label="Reward", capsize=5, fmt="none", c="C1")
        if use_temperature:
            ax[i].plot(range(6), np.mean(abstain_mean_temp[i], axis=1), label="Temperature", c="C2")
            ax[i].errorbar(range(6), np.mean(abstain_mean_temp[i], axis=1), yerr=np.std(abstain_mean_temp[i], axis=1), label="Temperature", capsize=5, fmt="none", c="C2")
        ax[i].set_title(noise)
        ax[i].set_xlabel("Corruption level")
        ax[i].set_ylabel("Abstain")
        ax[i].legend()

    plt.tight_layout()
    plt.savefig(f"{fn}{model_name}_abstain.png")



def get_reward_mean(fn, noise_types, corrupt_dataset):
    seeds = [0,77,144]
    rewards_mean = [[[] for _ in range(6)] for _ in range(len(noise_types))]
    abstain_mean = [[[] for _ in range(6)] for _ in range(len(noise_types))]
    for seed in seeds:
        for i, noise_type in enumerate(noise_types):
            fn_seed = fn.replace('_s0_', f'_s{seed}_')
            outputs = np.load(os.path.join(f'{fn_seed}model_{noise_type}_outputs.npy'), allow_pickle=True)
            for c_level in range(6):
                pred_class = np.argmax(outputs[c_level], axis=-1)
                labels = corrupt_dataset(c_level, noise_type)
                # make an array of 0s and 1s
                correct = (pred_class == labels).astype(int)
                # Replace every 0 with -4 and every 1 with 1
                correct = (5)*correct - 4
                # Set all indices with label 10 to 0
                correct[pred_class == 10] = 0
                rewards_mean[i][c_level].append(np.mean(correct, axis=0))
                abstain_mean[i][c_level].append(np.mean(pred_class == 10, axis=0))
    rewards_mean = np.array(rewards_mean)
    abstain_mean = np.array(abstain_mean)
    return rewards_mean, abstain_mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str,
                        choices=['cifar10', 'cifar100'],
                        default='cifar10',
                        help="Select which dataset to use")
    parser.add_argument("-sf", "--save_file", type=str,
                        required=True,
                        help="Specify the name of the file to load the model")
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
    model_name = args.save_file.split('/')[1].split('_')[0]
    run_plot6(args.save_file, model_name)