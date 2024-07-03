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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        """ self.fc1 = nn.Linear(512, 128)
        # self.fc1 = nn.Linear(1000, 128)
        self.fc2 = nn.Linear(128, 2) """
        self.fc1 = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """ x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x """
        return self.sigmoid(self.fc1(x))

def normalize_dataset(x, device):
    # x_flat = x.reshape(-1, 2)
    # mean = torch.mean(x_flat, dim=0)
    # std = torch.std(x_flat, dim=0)
    # x_flat = (x_flat - mean) / std
    # x = x_flat.reshape(x.shape)
    # return x
    scaler = StandardScaler()
    res = scaler.fit_transform(x.cpu())
    return torch.from_numpy(res).to(torch.float32).to(device)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fn = args.save_file
    # results = np.load(os.path.join(f'{fn}_ns_{args.noise_type}_outputs.npy'), allow_pickle=True)

    # In paper, they use a pretrained Resnet18 to obtain features, then they create a model trained on a subset of 1000 first images defined by the correct image features, and then 9000 samples from the out of distribution dataset.
    # The labels are set to ones for the 1000 distribution images and zeros for all the other 9000 ood images
    # The model is trained based on an sklearn pipeline with a StandardScaler and an SGDClassifier with a log loss function for 30 epochs

    # The final score is defined as the log probabilities of the original 1000 images, ignoring the outptus of the 9000 out of distribution images

    # This process is done for each of the corruption levels

    transform= transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

    if args.dataset == 'cifar10':
        num_classes = 10
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
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True)

    noise_types = ["impulse_noise", "shot_noise", "defocus_blur", "motion_blur", "speckle_noise"]
    # DEBUGGING
    # noise_types = ["impulse_noise"]
    # noise_types = ["defocus_blur"]
    corrupt_loader_levels = [[torch.utils.data.DataLoader(
                                    dataset=corrupt_dataset(i, noise_type),
                                    batch_size=args.batch_size,
                                    shuffle=True) 
                            for i in range(5)] for noise_type in noise_types]
    scores = [[[] for _ in range(5)] for _ in noise_types]
    test_data_features = get_features(test_loader)[:9000]
    for i, noise_type in enumerate(noise_types):
        print(f'Noise type: {noise_type}')
        for lev_corrpt in range(5):
            # Get the 1000 first images from test
            # Get the 9000 out of distribution images from corrupt_loader_levels[lev_corrpt]
            # TO Discuss -> Difference of obtaining features from Imagenette or using the image itself
            # X = torch.concatenate((test_loader.dataset.data[:1000], corrupt_loader_levels[i][lev_corrpt].dataset.data[:9000])).to(device)
            corrupt_features = get_features(corrupt_loader_levels[i][lev_corrpt])
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
            # criterion = nn.BCEWithLogitsLoss()
            # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=10)
            # optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=0.1)
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
                    # loss = criterion(outputs, torch.ones(1000).long().to(device))
                    # running_loss += loss.item()

                    if epoch == epochs-1:
                        print(f'Corruption level: {lev_corrpt+1} Epoch %d loss: %.3f' % (epoch + 1, 
                        running_loss / (epoch+1)))
            scores[i][lev_corrpt] = np.exp(np.mean(curr_score, axis=0))
            # FROM PAPER
            """ from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler 
            from sklearn.pipeline import make_pipeline
            from scipy.special import softmax
            curr_score = []
            for _ in range(30):
                clf = make_pipeline(StandardScaler(), SGDClassifier(loss="log", max_iter=20, alpha=10))
                clf.fit(X.cpu().numpy(), Y.cpu().numpy())
                score = clf.predict_log_proba(val_corrupt_features.cpu().numpy())[:, 0].mean()
                curr_score.append(score)
            # ood_scores_all[corruption_type].append(np.mean(curr_score))
            scores[i][lev_corrpt] = np.exp(np.mean(curr_score)) """
            # END PAPER
            print("Score", scores[i][lev_corrpt])
    scores = np.array(scores)
    # For each noise type, plot the scores with the intervals for each corruption level, all in the same plot
    seeds = [0, 77, 144]
    ax = plt.subplot(111)
    

    for i, noise_type in enumerate(noise_types):
        # ax.plot(scores[i], label=noise_type)
        # Plot the distance in MSE between the scores of the results array and the scores array
        # For each corruption level, get the mean and the standard deviation
        # dists = [[] for _ in range(5)]
        dists = [[] for _ in range(6)]
        for seed in seeds:
            fn_seed = fn.replace('_s0_', f'_s{seed}_')
            res = np.load(os.path.join(f'{fn_seed}_{noise_type}_outputs.npy'), allow_pickle=True)
            # for corruption_level in range(5):
                # dist = np.mean((res[corruption_level+1] - f_star)**2)
            # Add without corruption
            for corruption_level in range(6):
                if args.loss == 'mse':
                    # The OCS for MSE is the average of the labels of the network as defined in B.4
                    average_target = np.mean(test_loader.dataset.targets)
                    f_star = [-average_target for _ in range(10)]
                    dist = np.mean((res[corruption_level] - f_star)**2)
                elif args.loss == 'cross_entropy':
                    prob_class = nn.functional.softmax(torch.from_numpy(res[corruption_level]), dim=1)
                    # For cross entropy, the OCS is the prior probability of each class, which is 0.1
                    average_target = 0.1
                    prior = np.array([average_target for _ in range(10)])
                    # dist = np.mean((prob_class * np.log(prob_class/prior)).sum(axis=1))
                    dist = torch.mean((prob_class * torch.log(prob_class/prior))).numpy()
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
    # plt.tight_layout()
    plt.title(f'OOD Score vs Distance to OCS for {args.dataset} dataset ({args.loss} loss)')
    
    plt.savefig(f'{fn}_scores_plot_{args.loss}.png')

    plt.show()


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
    parser.add_argument("-bs", "--batch_size", type=int,
                        choices=[64, 128, 256],
                        default=128,
                        help="Select which batch size to use")
    parser.add_argument("-l", "--loss", type=str,
                        choices=['cross_entropy','mse'],
                        default='cross_entropy',
                        help="Specify the loss to use")

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

    main(args)


