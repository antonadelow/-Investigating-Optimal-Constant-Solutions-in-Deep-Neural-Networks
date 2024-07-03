import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# argparse
import argparse

from models.resNet import ResNet, ResidualBlock
from models.git_models import resnet20 as git_resnet20
from models.coatnet import CoAtNet

from dataset.CIFAR10C import CIFAR10C
from models.temperature import get_Temperature

def main(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    # num_epochs = 200
    # learning_rate = 0.001

    # Image preprocessing modules
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform= transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    if args.dataset == 'cifar10':
        # CIFAR-10 dataset
        test_dataset = torchvision.datasets.CIFAR10(
            root='../datasets/pytorch_resnet_cifar10/data/',
            train=False,
            # transform=transforms.ToTensor()
            transform=transform
            )
        # CIFAR-10C dataset
        
        corrupt_dataset = lambda corruption_level: CIFAR10C(
                path='../datasets/pytorch_resnet_cifar10/data/CIFAR-10-C/',
                corruption_type=args.noise_type,
                corruption_level=corruption_level,
                transform=transform
            )

    # Data loader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)
    
    corrupt_loader_levels = [torch.utils.data.DataLoader(
                                    dataset=corrupt_dataset(i),
                                    batch_size=args.batch_size,
                                    shuffle=False) 
                            for i in range(5)]

    if args.model == 'resnet20':
        if "reward" in args.save_file:
            num_classes = 11
            model = ResNet(ResidualBlock, [3, 3, 3], num_classes=num_classes).to(device)
        else:
            num_classes = 10
            model = ResNet(ResidualBlock, [3, 3, 3]).to(device)
    elif args.model == 'gitcifar10':
        model = git_resnet20().to(device)
    elif args.model == 'coatnet':
        D, L, num_classes, channels = [64, 64, 128, 256, 512], [2, 2, 3, 3, 2], 10, 3
        model = CoAtNet((32, 32), num_classes, L, D, channels, strides = [1, 1, 2, 2, 4],dropout=0.3)

    fn = args.save_file
    
    # model.load_state_dict(torch.load(fn))
    model.load_state_dict(torch.load(fn.replace('_DEBUG', ''), map_location=torch.device(device)))
    model.to(device)
    
    # Test the model
    model.eval()
    if args.temperature:
        print("Using temperature")
        temperature = get_Temperature(test_loader, model, num_classes)
        # from models.temperature import ModelWithTemperature
        # temperature = ModelWithTemperature(model, num_classes).set_temperature(test_loader).cpu().detach().numpy()
    acc, out = test(model, test_loader, device, temperature)
    res = [out]
    accs = [acc]
    for loader in corrupt_loader_levels:
        temperature = None
        if args.temperature:
            print("Using temperature")
            temperature = get_Temperature(loader, model, num_classes)
            # from models.temperature import ModelWithTemperature
            # temperature = ModelWithTemperature(model, num_classes).set_temperature(loader).cpu().detach().numpy()
        acc, out = test(model, loader, device, temperature)
        res.append(out)
        accs.append(acc)

    print(accs)
    save_name = fn.replace('.pt', '')
    if args.temperature:
        save_name = f"{save_name}_temp"
    
    np.save(f'{save_name}_{args.noise_type}_outputs.npy', res)


def test(model, test_loader, device, temperature=None):
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

            outputs_result.append(outputs_np)

            if "reward" in args.save_file:
                _, predicted = torch.max(outputs.data[:, :-1], 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    outputs_result = np.concatenate(outputs_result)
    return (100 * correct / total, outputs_result)
            
        # print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total} %')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str,
                        choices=['cifar10', 'cifar100'],
                        default='cifar10',
                        help="Select which dataset to use")
    parser.add_argument("-n", "--noise_type", type=str,
                        choices=['impulse_noise', 'shot_noise', 'defocus_blur', 'motion_blur', 'speckle_noise'],
                        default='impulse_noise',
                        help="Select which noise type to use")
    parser.add_argument("-sf", "--save_file", type=str,
                        required=True,
                        help="Specify the name of the file to load the model")
    parser.add_argument("-s", "--seed", type=int,
                        default=0,
                        help="Specify the seed to use")
    # parser.add_argument('-b', '--batch-size', default=128, type=int,
    #                 metavar='N', help='mini-batch size (default: 128)')
    """ parser.add_argument("-m", "--model", type=str,
                        choices=['resnet20'],
                        default='resnet20',
                        help="Select which model to use") """
    parser.add_argument("-t", "--temperature", action='store_true',
                        default=False,
                        help="Specify whether to use temperature outputs")
                        
    args = parser.parse_args()
    print(f"Evaluating for Noise type: {args.noise_type}")
    # For reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Get it from savename
    args.batch_size = int(args.save_file.split('_bs')[1].split('_')[0])
    args.model = args.save_file.split(f'/{args.dataset}/')[1].split('_')[0]

    main(args)
