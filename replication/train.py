import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# argparse
import argparse

from models.resNet import ResNet, ResidualBlock
from models.git_models import resnet20 as git_resnet20
from models.cait import CaiT
from models.coatnet import CoAtNet

def main(args):
    # Set seed
    
    def model_checkpoint(args):
        # Save the model checkpoint
        path = "./data/"+args.dataset+"/"
        adv = ""
        if args.adverserial_training:
            adv = "_adv"
        save_name = f"{path}{args.model}{adv}_s{args.seed}_ep{args.num_epochs}_lr{args.learning_rate}_bs{args.batch_size}_wd{args.weight_decay}_mo{args.momentum}_loss{args.loss}/"
        # Create the directory if it does not exist
        if not os.path.exists(save_name):
            print("Creating directory", save_name)
            os.makedirs(save_name)
        if args.save_file == 'model':
            fn = f"{save_name}model.pt"
        else:
            fn = path+args.save_file+'.pt'
        torch.save(model.state_dict(), fn)

        # Save the training and validation results
        np.save(f"{save_name}metrics_train.npy", train_res)
        np.save(f"{save_name}metrics_test.npy", test_res)

        # Print the configuration
        with open(f"{save_name}config.txt", "w") as f:
            f.write(str(args))
            f.write(f"\nTime: {datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
    
    def load_checkpoint(args, model):
        path = "./data/"+args.dataset+"/"
        adv = ""
        if args.adverserial_training:
            adv = "_adv"
        save_name = f"{path}{args.model}{adv}_s{args.seed}_ep{args.num_epochs}_lr{args.learning_rate}_bs{args.batch_size}_wd{args.weight_decay}_mo{args.momentum}_loss{args.loss}_dr{args.dropout}/"
        if args.save_file == 'model':
            fn = f"{save_name}model.pt"
        else:
            fn = path+args.save_file+'.pt'
            
        if os.path.exists(fn):
            model.load_state_dict(torch.load(fn))
            return model
        
        return
        
        

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        print("Using CUDA")
        torch.cuda.manual_seed_all(args.seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Image preprocessing modules
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize(mean=mean, std=std)
        ])

    if args.dataset == 'cifar10':
        # CIFAR-10 dataset
        train_dataset = torchvision.datasets.CIFAR10(
            root='../datasets/pytorch_resnet_cifar10/data/',
            train=True,
            transform=transform,
            download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            root='../datasets/pytorch_resnet_cifar10/data/',
            train=False,
            # transform=transforms.ToTensor()
            transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            )

    # Data loader
    num_workers = 2
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=num_workers,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=num_workers,
                                            shuffle=False)

    # Hyper-parameters
    # num_epochs = 200
    # learning_rate = 0.1
    # weight_decay = 0.0001
    # momentum = 0.9

    # Loss and optimizer
    if args.loss == 'cross_entropy':
        output_classes = 10
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)
    elif args.loss == 'reward':
        output_classes = 11
        def rewardLoss(outputs, labels):
            labels = labels.to(device)
            correct = nn.functional.one_hot(labels, num_classes=11-1)
            # Replace all zeros with -4
            correct[correct == 0] = -4
            # Add an extra column of 0 for abstain
            abstain = torch.zeros(correct.size(0), 1).to(device)
            targets = torch.cat((correct, abstain), dim=1)
            return nn.functional.mse_loss(outputs, targets)
        criterion = rewardLoss
    elif args.loss == 'mse':
        output_classes = 10
        criterion = nn.MSELoss()
    elif args.loss == 'mml':
        output_classes = 10
        criterion = nn.MultiMarginLoss(margin = 0.5)

    
    if args.model == 'resnet20':
        model = ResNet(ResidualBlock, [3, 3, 3], output_classes).to(device)
    elif args.model == 'gitcifar10':
        print("Using gitcifar10")
        model = git_resnet20()
    elif args.model == 'cait':
        model = CaiT(image_size = 32, patch_size = 4,
                    num_classes = output_classes, dim = 328,
                    depth = 5,   # depth of transformer for patch to patch attention only
                    cls_depth=2, # depth of cross attention of CLS tokens to patch
                    heads = 6, mlp_dim = 240, dropout = args.dropout, #runnning for new experiment
                    emb_dropout = 0.1, layer_dropout = 0.05)
        
        model.to(device)
        print("Number of parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        args.optimizer = 'adamw'
        args.scheduler = 'exponentiallr'
        
        lr_max = 5e-4
        T = 150
        
        model_load = load_checkpoint(args, model)
        if model_load:
            model = model_load
            model.train()
                    
    elif args.model == 'coatnet':
        # If using coatnet, set optimizer to adamw and scheduler to onecycle
        args.optimizer = 'adamw'
        args.scheduler = 'onecycle'
        D, L, channels = [64, 64, 128, 256, 512], [1, 1, 2, 2, 2], 3
        model = CoAtNet((32, 32), output_classes, L, D, channels, strides = [1, 1, 2, 2, 4], dropout=0.3)
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        model.to(device)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer == 'adamw':
        no_decay = []
        decay = []
        for name, param in model.named_parameters():
            if 'weight' in name and 'batchnorm' and 'layernorm' and 'relative_bias_table' not in name:
                decay.append(param)
            else:
                no_decay.append(param)


        optimizer = torch.optim.AdamW([
            {'params': decay, 'weight_decay': args.weight_decay},
            {'params': no_decay, 'weight_decay': 0}
        ], lr=1e-6,)
    # elif args.optimizer == 'batchNorm':
        

    # Learning rate scheduler
    if args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    elif args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                        steps_per_epoch=len(train_loader), epochs=args.num_epochs)
    elif args.scheduler == 'exponentiallr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

    # Train the model
    total_step = len(train_loader)
    # Ensure four print statements per epoch, including the last one
    print_every = 200
    train_res = []
    test_res = []
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    for epoch in range(args.num_epochs):
        model.train()
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            images.requires_grad = True

            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)

            if args.loss == 'mml':
                outputs = outputs.softmax(dim=1)
                loss = criterion(outputs, labels)
            else: 
                loss = criterion(outputs, labels)

            if args.adverserial_training:
                (loss/2).backward(retain_graph=True)
            else: 
                loss.backward()

            if args.adverserial_training:
                denorm_images = images * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
                data_grad = images.grad.data
                perturbed_data = denorm_images + 0.01*data_grad.sign()
                perturbed_data = torch.clamp(perturbed_data, 0, 1)
                perturbed_data = transforms.Normalize(mean=mean, std=std)(perturbed_data)

                outputs_adv = model(perturbed_data)
                adv_loss = criterion(outputs_adv, labels)
                (adv_loss/2).backward()
                loss = adv_loss/2 + loss/2

            # Backward and optimize
            if args.model == 'cait':
                for g in optimizer.param_groups:
                    g['lr'] = (epoch+1)/T * lr_max
            optimizer.step()
            if args.loss == 'reward':
                _, predicted = torch.max(outputs[:, :-1].data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if args.adverserial_training:
                if args.loss == 'reward':
                    _, predicted = torch.max(outputs_adv[:, :-1].data, 1)
                else:
                    _, predicted = torch.max(outputs_adv.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            if (i) % print_every == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} - Acc {:.4f}"
                    .format(epoch+1, args.num_epochs, i+1, total_step, loss.item(), 100 * correct / total))
        acc_tr = 100 * correct / total

        # Decay learning rate
        if args.model != 'cait':
            scheduler.step()
        else:
            if epoch>=T:
                scheduler.step()
            
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            correct_test = 0
            total_test = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                if args.loss == 'reward':
                    loss_test = criterion(outputs, labels)
                    _, predicted = torch.max(outputs[:, :-1].data, 1)
                elif args.loss == 'mml':
                    outputs = outputs.softmax(dim=1)
                    loss_test = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    loss_test = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        # Log the validation loss and accuracy
        train_res.append((loss.item(), acc_tr))
        test_res.append((loss_test.item(), 100 * correct_test / total_test))
        
        if (epoch+1)%100 and args.model=='cait':
            model_checkpoint(args)
        
        print ("Epoch [{}/{}] Loss: {:.4f} - Acc {:.4f} - Val Loss {:.4f} - Val Acc {:.4f}".format(epoch+1, args.num_epochs, loss.item(), 100 * correct / total, loss_test.item(), 100 * correct_test / total_test))
        
    
    model_checkpoint(args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str,
                        choices=['cifar10', 'cifar100'],
                        default='cifar10',
                        help="Select which dataset to use")
    parser.add_argument("-dr", "--dropout", type=float,
                        default=0.1,
                        help="Select dropout value")
    parser.add_argument("-m", "--model", type=str,
                        choices=['resnet20', 'gitcifar10', 'cait', 'coatnet'],
                        default='resnet20',
                        help="Select which model to use")
    parser.add_argument("-s", "--seed", type=int,
                        default=0,
                        help="Select which seed to use")
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=0.001,
                        help="Select which learning rate to use")
    parser.add_argument("-sch", "--scheduler", type=str,
                        choices=['multistep','onecycle'],
                        default='multistep',
                        help="Select which scheduler to use")
    parser.add_argument("-op", "--optimizer", type=str,
                        choices=['SGD','adamw', 'batchNorm'],
                        default='SGD',
                        help="Select which optimizers to use")
    parser.add_argument("-bs", "--batch_size", type=int,
                        choices=[64, 128, 256],
                        default=128,
                        help="Select which batch size to use")
    parser.add_argument("-ep", "--num_epochs", type=int,
                        default=200,
                        help="Select which number of epochs to use")
    parser.add_argument("-wd", "--weight_decay", type=float,
                        default=1e-4,
                        help="Select which weight decay to use with SGD")
    parser.add_argument("-mo", "--momentum", type=float,
                        default=0.9,
                        help="Select which momentum to use with SGD")
    parser.add_argument("-l", "--loss", type=str,
                        choices=['cross_entropy','reward','mse','mml'],
                        default='cross_entropy',
                        help="Select which loss function to use")
    parser.add_argument("-sf", "--save_file", type=str,
                        default='model',
                        help="Specify the name of the file to save the model")
    parser.add_argument("-adv", "--adverserial_training", action='store_true',
                        default=False,
                        help="Use adverserial training")
    args = parser.parse_args()


    
    main(args)
