import argparse
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
from tqdm import tqdm
from Resnet20 import resnet
import torch.nn.functional as F
import copy
import numpy as np
from channel_selection import channel_selection

def checkpoint(model, optimizer, PATH, trajectory = None):
    print("checkpoint")
    if trajectory:
        print(f"epoch: {trajectory[-1][0]} loss: {trajectory[-1][1]} accuracy: {trajectory[-1][2]}")
        checkpoint = {
        'epoch': trajectory[-1][0],
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'trajectory': trajectory,
    } 
    else:
        checkpoint = {
        'epoch': 0,
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'trajectory': None,
    } 

    torch.save(checkpoint, PATH)
    return checkpoint


def print_model(model):
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.BatchNorm2d) or isinstance(mod, channel_selection) or isinstance(mod, nn.Linear):
            print(mod)
            
        


def train(model, criterion, optimizer, train_loader, mask = None):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for imgs, targets in train_loader:
       
        optimizer.zero_grad()

        imgs = imgs.to(device)
        targets = targets.to(device)
        logits = model(imgs)

        loss = criterion(logits, targets)
        loss.backward()

        if mask:
            freeze_grads(model, mask)

        optimizer.step()
    return loss.item()

def test(model, test_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            logits = model(imgs)
            
            loss += F.cross_entropy(logits, targets, reduction='sum').item()
            preds = logits.max(1)[1]
            correct += preds.eq(targets.view_as(preds)).sum().item()
        loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
    return loss, accuracy

def prune_weights(model, ref_model, percent):
    mask = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for module, ref_module in zip(model.modules(), ref_model.modules()):
            if isinstance(module, nn.Conv2d):
                tensor = ref_module.weight.cpu().numpy()
                alive = tensor[np.nonzero(tensor)]
                percentile_value = np.percentile(abs(alive), percent)
                mask.append(torch.from_numpy(np.where(abs(tensor) < percentile_value, 0, 1)).to(device))

                module.weight.mul_(mask[-1])
    return mask

#refrence:https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/cifar/network-slimming/resprune.py
def prune_channels(init_model, ref_model, percent):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bn = torch.tensor((0,)).to(device)

    #find percentile value for batchnorm
    with torch.no_grad():
        for module in ref_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                bn = torch.cat((bn, module.weight))
        bn = bn[1:]
        percentile_value = np.percentile(abs(bn.cpu().numpy()), percent)
        

        #build channel config/config_mask
        config = []
        config_mask = []
        for init_module, ref_module in zip(init_model.modules(), ref_model.modules()):
            if isinstance(ref_module, nn.BatchNorm2d):
                weight = ref_module.weight.abs()
                mask = weight.gt(percentile_value).float().to(device)
                init_module.weight.mul_(mask)
                init_module.bias.mul_(mask)

                config.append(int(torch.sum(mask)))
                config_mask.append(mask.clone())
    print(percentile_value)
        
    #init model with correct shape
    pruned_model = resnet(args.depth, cfg=config)
    pruned_model.to(device)

    

    in_channels = torch.ones(3)
    out_channels = config_mask[0]

    init_modules = list(init_model.modules())
    pruned_modules = list(pruned_model.modules())
    print(config)


    config_layer = 0
    conv_count = 0

    #map non pruned channels from initialized weights to channels in the slimmed model 
    for layer_id in range(len(init_modules)):
        with torch.no_grad():
            if isinstance(init_modules[layer_id], nn.BatchNorm2d):
                #get indecies of non pruned channels
                out_channel_idxs =  np.squeeze(np.argwhere(np.asarray(out_channels.cpu().numpy())))
                if out_channel_idxs.size == 1:
                    out_channel_idxs = np.resize(out_channel_idxs, (1,))

                #map weights of first batch norm in block. never gets pruned
                if isinstance(init_modules[layer_id + 1], channel_selection):
                    pruned_modules[layer_id].weight = torch.nn.Parameter(init_modules[layer_id].weight.clone())
                    pruned_modules[layer_id].bias = torch.nn.Parameter(init_modules[layer_id].bias.clone())
                    pruned_modules[layer_id].running_mean = init_modules[layer_id].running_mean.clone()
                    pruned_modules[layer_id].running_var = init_modules[layer_id].running_var.clone()
                    
                    #set channel selector layer
                    pruned_modules[layer_id + 1].indexes.zero_()
                    pruned_modules[layer_id + 1].indexes[out_channel_idxs.tolist()] = 1.0
                
                #map batch norm that follows a conv2d
                else:
                    pruned_modules[layer_id].weight = torch.nn.Parameter(init_modules[layer_id].weight[out_channel_idxs.tolist()].clone())
                    pruned_modules[layer_id].bias = torch.nn.Parameter(init_modules[layer_id].bias[out_channel_idxs.tolist()].clone())
                    pruned_modules[layer_id].running_mean = init_modules[layer_id].running_mean[out_channel_idxs.tolist()].clone()
                    pruned_modules[layer_id].running_var = init_modules[layer_id].running_var[out_channel_idxs.tolist()].clone()
                    
                #shift channel selector mask
                config_layer += 1
                in_channels = out_channels.clone()
                if config_layer < len(config_mask):
                    out_channels = config_mask[config_layer]


            elif isinstance(init_modules[layer_id], nn.Conv2d):
                #if conv1
                if conv_count == 0:
                    pruned_modules[layer_id].weight = torch.nn.Parameter(init_modules[layer_id].weight.clone())
                    conv_count += 1

                #if in residual block
                elif isinstance(init_modules[layer_id - 1], channel_selection) or isinstance(init_modules[layer_id - 1], nn.BatchNorm2d):
                    conv_count += 1
                    in_channel_idxs = np.squeeze(np.argwhere(np.asarray(in_channels.cpu().numpy())))
                    out_channel_idxs = np.squeeze(np.argwhere(np.asarray(out_channels.cpu().numpy())))
                    if in_channel_idxs.size == 1:
                        in_channel_idxs = np.resize(in_channel_idxs, (1,))
                    if out_channel_idxs.size == 1:
                        out_channel_idxs = np.resize(out_channel_idxs, (1,))
                    
                 
                    pruned_modules[layer_id].weight = torch.nn.Parameter(init_modules[layer_id].weight[:, in_channel_idxs.tolist(), :, :].clone())
                    
                    #if not last layer in residual block
                    if conv_count % 3 != 1:

                        pruned_modules[layer_id].weight = torch.nn.Parameter(pruned_modules[layer_id].weight[out_channel_idxs.tolist(), :, :, :].clone())
                
                #if downsample
                else:
                    pruned_modules[layer_id].weight = torch.nn.Parameter(init_modules[layer_id].weight.clone())
            
            #fc layer
            elif isinstance(init_modules[layer_id], nn.Linear):
                in_channel_idxs = np.squeeze(np.argwhere(np.asarray(in_channels.cpu().numpy())))
                if in_channel_idxs.size == 1:
                    in_channel_idxs = np.resize(in_channel_idxs, (1,))
                
                pruned_modules[layer_id].weight = torch.nn.Parameter(init_modules[layer_id].weight[:, in_channel_idxs.tolist()].clone())
                pruned_modules[layer_id].bias = torch.nn.Parameter(init_modules[layer_id].bias.clone())

    return pruned_model

def freeze_grads(model, mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.grad.mul_(mask[idx])
                idx += 1

def run_test(args):
    init_model = resnet(args.depth)
    ref_model = resnet(args.depth)
    prune_channels(init_model, ref_model, 50)
def main(args):
    model = resnet(args.depth)
    ref_model = resnet(args.depth)
    transform_train = transforms.Compose([ transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    traindataset = datasets.CIFAR10('data', train=True, download=True,transform=transform_train)
    testdataset = datasets.CIFAR10('data', train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=False)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model.to(device)
    ref_model.to(device)

    checkpoints = []
    trajectory = []
    checkpoint_idx = []

    pretrained_checkpoint = None
    initial_checkpoint = None
    if args.pre_path:
        pretrained_checkpoint = torch.load(args.pre_path)
    if args.init_path:
        initial_checkpoint = torch.load(args.init_path)

    mask = None
    if args.mode == 'vanilla' and pretrained_checkpoint:
        model.load_state_dict(pretrained_checkpoint['model_state_dict'])
        optimizer.load_state_dict(pretrained_checkpoint['optimizer_state_dict'])
        trajectory = pretrained_checkpoint['trajectory']
        
    if args.mode == "magnitude":
        print("magnitude")
        model.load_state_dict(pretrained_checkpoint['model_state_dict'])
        ref_model.load_state_dict(pretrained_checkpoint['model_state_dict'])
    
        mask = prune_weights(model, ref_model, args.prune_ratio) 
        pretrained_epoch = pretrained_checkpoint['trajectory'][-1][0]
        print(pretrained_checkpoint['trajectory'])

    if args.mode == 'lottery':
        model.load_state_dict(initial_checkpoint['model_state_dict'])
        ref_model.load_state_dict(pretrained_checkpoint['model_state_dict'])

        mask = prune_weights(model, ref_model, args.prune_ratio) 
        pretrained_epoch = pretrained_checkpoint['trajectory'][-1][0]
        print(pretrained_checkpoint['trajectory'])

    if args.mode == 'early_bird':
        init_model = resnet(29)
        init_model.to(device)
        init_model.load_state_dict(initial_checkpoint['model_state_dict'])
        ref_model.load_state_dict(pretrained_checkpoint['model_state_dict'])

        model = prune_channels(init_model, ref_model, args.prune_ratio)
        pretrained_epoch = pretrained_checkpoint['trajectory'][-1][0]
        print(pretrained_checkpoint['trajectory'])

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #initial_state_checkpoint = checkpoint(model, optimizer, "models/initial_checkpoint.pt")

    if args.checkpoints:
        checkpoint_idx = args.checkpoints

    if args.epochs not in checkpoint_idx:
        checkpoint_idx.append(args.epochs)
    
    for prune_iteration in range(args.prune_iterations):

        if prune_iteration != 0:
            new_model = resnet(29)
            new_model.load_state_dict(initial_checkpoint['model_state_dict'])
            new_model.to(device)
            mask = prune_weights(new_model, model, args.prune_ratio)

            model = new_model
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


        for i in tqdm(range(args.start_epoch, args.epochs)):
            loss = train(model, criterion, optimizer, train_loader, mask)
            test_loss, test_accuracy = test(model, test_loader)
            if i in args.schedule:
                print("lr update")
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.gamma 

            trajectory.append((i + 1, test_loss, test_accuracy))

            if i + 1 in checkpoint_idx:
                
                if args.mode == 'vanilla':
                    checkpoints.append(checkpoint(model, optimizer, f"models/{args.mode}/{args.mode}_{i + 1}.pt", trajectory))
                elif args.mode == "lottery":
                    checkpoints.append(checkpoint(model, optimizer, f"models/{args.mode}/{args.mode}_{prune_iteration}_{args.prune_ratio}_{i + 1}.pt", trajectory))
                else:
                    checkpoints.append(checkpoint(model, optimizer, f"models/{args.mode}/{args.mode}_pre{pretrained_epoch}_{i + 1}_{args.prune_ratio}.pt", trajectory))

            elif i % args.freq == 0:
                print(f"epoch: {i + 1} loss: {test_loss} accuracy: {test_accuracy}")
        

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=.1)
parser.add_argument("--epochs", type=int, default=160)
parser.add_argument("--prune_ratio", "--pr", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--checkpoints", type=int, nargs="*", default= [50, 100])
parser.add_argument("--mode", type=str, default="vanilla")
parser.add_argument("--freq", type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
parser.add_argument('--pre_path','--pp' , type=str, default=None)
parser.add_argument('--init_path', '--ip', type=str, default=None)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--prune_iterations', '--pi', type=int, default=1)
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120])
parser.add_argument('--depth', type=int, default=29)



args = parser.parse_args()


main(args) 
# run_test(args)   