'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import pickle
import math
from sklearn.metrics import roc_auc_score

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Normalize

import os
import argparse
import time

from model import get_model
from data import get_data, make_planeloader
from utils import get_loss_function, get_scheduler, get_random_images, produce_plot, get_noisy_images, AttackPGD
from evaluation import train, test, test_on_trainset, decision_boundary
from options import options
from utils import simple_lapsed_time
from utils import produce_plot_alt

def renyi_entropy(probabilities, alpha):
    # If alpha equals 1, the formula diverges; handle as a special case.
    if alpha == 1:
        raise ValueError("Alpha should not be equal to 1")

    # Compute the sum of probabilities raised to the power of alpha
    sum_p_alpha = sum([p ** alpha for p in probabilities.values()])

    # Calculate Rényi entropy using the formula
    renyi_entropy_value = (1 / (1 - alpha)) * math.log(sum_p_alpha)

    return renyi_entropy_value

def plot(net_name, load_path, plot_path, testloader, normalize_transform):

    print('###############################')
    print(net_name)
    print(load_path)
    print(plot_path)
    start = time.time()
    args.net = net_name
    net = get_model(args, device)
    if torch.cuda.device_count() > 1:
        net.module.load_state_dict(torch.load(load_path))
    else:
        net.load_state_dict(torch.load(load_path))

    # test_acc, predicted = test(args, net, testloader, device)
    # print(test_acc)
    end = time.time()
    simple_lapsed_time("Time taken to train/load the model", end - start)


    start = time.time()


    sum_probs = {}
    ats_sum = 0.0
    for _ in range(args.num_plots):
        if args.imgs is None:
            # images, labels = get_random_images(trainloader.dataset)
            images, labels, ids = get_random_images(testloader.dataset)
        elif -1 in args.imgs:
            dummy_imgs, _ = get_random_images(testloader.dataset)
            images, labels = get_noisy_images(torch.stack(dummy_imgs), testloader.dataset, net, device)
        elif -10 in args.imgs:
            image_ids = args.imgs[0]
            #         import ipdb; ipdb.set_trace()
            images = [testloader.dataset[image_ids][0]]
            labels = [testloader.dataset[image_ids][1]]
            for i in list(range(2)):
                temp = torch.zeros_like(images[0])
                if i == 0:
                    temp[0, 0, 0] = 1
                else:
                    temp[0, -1, -1] = 1

                images.append(temp)
                labels.append(0)

        #         dummy_imgs, _ = get_random_images(testloader.dataset)
        #         images, labels = get_noisy_images(torch.stack(dummy_imgs), testloader.dataset, net, device)
        # incomplete
        else:
            image_ids = args.imgs
            images = [testloader.dataset[i][0] for i in image_ids]
            labels = [testloader.dataset[i][1] for i in image_ids]
            print(labels)

        # if args.adv:
        #     adv_net = AttackPGD(net, trainloader.dataset)
        #     adv_preds, imgs = adv_net(torch.stack(images).to(device), torch.tensor(labels).to(device))
        #     images = [img.cpu() for img in imgs]

        planeloader = make_planeloader(images, args)
        preds = decision_boundary(args, net, planeloader, device)

        # sampl_path = '_'.join(list(map(str, args.imgs)))
        args.plot_path = plot_path
        val_counts = produce_plot_alt(args.plot_path, preds, planeloader, images, labels, normalize_transform, temp=args.temp)
        # print(f"val_counts: {val_counts}")

        total_count = sum(val_counts.values())

        ats_sum_counts = sum(val_counts.get(label, 0) for label in labels if val_counts.get(label, 0) < total_count * 0.5)

        # Calculate the total count to normalize the probabilities

        # Compute the probabilities from the value counts
        probabilities = {label: count / total_count for label, count in val_counts.items()}

        for label, prob in probabilities.items():
            if label in sum_probs:
                sum_probs[label] += prob
            else:
                sum_probs[label] = prob

        ats = ats_sum_counts / sum(val_counts.values())

        # print(f"ats: {ats}")
        ats_sum += ats

    avg_probs = {}
    for label, prob in sum_probs.items():
        avg_probs[label] = sum_probs[label] / args.num_plots

    print(f"load_path: {load_path}")

    print(f"avg_probs: {avg_probs}")

    renyi_entropy_of_avg_probs = renyi_entropy(avg_probs, args.alpha)
    print(f"renyi_entropy_of_avg_probs: {renyi_entropy_of_avg_probs}")

    ats_avg = ats_sum / args.num_plots
    print(f"ats_avg: {ats_avg}")

    end = time.time()
    simple_lapsed_time("Time taken to plot the image", end - start)

    return {
        "renyi_entropy_of_avg_probs": renyi_entropy_of_avg_probs,
        "ats_avg": ats_avg
    }

def calculate_overall_auc(args):
    # Initialize dictionaries to store all predictions and ground truth labels across models
    all_predictions = {
        "renyi_entropy_of_avg_probs": [],
        "ats_avg": []
    }
    all_ground_truth = []

    print(f"args.load_path: {args.load_path}")
    # Loop through each folder in the specified load path
    for root, dirs, _ in os.walk(args.load_path):
        for dir_name in dirs:
            model_folder_path = os.path.join(root, dir_name)
            model_file_path = os.path.join(model_folder_path, 'model.pt')
            metadata_file_path = os.path.join(model_folder_path, 'metadata.pt')

            # Check if both the model and metadata files exist
            if os.path.exists(model_file_path) and os.path.exists(metadata_file_path):
                testloader, normalize_transform = create_test_loader(model_folder_path)

                metadata = torch.load(metadata_file_path)
                # args.num_classes = metadata.get('num_classes')
                dataset_name = metadata["config"]["dataset"]
                if dataset_name in ["cifar10", "mnist", "fmnist"]:
                    args.num_classes = 10
                elif dataset_name == "cifar100":
                    args.num_classes = 100
                    continue
                elif dataset_name == "gtsrb":
                    args.num_classes = 43
                    continue
                elif dataset_name == "tiny":
                    args.num_classes = 200
                elif dataset_name == "imagenet-30":
                    args.num_classes = 10
                elif dataset_name == "pubfig":
                    args.num_classes = 10

                print(f"args.num_classes: {args.num_classes}")

                # Call the plot function for the current model
                plot_result = plot(args.net, model_file_path, args.plot_path, testloader, normalize_transform)

                # Load ground truth from metadata.pt
                ground_truth = metadata.get('ground_truth')
                print(f"ground_truth: {ground_truth}")

                if "target_class" in metadata["config"].keys():
                    target_class = metadata["config"]["target_class"]
                    print(f"target_class: {target_class}")

                # Accumulate ground truth and predictions for each metric
                if ground_truth:
                    all_ground_truth.append(1)
                else:
                    all_ground_truth.append(0)

                for key in all_predictions:
                    all_predictions[key].append(plot_result[key])  # Collect predictions for each metric

                print()
            else:
                if not model_folder_path.endswith("test_dataset"):
                    print(f"Model or metadata file not found in {model_folder_path}")

    # Calculate AUC for each metric using the accumulated predictions and ground truth
    overall_auc = {}
    for key, predictions in all_predictions.items():
        auc = roc_auc_score(all_ground_truth, predictions)
        overall_auc[key] = auc

    print(f"overall_auc: {overall_auc}")
    return overall_auc

#args.plot_path
parser = argparse.ArgumentParser(description='Argparser for sanity check')

parser.add_argument('--net', default='ResNet', type=str)
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--plot_path', type=str, default=None)
parser.add_argument('--bs', type=int, default=None)
parser.add_argument('--num_plots', type=int, default=20)
parser.add_argument('--baseset', default='CIFAR10', type=str,
                            choices=['CIFAR10', 'CIFAR100','SVHN',
                            'CIFAR100_label_noise'])
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--imgs', default=None,
                        type=lambda s: [int(item) for item in s.split(',')])
parser.add_argument('--temp', default=1.0, type=float)
parser.add_argument('--alpha', default=10.0, type=float)
parser.add_argument('--range_l', default=0.5, type=float)
parser.add_argument('--range_r', default=0.5, type=float)
parser.add_argument('--plot_method', default='greys', type=str)
parser.add_argument('--resolution', default=500, type=float, help='resolution for plot')
parser.add_argument('--adv', action='store_true', help='Adversarially attack images?')

args = parser.parse_args()
print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)

        # Load the image
        image = Image.open(img_path).convert('RGB')

        # Extract the label from the filename
        label = int(img_name.split('_')[1].split('.')[0])

        # Apply the transformation if provided
        if self.transform:
            image = self.transform(image)

        return image, label

def create_test_loader(model_folder_path):
    # Get the directory of the test dataset from args.load_path
    test_data_dir = os.path.join(model_folder_path, 'test_dataset')

    # Load the metadata to extract the transformation
    metadata_path = os.path.join(model_folder_path, 'metadata.pt')
    metadata = torch.load(metadata_path)
    transform = metadata["config"]["transform"]
    normalize_transform = next((t for t in transform.transforms if isinstance(t, Normalize)), None)

    # Create the test dataset and data loader
    test_dataset = CustomDataset(data_dir=test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)

    return test_loader, normalize_transform

calculate_overall_auc(args)


# Archs = ['ResNet', 'VGG' , 'GoogLeNet' , 'DenseNet' , 'MobileNet']
# all_models_path = './saved_models/'
# all_final_plot_path = './saved_final_imgs'

# load_path = all_models_path + '/soft_distillation/from_' + from_method + '/' + arch + '_cifar10.pth'
# plot_path = all_final_plot_path + '/soft_distillation/' + arch + '/' + 'from_' + from_method

# for arch in Archs:
#     #net_name, load_path, plot_path
#     print('########################################################################')
#     net_name = arch
#
#     for originals in ['naive', 'mixup', 'cutmix']:
#         load_path = all_models_path + originals + '/' + arch + '_cifar10.pth'
#         plot_path = all_final_plot_path + '/soft_distillation/' + arch + '/' + originals
#         plot(net_name, load_path, plot_path)
#
#
#     for from_arch in Archs:
#         load_path = all_models_path + '/soft_distillation/from_' + from_arch + '/' + arch + '_cifar10.pth'
#         plot_path = all_final_plot_path  + '/soft_distillation/' + arch + '/' + 'from_' + from_arch
#         plot(net_name, load_path, plot_path)
#
#     for from_method in ['cutmix', 'mixup']:
#         load_path = all_models_path + '/soft_distillation/from_' + from_method + '/' + arch + '_cifar10.pth'
#         plot_path = all_final_plot_path + '/soft_distillation/' + arch + '/' + 'from_' + from_method
#         plot(net_name, load_path, plot_path)

