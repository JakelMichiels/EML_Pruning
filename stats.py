import torch
import torchvision.models as models
from Resnet20 import resnet
import torch.nn as nn
import matplotlib.pyplot as plt


vanilla = torch.load("models/vanilla/vanilla_160.pt", map_location=torch.device('cpu'))

mag_trajectories = {
    "Unpruned": vanilla['trajectory'],
    "50%": torch.load("models/magnitude/magnitude_pre160_160_50.pt", map_location=torch.device('cpu'))['trajectory'],
    "70%": torch.load("models/magnitude/magnitude_pre160_160_70.pt", map_location=torch.device('cpu'))['trajectory'],
    "90%": torch.load("models/magnitude/magnitude_pre160_160_90.pt", map_location=torch.device('cpu'))['trajectory'],
}

lot_trajectories = {
    "Unpruned": vanilla['trajectory'],
    "50%": torch.load("models/lottery/good/lottery_1_29_160.pt", map_location=torch.device('cpu'))['trajectory'],
    "70%": torch.load("models/lottery/good/lottery_1_45_160.pt", map_location=torch.device('cpu'))['trajectory'],
    "90%": torch.load("models/lottery/good/lottery_1_68_160.pt", map_location=torch.device('cpu'))['trajectory'],
}

# lot_trajectories = {
#     "Unpruned": vanilla['trajectory'],
#     "50%": torch.load("models/lottery/old/lottery_1_50_160.pt", map_location=torch.device('cpu'))['trajectory'],
#     "70%": torch.load("models/lottery/old/lottery_1_70_160.pt", map_location=torch.device('cpu'))['trajectory'],
#     #"90%": torch.load("models/lottery/old/lottery_1_90_160.pt", map_location=torch.device('cpu'))['trajectory'],
# }

early_trajectories = {
    "Unpruned": vanilla['trajectory'],
    "10" : {
        "30%": torch.load("models/early_bird/early_bird_pre10_160_30.pt", map_location=torch.device('cpu'))['trajectory'],
        "50%": torch.load("models/early_bird/early_bird_pre10_160_50.pt", map_location=torch.device('cpu'))['trajectory'],
    },
    "30" : {
        "30%": torch.load("models/early_bird/early_bird_pre30_160_30.pt", map_location=torch.device('cpu'))['trajectory'],
        "50%": torch.load("models/early_bird/early_bird_pre30_160_50.pt", map_location=torch.device('cpu'))['trajectory'],
    },
    "50" : {
        "30%": torch.load("models/early_bird/early_bird_pre50_160_30.pt", map_location=torch.device('cpu'))['trajectory'],
        "50%": torch.load("models/early_bird/early_bird_pre50_160_50.pt", map_location=torch.device('cpu'))['trajectory'],
    },
    "100" : {
        "30%": torch.load("models/early_bird/early_bird_pre100_160_30.pt", map_location=torch.device('cpu'))['trajectory'],
        "50%": torch.load("models/early_bird/early_bird_pre100_160_50.pt", map_location=torch.device('cpu'))['trajectory'],
    },
}


def task2(trajectories):
    fig, axs = plt.subplots(4)
    fig.suptitle('Percent of Parameters Pruned using Layer-wise Unstructured Pruning')

    for name, trajectory in trajectories.items():
        x_coords = [point[0] for point in trajectory]
        accuracies = [point[2] for point in trajectory]
        losses = [point[1] for point in trajectory]

        print(f"{name}")
        print(f"Initial Accuracy: {accuracies[0]} Final Accuracy: {accuracies[-1]}")
        

        if name == "Unpruned":
            axs[0].plot(x_coords, accuracies, label=name, linestyle='--')
            axs[1].plot(x_coords, losses, label=name, linestyle='--')
        else:
            axs[0].plot(x_coords, accuracies, label=name)
            axs[1].plot(x_coords, losses, label=name)

    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('loss')

    plt.show()

def task3(trajectories):
    fig, axs = plt.subplots(2)
    fig.suptitle('Percent of Parameters Pruned using Layer-wise Unstructured Pruning')

    for name, trajectory in trajectories.items():
        if name == "Unpruned":
            x_coords = [point[0] for point in trajectory]
            accuracies = [point[2] for point in trajectory]
            axs[0].plot(x_coords, accuracies, label=name, linestyle='--')
            axs[1].plot(x_coords, accuracies, label=name, linestyle='--')
            print(f"{name}")
            print(f"Initial Accuracy: {accuracies[0]} Final Accuracy: {accuracies[-1]}")
        
        else:
            x_coords_1 = [point[0] for point in trajectory[:160]]
            x_coords_2 = [point[0] for point in trajectory[160:]]
            accuracies_1 = [point[2] for point in trajectory[:160]]
            accuracies_2 = [point[2] for point in trajectory[160:]]

            axs[0].plot(x_coords, accuracies_1, label=name)
            axs[1].plot(x_coords, accuracies_2, label=name)
            print(f"{name}")
            print(f"Initial Accuracy: {accuracies_1[0]} Iteration 1 Final Accuracy: {accuracies_1[-1]} Iteration 2 Final Accuracy: {accuracies_2[-1]}")  

    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].set_title('Iteration 1')

    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Iteration 2')

    plt.show()

def task4(trajectories):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Percent of Parameters Pruned using Network Slimming')

    axs = [ax for tup in axs for ax in tup]

    x_coords = [point[0] for point in trajectories["Unpruned"]]
    accuracies = [point[2] for point in trajectories["Unpruned"]]
    for ax in axs:
        ax.plot(x_coords, accuracies, label="Unpruned", linestyle='--')
        
    print(f"Unpruned")
    print(f"Initial Accuracy: {accuracies[0]} Final Accuracy: {accuracies[-1]}")
    
    del trajectories["Unpruned"]

    for idx, dict_ in enumerate(trajectories.items()):
        epochs, ticket = dict_
        print(epochs)
        for name, trajectory in ticket.items():
            x_coords = [point[0] for point in trajectory]
           
            accuracies = [point[2] for point in trajectory]
            axs[idx].plot(x_coords, accuracies, label=name)
            axs[idx].set_title(f"{epochs} Epoch Ticket")

            
            print(f"{name}")
            print(f"Initial Accuracy: {accuracies[0]} Final Accuracy: {accuracies[-1]}")  

    axs[2].set_ylabel('Accuracy')
    axs[1].legend()
    axs[2].set_xlabel('Epochs')

    plt.show()

    
task2(mag_trajectories)
task3(lot_trajectories)
task4(early_trajectories)






