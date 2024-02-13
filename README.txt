the models folder contains all results from experiments as model checkpoints. These checkpoints include trajectories.

each task can be run using main.py

below are the available arguments
--lr
The initial learning rate of the optimizer
--epochs
the number of epochs to train for
--prune_ratio, --pr"
The pruning ratio, used for magnitude, lottery, and early_bird modes
--batch_size
the training and test batch size
--checkpoints
list of epochs to create checkpoints at
--mode
The prunning mode.
Options are vanilla, magnitude, lottery, and early_bird
--freq
frequency to print statistics
--gamma
amount to scale learning rate by
--momentum
SGD momentum
--weight-decay, --wd
Optimizer learning rate
--pre_path, --pp
Path to the fully trained model to be pruned
--init_path, --ip
Path to the checkpoint containing the initial weights of the fully trained model
--prune_iterations
number of training iterations
--schedule
list of epochs to scale learning rate by gamma
--depth
depth of blocks in the resnet model

For all experiments, all arguments were left as default except for pre_path, init_path, prune_ratio, prune_iterations and mode

with mode vanilla set (default mode), no parameters need to be specified
with mode set to magnitude (task 2), pre_path must be specified and prune_ratio set to desired integer percentage (default is 0)
with mode set to lottery (task 3), pre_path and init_path must be specified. prune_ratio should be set to desired integer percentage and prune_iterations should be set to 2
with mode set to early_bird (task 4), pre_path and init_path must be specified. prune_ratio should be set to desired integer percentage

all graphs are printed usings stats.py

This implementation also utilizes the preresnet20 architecture from https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/cifar/network-slimming