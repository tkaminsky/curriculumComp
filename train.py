import h5py
from torch import nn
import torch
from torch import optim
import numpy as np
import wandb
from particle_push_env import particlePush
from collections import deque
import argparse

# GLOBAL PARAMETERS
MAX_EPOCHS = 100
B = 256

MAX_ITERS = 150_000
MAX_ITERS = 500_000

parser = argparse.ArgumentParser()
parser.add_argument("--order", "-o", help="The kind of dataset ordering to use", default="random")
parser.add_argument("--dataset", "-d", help="The dataset to use", default="oracle_data_max_350_step_10")
# parser.add_argument("--use_curriculum", "-c", help="Indicates whether to use the curriculum at all.", default=False, type=bool)

order = parser.parse_args().order
ds_file = parser.parse_args().dataset
# use_curriculum = parser.parse_args().use_curriculum
# use_curriculum = False
use_curriculum = True
run = wandb.init(
    project="BC-ParticlePush-curriculum-3-buckets",
    name=f'Sm_Test-{order}-{ds_file}',
    # name=f"hard-supervised-{ds_file}",
    notes="Full dataset, larger network, fewer epochs.",
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 17),
        )

    def forward(self, x):
        # Get the raw output from the network
        output = self.linear_relu_stack(x)
        # Apply softmax to get probabilities
        output = torch.softmax(output, dim = 1)
        
        return output
    

def get_bucket_sizes(ds, buckets):
    bucket_sizes = np.zeros(len(buckets), dtype=np.int32)
    for i, bucket in enumerate(buckets):
        bucket_sizes[i] = len(ds[bucket]["actions"])
    return bucket_sizes

def get_current_ds(ds, buckets):
    bucket_sizes = get_bucket_sizes(ds, buckets)
    actions = np.zeros((sum(bucket_sizes), 1))
    observations = np.zeros((sum(bucket_sizes), 6))

    for i, bucket in enumerate(buckets):
        for j in range(bucket_sizes[i]):
            actions[sum(bucket_sizes[:i]) + j] = ds[str(bucket)]["actions"][j]
            observations[sum(bucket_sizes[:i]) + j] = ds[str(bucket)]["observations"][j]
    # Shuffle the dataset
    indices = np.arange(len(actions))
    np.random.shuffle(indices)
    actions = actions[indices]
    observations = observations[indices]
    return actions, observations

def get_bucket_order(method='random'):
    buckets = ds.keys()
    if ds_file == "oracle_data_max_350_step_10":
        buckets = [10 * i for i in range(1, 36)]
    else:
        buckets = ['easy', 'medium', 'hard']
    print(f"Buckets: {buckets}")
    if method == 'sequential':
        return buckets
    elif method == 'reversed':
        return buckets[::-1]
    elif method == 'random':
        np.random.shuffle(buckets)
        return buckets
    else:
        print("Invalid method: " + method)
        print("Valid methods are: 'sequential', 'reversed', 'random'")
        return None

# Load dataset from desired file
ds = h5py.File(f"{ds_file}.hdf5", "r")

test_ds = h5py.File(f"oracle_data_3_buckets_test.hdf5", "r")

# Load common test set
test_set = np.load("test_set.npy", allow_pickle=True).item()
ball_inits = test_set["ball_inits"]
ball_goals = test_set["ball_goals"]
agent_inits = test_set["agent_inits"]
# num_eval_samples = ball_inits.shape[0]
num_eval_samples = 100

# Get the buckets in the order specified by the order variable
buckets = get_bucket_order(method=order)

model = NeuralNetwork()
model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

iter_sched = [10_000, 10_000, 10_000, MAX_ITERS - 150_000]

iter_count = 0
bucket_count = 0

if use_curriculum:
    # For each set of buckets
    for cur_num_buckets in range(1, len(buckets) + 1):
        bucket_count += 1
        current_buckets = buckets[:cur_num_buckets]

        print(f"On dataset {cur_num_buckets} / {len(buckets)}")
        print("Using buckets: " + str(current_buckets))
        actions, observations = get_current_ds(ds, current_buckets)
        curr_ds_len = len(actions)
        print(f"Training on {curr_ds_len} samples")


        while iter_count < sum(iter_sched[:bucket_count]):
            # Shuffle the dataset
            indices = np.arange(curr_ds_len)
            np.random.shuffle(indices)
            actions = actions[indices]
            observations = observations[indices]

            for iter in range(int(np.round(curr_ds_len / B))):
                
                if iter_count % 2_000 == 0:
                    print(f"Iteration {iter_count} / {iter_sched[bucket_count]}")

                    # Evaluate the model
                    num_correct = 0
                    model.eval()
                    with torch.no_grad():
                        for i in range(num_eval_samples):
                            if i % 10 == 0:
                                print(f"Sample {i} / {num_eval_samples}")
                            # Create the environment
                            env = particlePush(render_mode = 'human')
                            env.set_env(agent_init = agent_inits[i], ball_inits = [ball_inits[i]], ball_goals = [ball_goals[i]])
                            obs, _ = env.reset()
                            # Run the environment
                            while True:
                                # Get the action from the model
                                obs = torch.tensor(obs).unsqueeze(0).float()

                                action_arr = model(obs)

                                action = torch.argmax(action_arr).item()


                                # Take the action
                                obs, _, term, trunc, _ = env.step(action)
                                env.render()
                                # If the ball has reached the goal, break
                                if term:
                                    num_correct += 1
                                    break
                                if trunc:
                                    break
                        # Log the percentage of successful runs in wandb vs the number of buckets
                        wandb.log({"success_rate": num_correct / num_eval_samples, "num_buckets": cur_num_buckets})
                        print(f"Success rate: {num_correct / num_eval_samples}\n")
                        # env.close()
                        # Save the model
                    model.train()

                # Get the current batch
                if iter == int(np.round(curr_ds_len / B)) - 1:
                    batch_actions = actions[iter * B :].squeeze( axis = 1)
                    batch_observations = observations[iter * B :]
                    batch_actions = torch.nn.functional.one_hot(torch.tensor(batch_actions).long(), num_classes=17).float()
                else:
                    batch_actions = actions[iter * B : (iter + 1) * B].squeeze( axis = 1)
                    batch_observations = observations[iter * B : (iter + 1) * B]
                    batch_actions = torch.nn.functional.one_hot(torch.tensor(batch_actions).long(), num_classes=17).float()

                # Get the output from the model
                output = model(torch.tensor(batch_observations).float())

                # Calculate the action accuracy
                with torch.no_grad():
                    action_accuracy = torch.sum(torch.argmax(output, dim=1) == torch.argmax(batch_actions, dim=1)).item() / B
                    wandb.log({"action_accuracy": action_accuracy})

                # Get the loss
                loss = loss_fn(output, batch_actions)

                iter_count += 1

                # Backpropagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Log the loss
                avg_loss = loss.item()
                wandb.log({"loss": avg_loss})


        print("Evaluating model")
        
        # Evaluate the model
        num_correct = 0
        model.eval()
        with torch.no_grad():
            for i in range(num_eval_samples):
                if i % 10 == 0:
                    print(f"Sample {i} / {num_eval_samples}")
                # Create the environment
                env = particlePush(render_mode = 'human')
                env.set_env(agent_init = agent_inits[i], ball_inits = [ball_inits[i]], ball_goals = [ball_goals[i]])
                obs, _ = env.reset()
                # Run the environment
                while True:
                    # Get the action from the model
                    obs = torch.tensor(obs).unsqueeze(0).float()

                    action_arr = model(obs)

                    action = torch.argmax(action_arr).item()


                    # Take the action
                    obs, _, term, trunc, _ = env.step(action)
                    env.render()
                    # If the ball has reached the goal, break
                    if term:
                        num_correct += 1
                        break
                    if trunc:
                        break
            # Log the percentage of successful runs in wandb vs the number of buckets
            wandb.log({"success_rate": num_correct / num_eval_samples, "num_buckets": cur_num_buckets})
            print(f"Success rate: {num_correct / num_eval_samples}\n")
            # env.close()
            # Save the model
            model.train()
            torch.save(model.state_dict(), f"model_{order}_{ds_file}_short.pt")
else:
    print("Not using a curriculum")
    # For each set of buckets
    for cur_num_buckets in range(1, len(buckets) + 1):
        bucket_count += 1
        # current_buckets = buckets[:cur_num_buckets]
        # current_buckets = [buckets]
        current_buckets = ['hard']

        print(f"On dataset {cur_num_buckets} / {len(buckets)}")
        print("Using buckets: " + str(current_buckets))
        actions, observations = get_current_ds(ds, current_buckets)
        curr_ds_len = len(actions)
        print(f"Training on {curr_ds_len} samples")


        while iter_count < sum(iter_sched[:bucket_count]):
            # Shuffle the dataset
            indices = np.arange(curr_ds_len)
            np.random.shuffle(indices)
            actions = actions[indices]
            observations = observations[indices]

            for iter in range(int(np.round(curr_ds_len / B))):
                
                if iter_count % 25_000 == 0:
                    print(f"Iteration {iter_count} / {iter_sched[bucket_count]}")

                    # Evaluate the model
                    num_correct = 0
                    model.eval()
                    with torch.no_grad():
                        for i in range(num_eval_samples):
                            if i % 10 == 0:
                                print(f"Sample {i} / {num_eval_samples}")
                            # Create the environment
                            env = particlePush(render_mode = 'human')
                            env.set_env(agent_init = agent_inits[i], ball_inits = [ball_inits[i]], ball_goals = [ball_goals[i]])
                            obs, _ = env.reset()
                            # Run the environment
                            while True:
                                # Get the action from the model
                                obs = torch.tensor(obs).unsqueeze(0).float()

                                action_arr = model(obs)

                                action = torch.argmax(action_arr).item()


                                # Take the action
                                obs, _, term, trunc, _ = env.step(action)
                                env.render()
                                # If the ball has reached the goal, break
                                if term:
                                    num_correct += 1
                                    break
                                if trunc:
                                    break
                        # Log the percentage of successful runs in wandb vs the number of buckets
                        wandb.log({"success_rate": num_correct / num_eval_samples, "num_buckets": cur_num_buckets})
                        print(f"Success rate: {num_correct / num_eval_samples}\n")
                        # env.close()
                        # Save the model
                    model.train()

                # Get the current batch
                if iter == int(np.round(curr_ds_len / B)) - 1:
                    batch_actions = actions[iter * B :].squeeze( axis = 1)
                    batch_observations = observations[iter * B :]
                    batch_actions = torch.nn.functional.one_hot(torch.tensor(batch_actions).long(), num_classes=17).float()
                else:
                    batch_actions = actions[iter * B : (iter + 1) * B].squeeze( axis = 1)
                    batch_observations = observations[iter * B : (iter + 1) * B]
                    batch_actions = torch.nn.functional.one_hot(torch.tensor(batch_actions).long(), num_classes=17).float()

                # Get the output from the model
                output = model(torch.tensor(batch_observations).float())

                # Calculate the action accuracy
                with torch.no_grad():
                    action_accuracy = torch.sum(torch.argmax(output, dim=1) == torch.argmax(batch_actions, dim=1)).item() / B
                    wandb.log({"action_accuracy": action_accuracy})

                # Get the loss
                loss = loss_fn(output, batch_actions)

                iter_count += 1

                # Backpropagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Log the loss
                avg_loss = loss.item()
                wandb.log({"loss": avg_loss})


        print("Evaluating model")

        # # Evaluate the model on the test set
        # test_actions, test_observations = get_current_ds(test_ds, current_buckets)
        # test_ds_len = len(test_actions)

        # print(f"Evaluating on {test_ds_len} samples")

        # model.eval()
        # with torch.no_grad():
        #     # Run the model on the test set
        #     test_losses = np.zeros(int(np.round(test_ds_len / B)))
        #     test_accs = np.zeros(int(np.round(test_ds_len / B)))
        #     for iter in range(int(np.round(test_ds_len / B))):
        #         # Get the current batch
        #         if iter == int(np.round(test_ds_len / B)) - 1:
        #             batch_actions = test_actions[iter * B :].squeeze( axis = 1)
        #             batch_observations = test_observations[iter * B :]
        #             batch_actions = torch.nn.functional.one_hot(torch.tensor(batch_actions).long(), num_classes=17).float()
        #         else:
        #             batch_actions = test_actions[iter * B : (iter + 1) * B].squeeze( axis = 1)
        #             batch_observations = test_observations[iter * B : (iter + 1) * B]
        #             batch_actions = torch.nn.functional.one_hot(torch.tensor(batch_actions).long(), num_classes=17).float()

        #         # Get the output from the model
        #         output = model(torch.tensor(batch_observations).float())

        #         # Calculate the action accuracy
        #         with torch.no_grad():
        #             action_accuracy = torch.sum(torch.argmax(output, dim=1) == torch.argmax(batch_actions, dim=1)).item() / B
        #             wandb.log({"test_accuracy": action_accuracy})
        #         # Get the loss
        #         loss = loss_fn(output, batch_actions)

        #         test_accs[iter] = torch.sum(torch.argmax(output, dim=1) == torch.argmax(batch_actions, dim=1)).item() / B

        #         # Log the loss
        #         avg_loss = loss.item()
        #         test_losses[iter] = avg_loss
        #     wandb.log({"test_loss": np.average(test_losses), "test_acc": np.average(test_accs)})
        
        # Evaluate the model
        num_correct = 0
        model.eval()
        with torch.no_grad():
            for i in range(num_eval_samples):
                if i % 10 == 0:
                    print(f"Sample {i} / {num_eval_samples}")
                # Create the environment
                env = particlePush(render_mode = 'human')
                env.set_env(agent_init = agent_inits[i], ball_inits = [ball_inits[i]], ball_goals = [ball_goals[i]])
                obs, _ = env.reset()
                # Run the environment
                while True:
                    # Get the action from the model
                    obs = torch.tensor(obs).unsqueeze(0).float()

                    action_arr = model(obs)

                    action = torch.argmax(action_arr).item()


                    # Take the action
                    obs, _, term, trunc, _ = env.step(action)
                    env.render()
                    # If the ball has reached the goal, break
                    if term:
                        num_correct += 1
                        break
                    if trunc:
                        break
            # Log the percentage of successful runs in wandb vs the number of buckets
            wandb.log({"success_rate": num_correct / num_eval_samples, "num_buckets": cur_num_buckets})
            print(f"Success rate: {num_correct / num_eval_samples}\n")
            # env.close()
            # Save the model
            model.train()
            torch.save(model.state_dict(), f"model_{order}_{ds_file}.pt")