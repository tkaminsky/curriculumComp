import h5py
from torch import nn
import torch
from torch import optim
import numpy as np
import wandb
from particle_push_env import particlePush
from collections import deque
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--order", "-o", help="The kind of dataset ordering to use", default="random")

order = parser.parse_args().order

run = wandb.init(
    project="BC-ParticlePush-curriculum_test",
    name=f'Test-{order}',
    notes="First quick trial. Max epochs 200, batch size 256, 100 test trials.",
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
            nn.Linear(512, 17),
        )

    def forward(self, x):
        # Get the raw output from the network
        output = self.linear_relu_stack(x)
        # Apply softmax to get probabilities
        output = torch.softmax(output, dim = 1)
        
        return output
    
def get_current_ds(ds, buckets):
    actions = np.zeros((len(buckets) * 2000, 1))
    observations = np.zeros((len(buckets) * 2000, 6))

    for i, bucket in enumerate(buckets):
        for j in range(2000):
            actions[i * 2000 + j] = ds[str(bucket)]["actions"][j]
            observations[i * 2000 + j] = ds[str(bucket)]["observations"][j]
    # Shuffle the dataset
    indices = np.arange(len(actions))
    np.random.shuffle(indices)
    actions = actions[indices]
    observations = observations[indices]
    return actions, observations

def get_bucket_order(method='random'):
    buckets = ds.keys()
    buckets = list(buckets)
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

# Load dataset from oracle_data_max_350_step_10.hdf5
ds = h5py.File("oracle_data_max_350_step_10.hdf5", "r")

MAX_EPOCHS = 200
B = 256

# Load common test set
test_set = np.load("test_set.npy", allow_pickle=True).item()
ball_inits = test_set["ball_inits"]
ball_goals = test_set["ball_goals"]
agent_inits = test_set["agent_inits"]
num_eval_samples = ball_inits.shape[0]

# Get the buckets in the order specified by the order variable
buckets = get_bucket_order(method=order)

# For each set of buckets
for cur_num_buckets in range(1, len(buckets)):
    current_buckets = buckets[:cur_num_buckets]
    print(f"On dataset {cur_num_buckets} / {len(buckets)}")
    actions, observations = get_current_ds(ds, current_buckets)
    curr_ds_len = len(actions)

    model = NeuralNetwork()
    if cur_num_buckets != 1:
        # Load the weights from the previous model
        model.load_state_dict(torch.load(f"model_{order}_{cur_num_buckets - 1}.pt"))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    loss_buffer = deque(maxlen=50)
    converged = False

    # While less than MAX_EPOCHS have passed and the loss is still decreasing
    for epoch in range(MAX_EPOCHS):
        # Shuffle the dataset
        indices = np.arange(curr_ds_len)
        np.random.shuffle(indices)
        actions = actions[indices]
        observations = observations[indices]
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} / {MAX_EPOCHS}")
        for iter in range(int(np.round(curr_ds_len / B))):
            # Get the current batch
            if iter == int(np.round(curr_ds_len / B)) - 1:
                batch_actions = actions[iter * B :]
                batch_observations = observations[iter * B :]
                # Make batch actions one-hot over 17
                batch_actions = np.eye(17)[batch_actions.astype(int)]
                batch_actions = batch_actions.reshape((batch_observations.shape[0], 17))
            else:
                batch_actions = actions[iter * B : (iter + 1) * B]
                batch_observations = observations[iter * B : (iter + 1) * B]
                # Make batch actions one-hot over 17
                batch_actions = np.eye(17)[batch_actions.astype(int)]
                # Reshape from (B, 1, 17) to (B, 17)
                batch_actions = batch_actions.reshape((B, 17))

            # Get the output from the model
            output = model(torch.tensor(batch_observations).float())
            # Get the loss
            loss = loss_fn(output, torch.tensor(batch_actions).float())
            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Log the loss
            avg_loss = loss.item()
            wandb.log({"loss": avg_loss})

            # Add the loss to the buffer
            loss_buffer.append(avg_loss)

            # If the buffer is full, check if the loss is decreasing
            if len(loss_buffer) == 50:
                # compare the loss of the last 25 iters to the loss of the first 25 iters
                prev_loss = np.mean(list(loss_buffer)[:25])
                curr_loss = np.mean(list(loss_buffer)[25:])
                wandb.log({"loss difference": np.abs(curr_loss - prev_loss)})
                if np.abs(curr_loss - prev_loss) < 1e-7:
                    print("Model converged for current dataset")
                    wandb.log({"Converge_epochs": epoch})
                    converged = True
                    break

        if converged:
            break

    if not converged:
        print("Warning: model did not converge for current dataset")
        wandb.log({"Converge_epochs": MAX_EPOCHS})
    
    
    # Empty the loss buffer
    loss_buffer = deque(maxlen=50)

    print("Evaluating model")
    
    # Evaluate the model
    num_correct = 0
    model.eval()
    with torch.no_grad():
        for i in range(num_eval_samples):
            if i % 10 == 0:
                print(f"Sample {i} / {num_eval_samples}")
            # Create the environment
            env = particlePush(render_mode = 'None')
            env.set_env(agent_init = agent_inits[i], ball_inits = [ball_inits[i]], ball_goals = [ball_goals[i]])
            obs, _ = env.reset()
            # Run the environment
            while True:
                # Get the action from the model
                obs = torch.tensor(obs.reshape((1, 6))).float()

                action = model(obs)
                action = torch.argmax(action).item()
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
        print(f"Success rate: {num_correct / num_eval_samples}")
    # Save the model
    model.train()
    torch.save(model.state_dict(), f"model_{order}_{cur_num_buckets}.pt")
