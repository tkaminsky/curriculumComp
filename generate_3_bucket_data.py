import numpy as np
from particle_push_env import particlePush
import h5py

# Requires access to the environment parameters and the environment to function
def oracle(state, env_params, env):
    # Says whether each ball is in its goal
    ball_status = env.get_info()
    # Find the smallest index of a ball that isn't in its goal
    goal_ball_idx = np.argmin(ball_status)
    # Find the line between the ball and the goal
    state = state.reshape((2 * env.num_balls + 1, 2))
    agent_state = state[0]
    ball_states = state[1:env.num_balls + 1]
    ball_goals = state[env.num_balls + 1:]

    ball_state = ball_states[goal_ball_idx]
    ball_goal = env_params["ball_goals"][goal_ball_idx] - np.array([env.w/2, env.h/2])
    ball_goal = ball_goal / np.array([env.w/2, env.h/2])

    ball_goal = np.array(ball_goal)
    ball_state = np.array(ball_state)
    agent_state = np.array(agent_state)

    # Find the vector from the ball to the goal
    goal_to_ball = ball_state - ball_goal
    btg_length = np.linalg.norm(goal_to_ball)
    # Scale the vector to have length 1
    goal_to_ball = goal_to_ball / btg_length

    # Find the vector from the agent to the ball
    ball_to_agent = agent_state - ball_state
    atb_length = np.linalg.norm(ball_to_agent)
    # Scale the vector to have length 1
    ball_to_agent = ball_to_agent / atb_length

    atg_length = np.linalg.norm(ball_goal - agent_state)

    min_l = (env.agent_size + env.ball_sizes[goal_ball_idx] + 50) / (env.w)

    # If the distance between the agent and the ball is greater than 30, move towards the ball
    if atb_length > min_l:
        action = -ball_to_agent

    # # If the dot product of the two vectors isn't 1
    elif np.dot(goal_to_ball, ball_to_agent) < .98:
        tangent = np.array([-ball_to_agent[1], ball_to_agent[0]])
        agent_state_moved = agent_state + (tangent / env.w)

        new_dot_pos = np.dot(goal_to_ball, (agent_state_moved - ball_state) / np.linalg.norm(agent_state_moved - ball_state))
        agent_state_moved_neg = agent_state - (tangent / env.w)
        new_dot_neg = np.dot(goal_to_ball, (agent_state_moved_neg - ball_state) / np.linalg.norm(agent_state_moved_neg - ball_state))
        # Move the agent in the direction with the greater dot_product
        if new_dot_pos > new_dot_neg:
            action = tangent
        else:
            action = -tangent
    else:
        action = -ball_to_agent

    # Find the action in the action space closest to the action
    action_options = range(env.action_space.n)
    action_vectors = [env.action_map[i] for i in action_options]
    action_vectors = np.array(action_vectors)
    
    action = np.array(action)
    action = action / np.linalg.norm(action)
    action = np.dot(action_vectors, action)
    action = np.argmax(action)
    
    return action

env = particlePush(render_mode = 'human')

ds_size = 50
num_buckets = 3
max_run_len = 500


# Create a file to store the data
f = h5py.File(f"oracle_data_{num_buckets}_buckets_{ds_size}_runs_test.hdf5", "w")

observations = {}
actions = {}
run_lengths = np.zeros(ds_size, dtype=np.int32)

# For each run, create random initial conditions and run the oracle
for i in range(ds_size):
    if i % 10 == 0:
        print(i)
    ball_inits = np.random.randint(30, 370, size=(2,))
    ball_goals = np.random.randint(30, 370, size=(2,))
    agent_init = np.random.randint(10, 390, size=(2,))

    env.set_env(agent_init = agent_init, ball_inits = [ball_inits], ball_goals = [ball_goals])

    obs, _ = env.reset()
    t = 0

    actions_now = np.zeros((max_run_len, 1))
    observations_now = np.zeros((max_run_len, 6))

    while True:

        action = oracle(obs, env_params = {"ball_goals": env.ball_goals}, env = env)

        # Add the observation and action to the dataset
        observations_now[t] = obs
        actions_now[t] = action

        t += 1

        #action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        # Render the game
        env.render()

        if term or trunc:
            break

    # Crop observations and actions to the correct size
    observations_now = observations_now[:t]
    actions_now = actions_now[:t]

    # Add the data to the dictionary
    observations[i] = observations_now
    actions[i] = actions_now
    run_lengths[i] = t

# Sort the data by run length
sort_idx = np.argsort(run_lengths)
# run_lengths = run_lengths[sort_idx]
print(sort_idx)
print(run_lengths)


total_len = 0
for idx in sort_idx:
    observations[idx] = np.array(observations[idx])
    actions[idx] = np.array(actions[idx])
    total_len += observations[idx].shape[0]

# Contains all of the data in order
observations_total = np.zeros((total_len, 6))
actions_total = np.zeros((total_len, 1))

print(f"The total length of the dataset is {total_len}")

# Add all the data to a vector
for i in range(len(sort_idx)):
    # The start position of the current run
    idx_in_total = sum([run_lengths[sort_idx[v]] for v in range(i)])
    # The length of the current run to be added
    len_ep = run_lengths[sort_idx[i]]
    # Add the data to the total dataset
    observations_total[idx_in_total:idx_in_total + len_ep] = observations[sort_idx[i]]
    actions_total[idx_in_total:idx_in_total + len_ep] = actions[sort_idx[i]]

# Buckets will be evenly split
bucket_sizes = [total_len // num_buckets] * num_buckets
bucket_sizes[-1] += total_len % num_buckets

# Create the datasets for each bucket
for i, bucket in enumerate(['easy', 'medium', 'hard']):
    group = f.create_group(bucket)
    group.create_dataset("observations", (bucket_sizes[i], 6), dtype=np.float32)
    group.create_dataset("actions", (bucket_sizes[i], 1), dtype=np.int32)
    
    # Add corresponding 1/3 of the data to the bucket
    group['observations'][:] = observations_total[sum(bucket_sizes[:i]):sum(bucket_sizes[:i+1])]
    group['actions'][:] = actions_total[sum(bucket_sizes[:i]):sum(bucket_sizes[:i+1])]


# Print the last element of observations_total to make sure it's correct
print(observations_total[-1])


# Check that the buckets have the right sizes
for bucket in ['easy', 'medium', 'hard']:
    print(f"Bucket {bucket} has size {f[bucket]['observations'].shape[0]}")

# Save the data
f.close()


# Plot a histogram of the run lengths
import matplotlib.pyplot as plt
plt.hist(run_lengths)

plt.show()

# Save the plot

plt.savefig("run_lengths.png")