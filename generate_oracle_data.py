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

def get_bounded_pos(dist):
    ball_loc = np.random.randint(30, 370, size = (2,))
    # Choose a ball goal at most max_dist away from the ball
    theta = np.random.rand() * 2 * np.pi
    vec = np.array([np.cos(theta), np.sin(theta)]) * dist
    ball_goal = ball_loc + vec
    while ball_goal[0] < 30 or ball_goal[0] > 370 or ball_goal[1] < 30 or ball_goal[1] > 370:
        ball_loc = np.random.randint(30, 370, size = (2,))
        # print("Tryibg again")
        theta = np.random.rand() * 2 * np.pi
        vec = np.array([np.cos(theta), np.sin(theta)]) * dist
        ball_goal = ball_loc + vec
    
    # Turn ball_goal into ints
    ball_goal = np.round(ball_goal).astype(int)

    ball_inits = [ball_loc]
    ball_goals = [ball_goal]

    return ball_inits, ball_goals


env = particlePush(render_mode = 'human')

# Create a file to store the data
f = h5py.File("oracle_data_3_buckets.hdf5", "w")

distances = range(10, 351, 10)
for dist in distances:
    max_dist = dist
    print("Generating data for distance " + str(dist))
    # Create a group for each distance
    group = f.create_group(str(dist))
    observations = np.zeros((2000, 6))
    actions = np.zeros((2000, 1))

    frames_filled = 0

    ball_inits, ball_goals = get_bounded_pos(max_dist)
    env.set_env(ball_inits = ball_inits, ball_goals = ball_goals)

    obs, _ = env.reset()
    t = 0

    while frames_filled < 2000:
        if frames_filled % 100 == 0:
            print(str(frames_filled) +" / 2000")

        action = oracle(obs, env_params = {"ball_goals": env.ball_goals}, env = env)
        t += 1

        observations[frames_filled] = obs
        actions[frames_filled] = action
        frames_filled += 1

        #action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        # Render the game
        env.render()

        if trunc:
            # If unsuccessful, remove the all the frames from the last episode from the dataset
            frames_filled -= t
        if term or trunc:
            ball_inits, ball_goals = get_bounded_pos(max_dist)
            env.set_env(ball_inits = ball_inits, ball_goals = ball_goals)
            obs, _ = env.reset()
            t = 0
    
    # Add the observations and actions to the group
    group.create_dataset("observations", data = observations)
    group.create_dataset("actions", data = actions)

f.close()
