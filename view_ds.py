import h5py
from particle_push_env import particlePush
import numpy as np

f = h5py.File("oracle_data_3_buckets_500_runs.hdf5", "r")

print(f.keys())



hard = f['hard']

obs_total = hard['observations']
actions = hard['actions']


env = particlePush(render_mode = 'human')

obs_0 = (obs_total[0]) * 200 + 200
agent_init = obs_0[0:2]
ball_init = obs_0[2:4]
ball_goal = obs_0[4:6]

env.set_env(agent_init=agent_init, ball_inits=[ball_init], ball_goals=[ball_goal])
obs, _ = env.reset()

print(obs)
print(obs_total[0])

print("STARTING!!!")

counter = 0

while True:
    action = actions[counter][0]
    print(action)
    counter += 1
    obs, reward, term, trunc, info = env.step(action)
    env.render()

    if term or trunc:
        print(f"Term: {term}, Trunc: {trunc}")
        print("Run ended")
        obs_new = obs_total[counter] * 200 + 200
        agent_init = obs_new[0:2]
        ball_init = obs_new[2:4]
        ball_goal = obs_new[4:6]
        env.set_env(agent_init=agent_init, ball_inits=[ball_init], ball_goals=[ball_goal])
        obs, _ = env.reset()

    # Check if the two vectors are equal
    if not np.array_equal(obs[4:6], ball_goal):
        print("Goal changed")




