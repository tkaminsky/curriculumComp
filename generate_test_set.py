import numpy as np

# Generate test ball/goal positions for evaluation
num_eval_samples = 1000
ball_inits = np.random.randint(low=30, high=370, size=(num_eval_samples, 2))
ball_goals = np.random.randint(low=30, high=370, size=(num_eval_samples, 2))
agent_inits = np.random.randint(low=10, high=390, size=(num_eval_samples, 2))


# Save the dataset as a dictionary
test_set = {}
test_set["ball_inits"] = ball_inits
test_set["ball_goals"] = ball_goals
test_set["agent_inits"] = agent_inits

np.save("test_set.npy", test_set)