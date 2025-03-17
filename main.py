import arenax_sai
import numpy as np
from agent import agent_class
import torch
from nn_models import neural_network_model

# Selecting a competition
sai = arenax_sai.SAIClient(competition_id="2QRh5otC6PhD")

# Creating an agent and an environment
env = sai.make_env()
agent = agent_class()

# Maximum episodes and maximum steps in each episode
num_episodes = 50
max_steps = 100

# Learning Loop
for episode in range(num_episodes):
    
    obs_1 = env.reset()

    obs_1 = np.transpose(obs_1[0], (2, 0, 1))

    episode_reward = 0

    for step in range(max_steps):

        action = agent.act(obs_1)
        state_tensor = torch.tensor(obs_1, dtype = torch.float32).unsqueeze(0)
        agent.visit_state(state_tensor)
        obs_2 , reward , done , *extras = env.step(action)
        obs_2 = np.transpose(obs_2, (2, 0, 1))
        agent.remember(obs_1 , action , reward , obs_2 , done)
        agent.train_step()
        obs_1 = obs_2
        #env.render()
        if done:
            break
            
    print(f"Episode {episode+1}")

trained_Q_network = neural_network_model(input_channels=13 , action_dim=10)
torch.save(trained_Q_network, "trained_Q_network_full.pt")

env.close()

torch.save(trained_Q_network.state_dict(), "trained_Q_network_amir.pt")

final_submit_model = torch.jit.trace(trained_Q_network, torch.randn(1, 13, 13, 19))

sai.save_model(
    name="sai_amir", 
    model=final_submit_model
)

submission = sai.submit_model(
    name="sai_amir",
    model=final_submit_model
)