# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
import time

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("CartPole-v1", render_mode="human")

target_reward = 80
achieved = False
episode = 0

while not achieved:
    episode += 1
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward  += reward
        done = terminated or truncated

        time.sleep(0.02)

    print(f'Episódio {episode} terminou com recompensa: {total_reward}')

    if total_reward >= target_reward:
        achieved = True
        print(f'Conseguiu atingir {target_reward} pontos no episódio {episode}!')

env.close()
