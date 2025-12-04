import gymnasium as gym
import mlflow
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

# Iniciar MLflow e definir experimento
mlflow.set_experiment("CartPole-RL")

with mlflow.start_run(run_name="ppo_cartpole_run"):
    # Parâmetros
    timesteps = 100_000   # aumentei para treinar mais
    eval_episodes = 50    # número de episódios de avaliação
    mlflow.log_param("algorithm", "PPO")
    mlflow.log_param("timesteps", timesteps)
    mlflow.log_param("eval_episodes", eval_episodes)

    # Criar ambiente e modelo
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)

    # Treinar agente
    model.learn(total_timesteps=timesteps)

    # Avaliar agente
    test_env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(eval_episodes):
        obs, info = test_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)

    avg_reward = sum(rewards) / len(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)

    # Logar métricas no MLflow
    mlflow.log_metric("avg_reward", avg_reward)
    mlflow.log_metric("max_reward", max_reward)
    mlflow.log_metric("min_reward", min_reward)

    # Salvar modelo como artefato
    model.save("ppo_cartpole")
    mlflow.log_artifact("ppo_cartpole.zip")

# Parte visual: animação do agente treinado
env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()


# Save .mp4
env = gym.make('CartPole-v1', render_mode='rgb_array')
env = RecordVideo(env, video_folder='./video', name_prefix='cartpole')

obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
