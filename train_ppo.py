import gymnasium as gym
import mlflow
from stable_baselines3 import PPO
import argparse
from datetime import datetime

def train_ppo_cartpole(timesteps=100_000, eval_episodes=50, run_name=None):
    """
    Treina um modelo PPO no CartPole e registra no MLflow
    
    Args:
        timesteps: Número de timesteps de treinamento
        eval_episodes: Número de episódios para avaliação
        run_name: Nome do experimento (opcional)
    """
    
    # Iniciar MLflow e definir experimento
    mlflow.set_experiment("CartPole-RL")
    
    # Gerar nome automático se não fornecido
    if run_name is None:
        run_name = f"ppo_cartpole_{timesteps}steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Registrar parâmetros
        mlflow.log_param("algorithm", "PPO")
        mlflow.log_param("timesteps", timesteps)
        mlflow.log_param("eval_episodes", eval_episodes)
        mlflow.log_param("environment", "CartPole-v1")
        
        print(f"\n{'='*60}")
        print(f"Treinando: {run_name}")
        print(f"Timesteps: {timesteps}")
        print(f"Episódios de avaliação: {eval_episodes}")
        print(f"{'='*60}\n")
        
        # Criar ambiente e modelo
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=1)
        
        # Treinar agente
        model.learn(total_timesteps=timesteps)
        
        # Avaliar agente
        test_env = gym.make("CartPole-v1")
        rewards = []
        for episode in range(eval_episodes):
            obs, info = test_env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, info = test_env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
            if (episode + 1) % 10 == 0:
                print(f"Avaliação: {episode + 1}/{eval_episodes} - Recompensa: {total_reward}")
        
        avg_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)
        
        # Logar métricas no MLflow
        mlflow.log_metric("avg_reward", avg_reward)
        mlflow.log_metric("max_reward", max_reward)
        mlflow.log_metric("min_reward", min_reward)
        
        # Salvar modelo (salvo automaticamente pelo MLflow em artifacts)
        
        # Exibir resultados
        print(f"\n{'='*60}")
        print(f"Treinamento Concluído!")
        print(f"Recompensa Média: {avg_reward:.2f}")
        print(f"Recompensa Máxima: {max_reward:.2f}")
        print(f"Recompensa Mínima: {min_reward:.2f}")
        print(f"{'='*60}\n")
        
        return avg_reward, max_reward, min_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar PPO Agent para CartPole")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Número de timesteps de treinamento")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Número de episódios de avaliação")
    parser.add_argument("--name", type=str, help="Nome do experimento")
    
    args = parser.parse_args()
    
    train_ppo_cartpole(
        timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        run_name=args.name
    )
