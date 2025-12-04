import gymnasium as gym
import mlflow
from stable_baselines3 import PPO
import argparse
from datetime import datetime
import cv2
import numpy as np
import os
from pathlib import Path

def train_ppo_cartpole(timesteps=100_000, eval_episodes=50, run_name=None):
    """
    Treina um modelo PPO no CartPole e registra no MLflow
    
    Args:
        timesteps: Número de timesteps de treinamento
        eval_episodes: Número de episódios para avaliação
        run_name: Nome do experimento (opcional)
    
    Returns:
        Tupla (modelo, métricas, run_id)
    """
    
    # Iniciar MLflow e definir experimento
    mlflow.set_experiment("CartPole-RL")

    # Gerar nome automático se não fornecido
    if run_name is None:
        run_name = f"ppo_cartpole_{timesteps}steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as active_run:
        run_id = active_run.info.run_id

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
        model = PPO("MlpPolicy", env, verbose=0)

        # Treinar agente (normalmente chamaremos por checkpoints fora desta função)
        model.learn(total_timesteps=timesteps)

        # Avaliar agente
        test_env = gym.make("CartPole-v1")
        rewards = []
        for episode in range(eval_episodes):
            obs, info = test_env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
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

        # Exibir resultados
        print(f"\n{'='*60}")
        print(f"Treinamento Concluído!")
        print(f"Recompensa Média: {avg_reward:.2f}")
        print(f"Recompensa Máxima: {max_reward:.2f}")
        print(f"Recompensa Mínima: {min_reward:.2f}")
        print(f"{'='*60}\n")

        return model, {
            'run_name': run_name,
            'timesteps': timesteps,
            'avg_reward': avg_reward,
            'max_reward': max_reward,
            'min_reward': min_reward
        }, run_id

def record_agent_frames(model, steps=500):
    """
    Grava frames de um agente em ação
    
    Args:
        model: Modelo PPO treinado
        steps: Número de passos para gravar
    
    Returns:
        Lista de frames (frames numpy arrays)
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames = []
    
    obs, info = env.reset()
    done = False
    step = 0
    
    while step < steps and not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        step += 1
    
    env.close()
    return frames

def create_comparison_video(agents_data, output_path="comparison_agents.mp4", fps=30):
    """
    Cria vídeo comparando múltiplos agentes lado a lado
    
    Args:
        agents_data: Lista de dicts com 'model', 'name', 'metrics'
        output_path: Caminho do vídeo de saída
        fps: Frames por segundo
    """
    print(f"\n{'='*60}")
    print("Gravando vídeos dos agentes...")
    print(f"{'='*60}\n")
    
    # Gravar frames de cada agente
    all_frames = []
    for agent in agents_data:
        print(f"Gravando: {agent['name']}")
        frames = record_agent_frames(agent['model'], steps=500)
        all_frames.append(frames)
        print(f"✓ {len(frames)} frames capturados")
    
    # Encontrar tamanho do frame
    frame_height, frame_width = all_frames[0][0].shape[:2]
    
    # Dimensões do vídeo final (4 agentes lado a lado, 2x2)
    num_agents = len(agents_data)
    if num_agents == 1:
        final_width = frame_width
        final_height = frame_height
        grid_cols = 1
        grid_rows = 1
    elif num_agents == 2:
        final_width = frame_width * 2 + 20
        final_height = frame_height + 100
        grid_cols = 2
        grid_rows = 1
    else:
        final_width = frame_width * 2 + 20
        final_height = frame_height * 2 + 120
        grid_cols = 2
        grid_rows = 2
    
    # Criar VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (final_width, final_height))
    
    # Número máximo de frames
    max_frames = max(len(frames) for frames in all_frames)
    
    print(f"\nCriando vídeo comparativo...")
    
    # Processar frame por frame
    for frame_idx in range(max_frames):
        # Criar imagem final
        canvas = np.ones((final_height, final_width, 3), dtype=np.uint8) * 240
        
        # Processar cada agente
        for agent_idx, agent in enumerate(agents_data):
            frames = all_frames[agent_idx]
            
            # Pegar frame (se disponível)
            if frame_idx < len(frames):
                frame = frames[frame_idx]
            else:
                frame = frames[-1] if frames else np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 128
            
            # Adicionar informações no frame
            frame_copy = frame.copy()
            
            # Nome do agente
            cv2.putText(frame_copy, f"{agent['name']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Métrica de recompensa
            metrics = agent['metrics']
            reward_text = f"Avg: {metrics['avg_reward']:.1f}"
            cv2.putText(frame_copy, reward_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Calcular posição no grid
            row = agent_idx // grid_cols
            col = agent_idx % grid_cols
            
            y_start = row * (frame_height + 60)
            x_start = col * (frame_width + 10)
            
            # Colar frame no canvas
            canvas[y_start:y_start+frame_height, x_start:x_start+frame_width] = frame_copy
        
        # Adicionar título
        title = "CartPole Agents Comparison"
        cv2.putText(canvas, title, (final_width//2 - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Escrever frame
        out.write(canvas)
    
    out.release()
    print(f"\n✅ Vídeo salvo: {output_path}")
    print(f"{'='*60}\n")

def train_multiple_agents(configs):
    """
    Treina múltiplos agentes com diferentes configurações
    
    Args:
        configs: Lista de dicts com 'timesteps', 'name' (opcional)
    
    Returns:
        Lista de agents com modelos e métricas
    """
    agents = []
    
    print(f"\n{'='*80}")
    print("TREINANDO MÚLTIPLOS AGENTES")
    print(f"{'='*80}\n")
    
    for idx, config in enumerate(configs, 1):
        print(f"\n[{idx}/{len(configs)}] Iniciando treinamento...")
        print("-"*80)
        
        timesteps = config.get('timesteps', 100_000)
        name = config.get('name', f"Agent_{timesteps}steps")
        
        model, metrics, run_id = train_ppo_cartpole(
            timesteps=timesteps,
            eval_episodes=config.get('eval_episodes', 50),
            run_name=name
        )
        
        agents.append({
            'model': model,
            'name': name,
            'timesteps': timesteps,
            'metrics': metrics,
            'run_id': run_id
        })
    
    return agents


def train_with_checkpoints(timesteps=100_000, checkpoint=10_000, eval_episodes=5, video_steps=500, run_name=None, target_reward=490):
    """
    Treina um único agente em checkpoints, avalia em cada checkpoint, grava vídeo por checkpoint
    e gera um vídeo final mostrando a evolução.
    """
    mlflow.set_experiment("CartPole-RL")
    if run_name is None:
        run_name = f"ppo_evolution_{timesteps}steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as active_run:
        run_id = active_run.info.run_id
        mlflow.log_param("algorithm", "PPO")
        mlflow.log_param("total_timesteps", timesteps)
        mlflow.log_param("checkpoint", checkpoint)
        mlflow.log_param("eval_episodes", eval_episodes)

        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)

        videos_dir = Path("videos/evolution")
        videos_dir.mkdir(parents=True, exist_ok=True)

        checkpoints = list(range(checkpoint, timesteps + 1, checkpoint))
        solved_at = None

        for ck in checkpoints:
            print(f"\n-- Training until {ck} timesteps (increment {checkpoint}) --")
            model.learn(total_timesteps=checkpoint)

            # Evaluate
            eval_env = gym.make("CartPole-v1")
            rewards = []
            for ep in range(eval_episodes):
                obs, info = eval_env.reset()
                done = False
                total_reward = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                rewards.append(total_reward)

            avg_reward = float(np.mean(rewards))
            max_reward = float(np.max(rewards))
            min_reward = float(np.min(rewards))

            # Log metrics with step as key
            mlflow.log_metric("avg_reward", avg_reward, step=ck)
            mlflow.log_metric("max_reward", max_reward, step=ck)
            mlflow.log_metric("min_reward", min_reward, step=ck)

            print(f"Checkpoint {ck}: avg={avg_reward:.2f} max={max_reward:.2f} min={min_reward:.2f}")

            # record a short video for this checkpoint
            # use RecordVideo to write a single episode video
            from gymnasium.wrappers import RecordVideo
            rec_env = gym.make("CartPole-v1", render_mode="rgb_array")
            rec_env = RecordVideo(rec_env, video_folder=str(videos_dir), name_prefix=f"evolution_{ck}", episode_trigger=lambda x: True)
            obs, info = rec_env.reset()
            done = False
            step = 0
            while step < video_steps and not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = rec_env.step(action)
                done = terminated or truncated
                step += 1
            rec_env.close()

            # RecordVideo writes files named like 'evolution_{ck}-episode-0.mp4'
            # Find the generated file
            gen_files = list(videos_dir.glob(f"evolution_{ck}*.mp4"))
            if gen_files:
                gen = gen_files[-1]
                # rename to standardized name
                target = videos_dir / f"evolution_{ck}.mp4"
                if gen != target:
                    gen.replace(target)
                mlflow.log_artifact(str(target), artifact_path=f"videos/{ck}")

            # convergence logging
            if solved_at is None and avg_reward >= target_reward:
                solved_at = ck
                mlflow.log_metric("timesteps_to_solve", float(ck))

        # After all checkpoints, create a combined video ordering by checkpoints
        per_ck_videos = sorted(videos_dir.glob("evolution_*.mp4"), key=lambda p: int(p.stem.split("_")[1]))
        final_video = Path("videos") / f"evolution_{run_name}.mp4"

        # combine sequentially (concatenate frames)
        if per_ck_videos:
            # read first to get frame size
            cap0 = cv2.VideoCapture(str(per_ck_videos[0]))
            ret0, frame0 = cap0.read()
            cap0.release()
            if not ret0:
                print("Could not read checkpoint video frames; skipping final composition.")
            else:
                h, w = frame0.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(final_video), fourcc, 30, (w, h))
                for v in per_ck_videos:
                    cap = cv2.VideoCapture(str(v))
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                    cap.release()
                out.release()
                mlflow.log_artifact(str(final_video), artifact_path="videos")

        print(f"Final evolution video: {final_video}")
        if solved_at is not None:
            print(f"Solved at {solved_at} timesteps (logged to MLflow)")

        return str(final_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar múltiplos PPO Agents para CartPole")
    parser.add_argument("--single", action="store_true", help="Treinar um único agente")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Timesteps (para modo single)")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Episódios de avaliação")
    parser.add_argument("--name", type=str, help="Nome do experimento (para modo single)")
    
    args = parser.parse_args()
    
    if args.single:
        # Modo single agent
        train_ppo_cartpole(
            timesteps=args.timesteps,
            eval_episodes=args.eval_episodes,
            run_name=args.name
        )
    else:
        # Modo múltiplos agentes com comparação
        configs = [
            {'timesteps': 50_000, 'name': 'PPO - 50k steps', 'eval_episodes': 50},
            {'timesteps': 100_000, 'name': 'PPO - 100k steps', 'eval_episodes': 50},
            {'timesteps': 200_000, 'name': 'PPO - 200k steps', 'eval_episodes': 50},
        ]
        
        agents = train_multiple_agents(configs)
        
        # Gerar vídeo de comparação
        create_comparison_video(agents, output_path="videos/comparison_agents.mp4")
