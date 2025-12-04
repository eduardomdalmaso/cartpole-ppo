#!/bin/bash

# Script para treinar múltiplos agentes e gerar vídeo de comparação
# Uso: bash train_agents.sh

echo "=================================================="
echo "CartPole-PPO: Sistema de Treinamento de Múltiplos Agentes"
echo "=================================================="
echo ""
echo "Este script treinará 3 agentes com diferentes números de timesteps"
echo "e gerará um vídeo .mp4 comparando o desempenho deles."
echo ""
echo "Tempo estimado: ~3-5 minutos"
echo ""

cd /home/emd/Documents/Cartpole

# Ativar ambiente virtual
source venv/bin/activate

echo "Iniciando treinamento..."
echo ""

# Executar o script de treinamento
python train_ppo.py

echo ""
echo "=================================================="
echo "Treinamento concluído!"
echo "Vídeo de comparação: videos/comparison_agents.mp4"
echo "=================================================="
