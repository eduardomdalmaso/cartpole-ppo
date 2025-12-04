# CartPole Reinforcement Learning Project

Um projeto de aprendizado por reforço que implementa agentes para resolver o problema do CartPole usando a biblioteca Gymnasium e Stable Baselines3.

## 📋 Descrição

Este projeto treina um agente de inteligência artificial para balancear um poste (pole) em cima de um carrinho (cart) usando técnicas de aprendizado por reforço. O objetivo é manter o poste em pé pelo máximo de tempo possível.

## 🎯 Objetivos

- Treinar um agente PPO (Proximal Policy Optimization) para resolver o problema CartPole
- Registrar métricas de treinamento usando MLflow
- Avaliar o desempenho do agente treinado
- Armazenar modelos treinados para uso futuro

## 📁 Estrutura do Projeto

```
├── train_cartpole.py      # Script principal de treinamento com MLflow
├── agent-cart.py          # Script de demonstração com ação aleatória
├── requeriments.txt       # Dependências do projeto
├── README.md              # Este arquivo
├── mlruns/                # Histórico de execuções do MLflow
├── models/                # Modelos treinados armazenados
└── video/                 # Vídeos de demonstração (se aplicável)
```

## 🛠️ Requisitos

- Python 3.8+
- Gymnasium
- Stable Baselines3
- MLflow
- Numpy
- Matplotlib

Veja `requeriments.txt` para a lista completa de dependências.

## 📦 Instalação

1. Clone ou baixe este repositório
2. Instale as dependências:

```bash
pip install -r requeriments.txt
```

3. Instale o pacote clássico do Gymnasium (se necessário):

```bash
pip install "gymnasium[classic-control]"
```

## 🚀 Como Usar

### Treinar o Agente

Execute o script de treinamento:

```bash
python train_cartpole.py
```

Este script irá:
- Criar um ambiente CartPole
- Treinar um modelo PPO por 100.000 timesteps
- Avaliar o modelo em 50 episódios
- Registrar métricas no MLflow
- Salvar o modelo treinado

### Demonstração com Ação Aleatória

Para ver o ambiente funcionando com ações aleatórias:

```bash
python agent-cart.py
```

Este script executa episódios com ações aleatórias até atingir uma recompensa de 80 pontos.

## 📊 Métricas e Rastreamento

O projeto utiliza **MLflow** para rastrear:
- Algoritmo utilizado (PPO)
- Número de timesteps de treinamento
- Número de episódios de avaliação
- Recompensa média (avg_reward)
- Recompensa máxima (max_reward)
- Recompensa mínima (min_reward)

Visualize o histórico de execuções:

```bash
mlflow ui
```

Então acesse `http://localhost:5000` no seu navegador.

## 🎮 Ambiente CartPole

O CartPole é um ambiente clássico de controle do Gymnasium onde:
- **Objetivo**: Manter o poste em pé
- **Ação**: Empurrar o carrinho para esquerda (0) ou direita (1)
- **Recompensa**: +1 para cada timestep enquanto o poste está em pé
- **Episódio termina quando**: O poste cai ou máximo de timesteps é atingido

## 📈 Resultados Esperados

Um agente bem treinado deve atingir:
- Recompensa média: ~500+ pontos
- Recompensa máxima: Próximo ao máximo do ambiente

## 🔧 Personalização

Para modificar os parâmetros de treinamento, edite `train_cartpole.py`:

```python
timesteps = 100_000      # Número de timesteps de treinamento
eval_episodes = 50       # Número de episódios para avaliação
```

## 📚 Referências

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [CartPole Problem](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

## 📝 Licença

Este projeto é fornecido como está para fins educacionais.

## 👤 Autor

Projeto de aprendizado por reforço desenvolvido em Python.

---

**Última atualização**: Dezembro de 2025
