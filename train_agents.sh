#!/usr/bin/env bash

# train_agents.sh
# Helper para rodar os testes de evolução com checkpoints.
# Uso:
#   ./train_agents.sh [mode]
# Modes:
#   quick  - teste rápido (pequeno, rápido para validação)
#   normal - padrão (uso interactivo / recomendável)
#   full   - mais longo (mais checkpoints)
#   multi  - o comportamento antigo (treina 3 agentes e gera comparação)

set -euo pipefail

MODE=${1:-normal}
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Mode: $MODE"
echo "Project dir: $ROOT_DIR"

cd "$ROOT_DIR"

# Ativar ambiente virtual se existir
if [ -f "venv/bin/activate" ]; then
	# shellcheck disable=SC1091
	source venv/bin/activate
else
	echo "Warning: virtualenv venv not found. Assuming global python environment has deps installed."
fi

run_checkpoints() {
	local TIMESTEPS=$1
	local CHECKPOINT=$2
	local EVAL_EP=$3
	local VIDEO_STEPS=$4
	local NAME=$5

	echo "Running train_with_checkpoints: timesteps=${TIMESTEPS}, checkpoint=${CHECKPOINT}, eval_episodes=${EVAL_EP}, video_steps=${VIDEO_STEPS}, name=${NAME}"

	python - <<PYCODE
from train_ppo import train_with_checkpoints
train_with_checkpoints(timesteps=${TIMESTEPS}, checkpoint=${CHECKPOINT}, eval_episodes=${EVAL_EP}, video_steps=${VIDEO_STEPS}, run_name='${NAME}')
PYCODE
}

case "$MODE" in
	quick)
		# Teste rápido: pequeno e rápido
		run_checkpoints 20000 5000 3 200 "test_quick"
		;;
	normal)
		# Padrão: útil para testes locais
		run_checkpoints 100000 20000 5 400 "test_normal"
		;;
	full)
		# Mais longo
		run_checkpoints 200000 50000 10 500 "test_full"
		;;
	multi)
		# Comportamento antigo: treina 3 agentes e gera comparação
		echo "Running legacy multi-agent training (original behavior)"
		python train_ppo.py
		;;
	*)
		echo "Unknown mode: $MODE"
		echo "Available: quick|normal|full|multi"
		exit 2
		;;
esac

echo ""
echo "=================================================="
echo "Run finished. Check 'videos/' and MLflow UI for artifacts and metrics."
echo "To inspect MLflow: mlflow ui  (open http://localhost:5000)"
echo "=================================================="
