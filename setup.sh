#!/usr/bin/env bash
# Crea el entorno virtual e instala todas las dependencias.
#
#   ./setup.sh
#
# Requiere: Python 3.10+ y una GPU NVIDIA con drivers CUDA instalados
# (el pipeline usa torch.cuda; no corre en CPU).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

echo "== Verificando GPU NVIDIA =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Aviso: no se encontró 'nvidia-smi'. Este proyecto necesita una GPU NVIDIA con CUDA."
    echo "El setup continúa, pero el pipeline no podrá correr sin GPU."
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

echo
echo "== Creando entorno virtual en $VENV_DIR =="
if [ -d "$VENV_DIR" ]; then
    echo "Ya existe $VENV_DIR — se reutiliza (bórralo antes si quieres uno limpio)."
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo
echo "== Instalando dependencias =="
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "== Listo =="
echo "Activa el entorno con:  source $VENV_DIR/bin/activate"
echo "Coloca tus videos de prueba en input_videos/ y corre:"
echo "  python main_pose.py video -i input_videos/tu_video.mp4"
echo "  python main_pose.py videos"
echo "  python main_pose.py stream --camera 0 --display"
echo
echo "Nota: los pesos de YOLO (yolo11n.pt, yolo26n-pose.pt) y del VLM"
echo "(Qwen2-VL-2B-Instruct) se descargan solos la primera vez que corres algo."
echo
echo "Si HuggingFace te pide autenticación (modelos con más restricciones),"
echo "exporta tu propio token antes de correr:  export HF_TOKEN=tu_token_aqui"
