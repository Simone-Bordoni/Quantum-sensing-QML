#!/bin/bash

#SBATCH --job-name=train_lfr0        # Job name
#SBATCH --account=phocqal          # Account name
# una cosa che sto levando        #SBATCH --chdir=home/e/emurano/comm_det_snap
#SBATCH --output=/home/e/emurano/comm_det_snap/logs/training_%j.log
#SBATCH --error=/home/e/emurano/comm_det_snap/logs/training_%j.err
#SBATCH --nodes=1                        # Single node
#SBATCH --ntasks-per-node=1            # One task per GPU  (4 H100s)
#SBATCH --partition=bare-metal-GPU        # GPU partition
#SBATCH --gres=gpu:tesla:1              # 4 Tesla GPUs
#SBATCH --cpus-per-task=48              # 192 cores / 4 tasks
#SBATCH --time=6-23:59:00               # Max wall time
#SBATCH --export=ALL
#SBATCH --mem=0                         # Use all available RAM

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Job Name     : $SLURM_JOB_NAME"
echo "Node         : $SLURM_NODELIST"
echo "Nodes        : $SLURM_NNODES"
echo "Tasks        : $SLURM_NTASKS"
echo "Tasks/Node   : $SLURM_NTASKS_PER_NODE"
echo "GPUs on node : $SLURM_GPUS_ON_NODE"
echo "CPUs/Task    : $SLURM_CPUS_PER_TASK"
echo "Start        : $(date)"
echo "=========================================="


# ── LCG environment ──────────────────────────────────────────────────────────
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh

# ── Virtual environment ──────────────────────────────────────────────────────
VENV_DIR="$HOME/comm_det_snap/venvs/myenv"
echo "Activating virtual environment: $VENV_DIR"
export PYTHONPATH="$VENV_DIR/lib/python3.12/site-packages:$PYTHONPATH"
export PATH="$VENV_DIR/bin:$PATH"




# ── CPU threading ────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK




# ── NCCL / networking (H100 NVLink + optional InfiniBand) ────────────────────
export NCCL_DEBUG=WARN                  # INFO for verbose inter-GPU logging
export NCCL_IB_DISABLE=0               # Keep InfiniBand enabled (multi-node)
export NCCL_P2P_LEVEL=NVL              # Prefer NVLink for intra-node P2P (H100)
export NCCL_SOCKET_IFNAME=^lo,docker   # Exclude loopback / docker interfaces
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2




# ── DDP rendezvous ────────────────────────────────────────────────────────────
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export PYTHONUNBUFFERED=1




# ── PyTorch memory allocator (reduces fragmentation) ─────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"




# ── CUDA launch flags ─────────────────────────────────────────────────────────
export CUDA_LAUNCH_BLOCKING=0   # Set to 1 only for debugging




# -- Resume checkpoint --------------------------------------------------------
# Leave empty to start from scratch.
#export RESUME_CKPT="Logs_SATQML_0605/trial0_L10_MOE10_E4_PQC16_lat256_lr1e-04_20260508-101041/checkpoints/last.ckpt"
#export RESUME_CKPT="Logs_SATQML_1405/trial0_mmpn_L10_MOE10_E4_mlp_silu_PQC16_lat128_lr1e-06_20260516-083604/checkpoints/last.ckpt"
#export RESUME_CKPT=" "
#export RESUME_USE_CKPT_HPARAMS=0




# ── GPU information ───────────────────────────────────────────────────────────
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version \
           --format=csv,noheader
echo ""




# ── Working directory ─────────────────────────────────────────────────────────
cd /home/e/emurano/comm_det_snap || exit 1
echo "Working directory: $(pwd)"
echo ""




# ── Runtime / library info ────────────────────────────────────────────────────
echo "Python version:"
python3 --version
which python3
echo ""

echo "PyTorch info:"
python3 -c "
import torch
print(f'PyTorch       : {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices  : {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name}  ({p.total_memory/1024**3:.1f} GB)')
    print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')
"
echo ""

echo "Required packages:"
python3 -c "import pytorch_lightning as pl; print(f'✓ PyTorch Lightning : {pl.__version__}')" \
    || echo "✗ pytorch_lightning not found"
python3 -c "import torch_geometric as pyg; print(f'✓ PyTorch Geometric : {pyg.__version__}')" \
    || echo "✗ torch_geometric not found"
python3 -c "import torch_scatter; print('✓ torch_scatter')"   \
    || echo "✗ torch_scatter not found"
python3 -c "import torchdiffeq;   print('✓ torchdiffeq')"    \
    || echo "  (optional) torchdiffeq not found – RK4 fallback will be used"
echo ""



# ── Training ──────────────────────────────────────────────────────────────────
echo "Starting MMNPN training on H100 ..."
echo "Resume checkpoint: ${RESUME_CKPT:-<none>}"
echo "=========================================="

# srun spawns one task per GPU; Lightning DDP detects SLURM_NTASKS_PER_NODE
# automatically and assigns one GPU per rank.
export SATQML_PQC10_REAL_BACKEND=triton
srun python3 -u training.py

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="