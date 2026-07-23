#!/bin/bash
# Adapted from a colleague's PyTorch/DDP script on the SAME cluster.
# Submit: sbatch job.sbatch <path-to-script.py> | Status: squeue --me | Log: tail -f logs/qsopt_<id>.log
#   e.g.  sbatch job.sbatch examples/articolo/coherent_drive_2qb_corner_sweep.py
# NOTE: run `mkdir -p logs` once before the first submit (SLURM opens the log below).

#SBATCH --job-name=qsopt                 # name shown in squeue
#SBATCH --account=phocqal                # billing/allocation account (unchanged: same cluster)
# #SBATCH --chdir=...                     # left off: we cd into $SLURM_SUBMIT_DIR below instead
#SBATCH --output=logs/qsopt_%j.log       # stdout -> logs/ (%j = job id)
#SBATCH --error=logs/qsopt_%j.err        # stderr
#SBATCH --nodes=1                        # single node
#SBATCH --ntasks-per-node=1              # ONE process: qsopt/JAX is single-process (no DDP)
#SBATCH --partition=bare-metal-GPU       # GPU partition (unchanged)
#SBATCH --gres=gpu:tesla:1               # ONE GPU is enough for a single JAX process
#SBATCH --cpus-per-task=48               # CPU cores for host-side work (lower if you want to queue faster)
#SBATCH --time=6-23:59:00                # max wall time D-HH:MM:SS (lower = schedules sooner)
#SBATCH --export=ALL                     # export current env into the job
#SBATCH --mem=0                          # use all RAM on the node

# ── Job info banner ──────────────────────────────────────────────────────────
echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"          # unique id, also in the log filename
echo "Job Name     : $SLURM_JOB_NAME"
echo "Node         : $SLURM_NODELIST"        # which physical node we landed on
echo "CPUs/Task    : $SLURM_CPUS_PER_TASK"
echo "Start        : $(date)"
echo "=========================================="

# ── Working directory ─────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR" || exit 1            # run from wherever you submitted (your repo checkout)
mkdir -p logs                                # ensure logs/ exists for later runs
echo "Working directory: $(pwd)"

# ── LCG environment (same cluster: provides Python + toolchain) ──────────────
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh

# ── Virtual environment ──────────────────────────────────────────────────────
# One-time setup on the cluster (from repo root, after sourcing LCG above):
#   python3 -m venv venvs/qsopt && pip install -e '.[cuda12]'   # jax[cuda12] + qsopt
# qsopt needs Python >=3.13 (see pyproject.toml) -> check the LCG view's version matches.
VENV_DIR="$SLURM_SUBMIT_DIR/venvs/qsopt"                                    # your qsopt venv
PYVER="$(python3 -c 'import sys;print(f"python{sys.version_info[0]}.{sys.version_info[1]}")')"  # e.g. python3.13
export PYTHONPATH="$VENV_DIR/lib/$PYVER/site-packages:$PYTHONPATH"          # make the venv importable
export PATH="$VENV_DIR/bin:$PATH"                                           # prefer venv executables
echo "Activating virtual environment: $VENV_DIR ($PYVER)"

# ── CPU threading ────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # cap OpenMP threads to allocated cores
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK   # same for MKL (numpy/scipy backends)

# ── JAX / GPU settings (replaces the PyTorch allocator block) ────────────────
export XLA_PYTHON_CLIENT_PREALLOCATE=false    # don't grab the whole GPU at startup
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9     # cap JAX at 90% of GPU RAM
export PYTHONUNBUFFERED=1                      # stream prints into the log immediately
# export JAX_TRACEBACK_FILTERING=off           # uncomment for full tracebacks when debugging

# ── PyTorch/DDP settings from the original: NOT needed for single-GPU JAX ─────
# Kept commented for reference in case you ever run a multi-GPU / multi-node job.
# export NCCL_DEBUG=WARN                       # inter-GPU comm logging
# export NCCL_IB_DISABLE=0                      # InfiniBand for multi-node
# export NCCL_P2P_LEVEL=NVL                     # prefer NVLink for intra-node P2P
# export NCCL_SOCKET_IFNAME=^lo,docker          # exclude loopback/docker interfaces
# export MASTER_ADDR=$(hostname -s)             # DDP rendezvous host
# export MASTER_PORT=29500                       # DDP rendezvous port
# export WORLD_SIZE=$SLURM_NTASKS                # number of DDP ranks
# export CUDA_LAUNCH_BLOCKING=0                  # set 1 to serialize CUDA launches when debugging
# export RESUME_CKPT=""                          # checkpoint resume (their app only)

# ── GPU + runtime info (sanity check before the real run) ────────────────────
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version \
           --format=csv,noheader                                    # confirm a GPU was allocated
echo ""
echo "Python: $(which python3) $(python3 --version)"                 # confirm venv python is first on PATH
python3 -c "import jax; print('JAX', jax.__version__, '- devices:', jax.devices())"  # confirm JAX sees the GPU
echo ""

# ── Run the experiment ───────────────────────────────────────────────────────
SCRIPT="${1:?Usage: sbatch job.sbatch <path-to-script.py> [script-args...]}"  # script path passed as an argument
echo "Running: $SCRIPT ${*:2}"
echo "=========================================="
python3 -u "$SCRIPT" "${@:2}"                                         # -u = unbuffered output; forward any extra args to the script
# srun python3 -u "$SCRIPT" "${@:2}"                                  # original used srun (only needed for multi-task/DDP)

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="
