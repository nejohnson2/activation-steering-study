# The Geometry of Persona

**Comparing Activation Steering Methods for Behavioral Control in Large Language Models**

## Abstract

Activation steering has emerged as a promising alternative to prompt engineering for controlling large language model (LLM) behavior, yet systematic comparisons of vector extraction methods remain scarce. We present a comprehensive study comparing three approaches to persona steering vector identification -- Contrastive Activation Addition (CAA), PCA-based extraction, and supervised linear probes -- across three architecturally distinct model families (Llama 3.1 8B, Gemma 2 9B, Qwen 2.5 7B) and five diverse persona types. For each method-model combination, we map the layer-wise effectiveness of steering vectors, revealing whether optimal injection points follow a consistent pattern relative to network depth or vary with architecture and persona type. We benchmark all steering conditions against prompt engineering baselines using a three-tier evaluation framework: representation-level metrics (cosine similarity shift, projection magnitude), a persona classifier trained on synthetic data, and LLM-as-judge scoring on a five-point Likert scale. Our findings characterize the trade-offs between extraction methods in terms of steering fidelity, behavioral consistency, and fluency preservation, while identifying the conditions under which activation steering surpasses -- or falls short of -- simple system-prompt-based persona control. We release our full experimental framework, steering vectors, and evaluation pipeline to support reproducible research in mechanistic approaches to LLM behavioral control.

## Research Questions

1. **RQ1 (Method Comparison):** How do CAA, PCA-based, and linear-probe steering vectors compare in steering fidelity, behavioral consistency, and side-effect minimization?
2. **RQ2 (Layer-wise Topology):** Is there a consistent optimal steering layer relative to model depth, or does it vary by architecture and persona type?
3. **RQ3 (Steering vs. Prompting):** Does activation steering achieve persona adherence that prompt engineering cannot?

## Methods

### Vector Extraction
- **Contrastive Activation Addition (CAA):** Mean activation difference between persona-present and persona-absent pairs
- **PCA-based:** First principal component of combined positive/negative activations, oriented toward positive centroid
- **Linear Probe:** Weight vector from a trained binary classifier distinguishing persona-present from persona-absent activations

### Models
- Llama 3.1 8B Instruct
- Gemma 2 9B IT
- Qwen 2.5 7B Instruct

### Personas
- Formal Academic
- Empathetic Counselor
- Socratic Teacher
- Creative Storyteller
- Terse Engineer

### Evaluation (3-tier)
1. **Representation-level:** Cosine similarity shift, projection magnitude
2. **Persona classifier:** MLP trained on sentence embeddings of synthetic persona data
3. **LLM-as-judge:** Local Ollama model rates persona adherence on a 1-5 Likert scale

## Setup

### Local Development (macOS/MPS)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### NVWulf Cluster
```bash
module load cuda12.8/toolkit/12.8.1
conda create -n steering-study python=3.11
conda activate steering-study
pip install -r requirements.txt
```

### Ollama (LLM Judge)
The LLM-as-judge evaluation runs locally via Ollama. Install from [ollama.com](https://ollama.com), then pull a judge model:
```bash
ollama pull llama3.1:8b
```

### Environment Variables
```bash
export WANDB_API_KEY="your-key"        # Optional: W&B tracking
export HF_TOKEN="your-token"           # HuggingFace model access (gated models)
```

## Usage

### Two-Stage Workflow

**Stage 1: GPU compute (NVWulf)** -- extraction, steering, and baselines:
```bash
make slurm-all                         # Submit all models
make slurm-model MODEL=llama3.1-8b     # Submit single model
```

**Stage 2: Evaluation (local Mac)** -- classifier + Ollama judge (no GPU needed):
```bash
make evaluate                          # Run all evaluation tiers
make evaluate-classifier               # Classifier only (fast)
make evaluate-judge                    # LLM judge only (requires Ollama)
```

### Development Run (single model, single persona, reduced params)
```bash
make dev
```

### Full Pipeline (local, all phases)
```bash
make all
```

### Individual Phases
```bash
make extract                           # Phase 1: Extract steering vectors
make baselines                         # Phase 2: Prompt engineering baselines
make steer                             # Phase 3: Apply steering vectors
make evaluate                          # Phase 4: Run all evaluation tiers
```

### Filter by Model or Persona
```bash
make extract MODEL=llama3.1-8b
make steer MODEL=gemma2-9b PERSONA=formal_academic
```

### Inspect Outputs
```bash
python scripts/inspect_outputs.py --model llama3.1-8b --persona formal_academic --summary
python scripts/inspect_outputs.py --model llama3.1-8b --persona formal_academic --baselines
python scripts/inspect_outputs.py --model llama3.1-8b --persona formal_academic --method caa --layer 16 --multiplier 2.0
```

## Project Structure
```
activation-steering-study/
├── configs/
│   ├── default.yaml          # Full experiment parameters
│   └── dev.yaml              # Lightweight dev/test parameters
├── src/
│   ├── extraction/
│   │   ├── hooks.py           # PyTorch hook-based activation extraction
│   │   ├── caa.py             # Contrastive Activation Addition
│   │   ├── pca.py             # PCA-based extraction
│   │   └── linear_probe.py    # Linear probe extraction
│   ├── steering/
│   │   └── injector.py        # Steering vector injection
│   ├── evaluation/
│   │   ├── representation.py  # Representation-level metrics
│   │   ├── classifier.py      # Persona classifier
│   │   └── llm_judge.py       # LLM-as-judge evaluation
│   ├── data/
│   │   └── personas.py        # Persona dataset and prompt generation
│   └── utils/
│       ├── device.py          # Device auto-detection
│       └── tracking.py        # W&B / file-based experiment tracking
├── scripts/
│   ├── run_extraction.py      # Phase 1: Vector extraction
│   ├── run_steering.py        # Phase 2: Steering experiments
│   ├── run_baselines.py       # Phase 3: Prompt baselines
│   ├── run_evaluation.py      # Phase 4: Evaluation
│   └── inspect_outputs.py     # Qualitative output inspection
├── slurm/
│   ├── extract_vectors.sbatch
│   ├── run_steering.sbatch
│   ├── run_baselines.sbatch
│   └── run_all.sh             # Submit full pipeline
├── results/                   # Output directory (gitignored)
├── Makefile
├── requirements.txt
└── README.md
```

## Experiment Tracking

W&B is the default tracker. If unavailable (no account or no network), falls back to file-based JSONL logging in `results/logs/`.

To use W&B:
1. Create a free account at wandb.ai
2. Run `wandb login`
3. Set `tracking.wandb.entity` in `configs/default.yaml` to your username

## Reproducibility

- All random seeds are set via `project.seed` in the config (default: 42)
- Exact dependency versions pinned in `requirements.txt`
- All hyperparameters stored in `configs/default.yaml`
- Experiment configs logged to W&B or JSONL at run start
