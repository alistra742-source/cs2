# DragSolver Fine-Tuning Pipeline

Fine-tune `ahmadwaqar/smolvlm2-256m-video:q8_0` (or any SmolVLM2 model)
on your captcha variant. Scales from **50 to 50,000 samples**.

## Step 1: Collect — Unlimited Auto-Collection

The bot auto-saves every successful solve. Just let it run:

```bash
# After each run, check how many samples you have:
ls train/data/images/ | wc -l
wc -l train/data/samples.jsonl
```

No limit. Run as many accounts as you want, the collector keeps appending.

## Step 2: Train — Adapts to Your Dataset Size

The training script auto-tunes itself based on how many samples you have:

| Samples | Epochs | Batch | LR | Time (T4) | Expected Slider Acc |
|---------|--------|-------|-----|-----------|---------------------|
| 200     | 20     | 4     | 3e-4 | ~25 min   | ~75%                |
| 1,000   | 10     | 4     | 2e-4 | ~2 hrs    | ~88%                |
| 5,000   | 4      | 4     | 1.5e-4 | ~6 hrs  | ~93%                |
| 20,000  | 2      | 2     | 1e-4  | ~14 hrs  | ~96%                |
| 50,000  | 1      | 2     | 8e-5  | ~18 hrs  | ~97%                |

Just run:

```bash
# Train on everything you've collected
python train/fine_tune_smolvlm.py --data_dir train/data

# For >5000 samples, enable streaming to avoid OOM:
python train/fine_tune_smolvlm.py --data_dir train/data --stream

# Override auto-params if you want:
python train/fine_tune_smolvlm.py --data_dir train/data --epochs 5 --batch_size 4
```

The script:
- **Auto-calculates epochs**: moves to next epoch faster as data grows
- **Auto-tunes LR**: lower LR for more data (stable convergence)
- **Auto-tunes batch size**: smaller batches for thousands (fits in VRAM)
- **Gradient checkpointing** on by default
- **Lazy streaming** with `--stream` for datasets >5000 (never loads all images into RAM)

### Run on Free GPU

Copy `fine_tune_smolvlm.py` into Google Colab with **T4 GPU runtime**:
[Open In Colab](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp)

Upload your `train/data/` folder and run. No cost.

### Run on RunPod (for 10,000+ samples)

Cheapest option: RunPod RTX 4090 (~$0.34/hr) — trains 5000 samples in ~3 hrs = ~$1.

## Step 3: Export to Ollama

```bash
python train/fine_tune_smolvlm.py --export_ollama train/drag-solver-model

# Then follow the printed commands:
# convert_hf_to_gguf.py -> ollama create -> OLLAMA_MODEL=drag-solver
```

## Step 4: Use It

```bash
export OLLAMA_MODEL=drag-solver
# Run the bot — now with your custom model
```

## Continuous Training

As you collect more data over weeks, just re-run the training script:

```bash
# After collecting 500 more samples:
python train/fine_tune_smolvlm.py --data_dir train/data \
    --output_dir train/drag-solver-model-v2
```

The collector always appends — never duplicates. You build a better model
with every bot session.

## Hardware Requirements

| # Samples | Min VRAM | Cheap GPU | Cost |
|-----------|----------|-----------|------|
| < 5,000   | 6 GB     | T4 (Colab free)  | $0   |
| < 20,000  | 8 GB     | RTX 3060 ($0.10/hr)  | ~$1 |
| Any       | 12 GB    | RTX 4090 ($0.34/hr)  | ~$2 |

## Upgrading the Base Model

For better results with the same pipeline, change the base model:

```python
# In fine_tune_smolvlm.py, change:
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-2.2B-Instruct"  # stronger, needs 12GB
# or:
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"  # best small VLM, needs 12GB
```