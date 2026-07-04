#!/bin/bash
#SBATCH --job-name=qwen_train_v3
#SBATCH --partition=large
#SBATCH --gres=gpu:nvidia_h200_4g.71gb:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/svc_td_ppml/qrx527/niaz_research_v3_enriched_labels_single_strategy_with_normal_negatives/logs/train_v3_%j.log
#SBATCH --error=/scratch/svc_td_ppml/qrx527/niaz_research_v3_enriched_labels_single_strategy_with_normal_negatives/logs/train_v3_%j.err

set -e

V3ROOT=/scratch/svc_td_ppml/qrx527/niaz_research_v3_enriched_labels_single_strategy_with_normal_negatives
export HF_HOME=/scratch/svc_td_ppml/qrx527/niaz_research/.hf_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== Job started: $(date) ==="
nvidia-smi | head -15

module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/vad_env_v2/bin/activate          # reuse the same venv

mkdir -p $V3ROOT/{logs,lf_data,models}

echo "=== Step 1: prepare LLaMA-Factory data ==="
python $V3ROOT/code/prepare_lf_data_v3.py $V3ROOT/data/out_v3 $V3ROOT/lf_data

echo "=== Step 2: register v3 dataset with LLaMA-Factory ==="
LF=/scratch/svc_td_ppml/qrx527/niaz_research_v2/LLaMA-Factory   # reuse existing LF clone
cp $V3ROOT/lf_data/rtfm_qwen_sft_v3_train.json $LF/data/
cp $V3ROOT/lf_data/rtfm_qwen_sft_v3_eval.json  $LF/data/

python - << PY
import json
from pathlib import Path
p = Path("$LF/data/dataset_info.json")
info = json.loads(p.read_text()) if p.is_file() else {}
tags = {"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant","system_tag":"system"}
for suf, fn in (("_train","rtfm_qwen_sft_v3_train.json"), ("_eval","rtfm_qwen_sft_v3_eval.json")):
    info[f"rtfm_qwen_sft_v3{suf}"] = {
        "file_name": fn, "formatting": "sharegpt",
        "columns": {"messages":"messages","images":"images"},
        "tags": tags,
    }
p.write_text(json.dumps(info, indent=2))
print("Registered v3 datasets in", p)
PY

echo "=== Step 3: launch LLaMA-Factory training ==="
cd $LF
llamafactory-cli train $V3ROOT/code/train_v3.yaml

echo "=== Step 4: list adapter ==="
ls -la $V3ROOT/models/qwen3vl32_rtfm_lora_v3
echo "=== Job finished: $(date) ==="
