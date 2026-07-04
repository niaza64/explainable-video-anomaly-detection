#!/bin/bash
#SBATCH --job-name=qwen_train_v2
#SBATCH --partition=large
#SBATCH --gres=gpu:nvidia_h200_4g.71gb:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/svc_td_ppml/qrx527/niaz_research_v2/logs/train_v2_%j.log
#SBATCH --error=/scratch/svc_td_ppml/qrx527/niaz_research_v2/logs/train_v2_%j.err

set -e

V2ROOT=/scratch/svc_td_ppml/qrx527/niaz_research_v2
export HF_HOME=/scratch/svc_td_ppml/qrx527/niaz_research/.hf_cache   # share base model cache with v1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME"
nvidia-smi | head -15

module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/vad_env_v2/bin/activate

mkdir -p $V2ROOT/{logs,lf_data,models}

echo "=== Step 1: prepare LLaMA-Factory data ==="
python $V2ROOT/code/prepare_lf_data.py $V2ROOT/data/out_v2 $V2ROOT/lf_data

echo "=== Step 2: register dataset with LLaMA-Factory ==="
LF=$V2ROOT/LLaMA-Factory
if [ ! -d "$LF" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git $LF
    cd $LF && pip install -q -e ".[metrics]" && cd -
fi

cp $V2ROOT/lf_data/rtfm_qwen_sft_v2_train.json $LF/data/
cp $V2ROOT/lf_data/rtfm_qwen_sft_v2_eval.json  $LF/data/

python - << 'PY'
import json
from pathlib import Path
import os
LF = Path(os.environ['LF']) if 'LF' in os.environ else Path("/scratch/svc_td_ppml/qrx527/niaz_research_v2/LLaMA-Factory")
p = LF / "data" / "dataset_info.json"
info = json.loads(p.read_text()) if p.is_file() else {}
tags = {"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant","system_tag":"system"}
for suf, fn in (("_train","rtfm_qwen_sft_v2_train.json"), ("_eval","rtfm_qwen_sft_v2_eval.json")):
    info[f"rtfm_qwen_sft_v2{suf}"] = {
        "file_name": fn, "formatting": "sharegpt",
        "columns": {"messages":"messages","images":"images"},
        "tags": tags,
    }
p.write_text(json.dumps(info, indent=2))
print("Registered v2 datasets in", p)
PY

echo "=== Step 3: launch LLaMA-Factory training ==="
cd $LF
llamafactory-cli train $V2ROOT/code/train_v2.yaml

echo "=== Step 4: copy adapter to models/ ==="
SRC=$V2ROOT/models/qwen3vl32_rtfm_lora_v2
ls -la $SRC
echo "Adapter size:"
du -sh $SRC

echo "=== Job finished: $(date) ==="
