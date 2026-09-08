# DSBD
Code for AAAI'25 "[Dynamic-Width Speculative Beam Decoding for Efficient LLM Inference](https://arxiv.org/abs/2409.16560)"

# Environment

For **Llama-3.2-1B + Llama-3.1-8B** (this eval), use the pins in `requirements.txt`. Do **not** install the paper's `transformers==4.35.2` — those Llama checkpoints need `transformers>=4.43.2`.

| Package | Version | Why |
|---------|---------|-----|
| `torch` | keep your CUDA wheel (`>=2.1`) | paper used 2.1.1; do not force-downgrade a working install |
| `transformers` | **4.44.2** | Llama 3.1/3.2 configs + tokenizers; `<4.50` avoids GenerationMixin breakage |
| `tokenizers` | **0.19.1** | matches transformers 4.44.x (0.22.x is for 4.57+) |
| `accelerate` | **0.34.2** | required for `device_map="auto"` |
| `huggingface_hub` | `>=0.23.4,<0.26` | gated Meta downloads |

```bash
# 1) Install a CUDA torch that matches the machine (example CUDA 12.1)
# pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 2) Then the rest (does not pin torch)
pip install -r requirements.txt

# 3) Llama-3 weights are gated: accept the license on Hugging Face, then
huggingface-cli login
# or: export HFTOKEN=hf_...   (evaluation.py reads HFTOKEN)
```

Mismatch that usually fails installs: `transformers 4.35` + `tokenizers 0.22`, or `transformers 4.57` + this repo's old beam helpers. After install you want `transformers==4.44.2` and `tokenizers==0.19.1`.

Alternatively, the paper docker image is `zongyueq/llmss:0.0.2` (`source ~/miniconda3/bin/activate myenv; conda activate myenv`). That image is the old 4.35 stack — fine for OPT / Llama-2, not for Llama 3.2.

Your GPU needs to support `nvidia-smi` to measure GPU energy consumption.

# Data

**SQuAD** downloads automatically via Hugging Face (`datasets`).

**Spider** does **not**. This repo needs **Spider 1.0** (Yale text-to-SQL), not Spider 2.0.

## Get Spider 1.0

1. Open the official page: https://yale-lily.github.io/spider  
   Click **Spider Dataset** (Google Drive). Direct file:  
   https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing  
   License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

2. From this repo root, unzip so the paths below exist. The Drive zip usually extracts as a folder named `spider/` containing `dev.json`, `tables.json`, and `database/`. This code looks one extra level down for the JSON files:

```bash
cd /path/to/DSBD
# after downloading spider.zip (or spider_data.zip) from Drive:
unzip spider.zip

# If you now have ./spider/dev.json (typical), nest the JSON files:
mkdir -p spider/spider
mv spider/dev.json spider/tables.json spider/train_spider.json spider/train_others.json spider/spider/ 2>/dev/null || true

# Keep sqlite DBs at ./spider/database/<db_id>/<db_id>.sqlite  (do not nest those)
```

Required layout (what the code reads):

```
DSBD/
  spider/
    spider/
      dev.json
      tables.json
    database/                    # sqlite DBs used for exec accuracy
      concert_singer/concert_singer.sqlite
      ...
```

If your unzip is `spider/spider/{dev.json,tables.json,database/}` (common), add one link from the repo root:

```bash
ln -sfn spider/database spider/database
# i.e. DSBD/spider/database -> DSBD/spider/spider/database
```

Sanity check:

```bash
test -f spider/spider/dev.json && test -f spider/spider/tables.json \
  && test -f spider/database/concert_singer/concert_singer.sqlite \
  && echo "Spider layout OK"
```

Do **not** commit `spider/` (it is gitignored). If `execution_accuracy` cannot open a DB, the path in `sampling/utils.py` is `./spider/database/{db}/{db}.sqlite`.

Copy the dataset to another machine (about 1.8 GB). From this repo root:

```bash
# recursive copy of the local spider/ tree (JSON + sqlite DBs + symlink)
scp -r spider priyanshi@lamport.cse.iitd.ac.in:~/DSBD/

# if ~/DSBD/ does not exist on lamport yet:
# ssh priyanshi@lamport.cse.iitd.ac.in 'mkdir -p ~/DSBD'
# scp -r spider priyanshi@lamport.cse.iitd.ac.in:~/DSBD/
```

On lamport, `cd ~/DSBD` and re-check the layout (the `spider/database` symlink may need to be recreated):

```bash
ln -sfn spider/database spider/database
test -f spider/spider/dev.json && test -f spider/database/concert_singer/concert_singer.sqlite && echo "Spider layout OK"
```

# Example Run

`python evaluation.py --approx_model_name meta-llama/Llama-3.2-1B --target_model_name meta-llama/Llama-3.1-8B --max_tokens 200 --max_seconds 10000 --log_file /llmss/DSBD/logs/tmp.log --dataset squad --top_k=10 --top_p=0.9 --num_inputs=10`

- appox\_model\_name: path of the draft model
- target\_model\_name: path of the target model
- max\_tokens: the number of tokens to generate (values we used: 100, 200)
- max\_seconds: the time limit for each method
- log\_file: path to the log file
- dataset: squad or spider
- top\_k: k for top k sampling (values we used: 10, 20)
- top\_p: p for top p sampling (values we used: 0.8, 0.9)
- num\_inputs: the number of inputs to test (values we used: 100, 200)

To run experiments with MT-Bench, please download its repo and replace the decoding function in "gen\_model\_answer.py" with our "beam\_speculative\_sampling".

# Constrained DSBD (xgrammar + Z3) on Spider

Measures how beam search + speculative decoding behaves with **syntactic** (xgrammar) and **semantic** (Z3 schema) constraints. Primary metrics: **execution accuracy** and **goodput** (useful committed tokens / wall time — only counted when the SQL is executable), vs raw **throughput**.

## Setup

```bash
pip install -r requirements.txt   # includes xgrammar, z3-solver, matplotlib
# Spider layout OK (see Data above). SQuAD is not used for this study.
export HFTOKEN=...                # required for gated Llama-3 models
# optional: huggingface-cli login
```

Need a GPU with enough memory for the 8B target (or change the model names). `evaluation.py` pins `CUDA_VISIBLE_DEVICES=0`.

## Run sequence

**1. Smoke (optional, ~2–5 min on a single mid/high-end GPU)** — 5 examples, one constraint mode:

```bash
python evaluation.py \
  --approx_model_name meta-llama/Llama-3.2-1B \
  --target_model_name meta-llama/Llama-3.1-8B \
  --dataset spider --num_inputs 5 --max_tokens 64 \
  --constraints both --fixed_dsbd --skip_baselines \
  --metrics_csv logs/constraint_smoke.csv \
  --log_file logs/constraint_smoke.txt
```

**2. Full 2×2 ablation (recommended microbench)** — fixed DSBD (`width=4`, `gamma=3`, `w_thres=0.9`), modes `none / xgrammar / z3 / both`:

```bash
python evaluation.py \
  --approx_model_name meta-llama/Llama-3.2-1B \
  --target_model_name meta-llama/Llama-3.1-8B \
  --dataset spider --num_inputs 50 --max_tokens 64 \
  --constraint_ablation --fixed_dsbd --skip_baselines \
  --metrics_csv logs/constraint_metrics.csv \
  --log_file logs/constraint_ablation.txt
```

**3. Aggregate + plots:**

```bash
python scripts/analyze_constraint_results.py \
  --metrics_csv logs/constraint_metrics.csv \
  --out_csv logs/constraint_summary.csv \
  --plot_dir logs/constraint_plots
```

Useful flags: `--constraints {none,xgrammar,z3,both}`, `--num_inputs`, `--dsbd_width`, `--dsbd_gamma`, `--dsbd_w_thres`, `--max_seconds`.

## Approximate runtime

Times assume models are already cached on disk. First download of Llama weights can add **10–30+ min**.

| Stage | Examples × modes | Approx. wall time (1× A100 / RTX 4090-class) |
|--------|------------------|-----------------------------------------------|
| Smoke (`num_inputs=5`, one mode) | 5 | **2–5 min** (incl. model load) |
| Microbench (`num_inputs=50`, 4 modes) | 200 decode runs | **45–90 min** |
| Paper-scale (`num_inputs=100–200`, 4 modes) | 400–800 runs | **1.5–4 hours** |

Rough breakdown for the **50×4** microbench:
- Model load once: **~2–5 min**
- Decode: ~50 examples × 4 constraint modes; SQL gens are short (`max_tokens=64`) but xgrammar masking and Z3 checks add overhead vs plain DSBD — often **~5–20 s/example/mode** depending on GPU and accept length
- Analysis script: **seconds**

On smaller GPUs (e.g. 16–24 GB) or if weights are offloaded, expect **~1.5–2×** slower. CPU-only is not practical for the 8B target.

# Acknowledgement

The code is forked from "https://github.com/feifeibear/LLMSpeculativeSampling"
