This is the reproduction Cerebro which proposed in TOSEM 25' paper "Killing Two Birds with One Stone: Malicious Package Detection in NPM and PyPI using a Single Model of Malicious Behavior Sequence"


## Prerequesities

1. Download jelly

    ```bash
    npm install -g @cs-au-dk/jelly
    ```

2. Prepare python dependencies

    ```bash
    pip install -r requirements.txt # which not contains torch related dependencies
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # switch CUDA/CPU if needed
    pip install transformers datasets scikit-learn
    ```

## Usage

1. Prepare dataset

    Input: dataset dir contains samples in .tar(.zip/.tar.gz)
    Output: sequences.jsonl 

    ```bash
    # label = 1 means malicious
    python scripts/run_all.py --dataset_dir /dir/to/mal_dataset --out_dir ./outputs  --workers 16 --jelly_timeout 1000 --label 1
    ```

2. Train the model

    ```bert
    python .\scripts\05_train_bert.py
    ```

## Dir tree

```
cerebro-repro/
├─ outputs/
│  ├─ logs
│  ├─ sequences.jsonl            # the input of BERT
│  └─ models/                    # BERT fine-tune checkpoint
├─ scripts/
│  ├─ 01_gen_cg.sh
│  ├─ 02_extract_entries.py
│  ├─ 03_ast_walk_and_map_dims.py
│  ├─ 04_build_sequences.py
│  ├─ 05_train_bert.py
│  └─ 06_infer.py
└─ README.md
```
