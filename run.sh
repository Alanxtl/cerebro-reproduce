# 0) 准备：进入某个包目录并安装依赖（或把源码放 data/npm_pkgs/<pkg>）
cd cerebro-repro
npm i               # 在包目录执行，便于 Jelly 分析 node_modules

# 1) 生成 CG
bash scripts/01_gen_cg.sh ./data/npm_pkgs/<your_pkg> ./outputs/cg.json ./outputs/cg.html

# 2) 提取安装脚本入口
python scripts/02_extract_entries.py ./data/npm_pkgs/<your_pkg> > ./outputs/entry_files.txt

# 3) AST 识别 + 维度映射（对入口内的每个 js 做）
python scripts/03_ast_walk_and_map_dims.py $(cat ./outputs/entry_files.txt) > ./outputs/features.jsonl

# 4) 融合 CG 生成“自然语言”行为序列
python scripts/04_build_sequences.py ./outputs/cg.json ./outputs/features.jsonl ./outputs/sequences.jsonl

# 5) 训练（如你已有标签）
python scripts/05_train_bert.py

# 6) 推理
python scripts/06_infer.py ./outputs/sequences.jsonl
