python scripts/bytellama_prepare_openwebtext.py \
  --destination_path data/lit-openwebtext


python pretrain/bytellama.py --devices 6 --data_dir data/lit-openwebtext