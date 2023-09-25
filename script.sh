python scripts/bytellama_prepare_openwebtext.py \
  --destination_path data/lit-openwebtext


python pretrain/bytellama.py --devices 8 --data_dir data/lit-openwebtext


python scripts/convert_lit_checkpoint.py --checkpoint_name iter-150000-ckpt.pth --out_dir out/bytellama_16384/ --model_name bytellama_16384 --bytellama True

