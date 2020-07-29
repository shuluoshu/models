export CUDA_VISIBLE_DEVICES=7
python train.py \
    --conf depth_guided_config \
    --save_dir d4lcn_pytorch_init_bs2 \
    2>&1 | tee d4lcn_pytorch_init_bs2.log
