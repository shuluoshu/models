python -m paddle.distributed.launch \
    --selected_gpus=4,5,6,7 \
    --log_dir ./mylog_onecycle_paddle_lr \
    multi_train.py \
    --use_data_parallel True \
    --conf depth_guided_config \
    --save_dir d4lcn_paddle_real_init_onecycle_lr \
    2>&1 | tee d4lcn_paddle_real_init_onecycle_lr.log
    
