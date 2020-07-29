export CUDA_VISIBLE_DEVICES=0
python test.py  --conf_path output/depth_guided_config/conf.pkl --weights_path iter35000.0_params.pdparams
