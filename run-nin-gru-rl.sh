nohup python nin-gru-rl.py \
--name "ml20-nopre-4x1,4" \
--data_file "ml20.csv" \
--batch_size 256 \
--rl_iter 0 \
--temp 10 \
--n_blocks 4 \
--block_shape "1,4" \
--gamma 1 \
--store_root "store/nin-gru-rl" \
--channel 256 \
--kernel_size 3 \
--gpu 0 \
> output.log &

# for 2-stage training, a pre-trained model with same hyper-parameters is required
# --use_pre --pre /path/to/model.tfkpt