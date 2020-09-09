nohup python nin-ori.py \
--name "ml20-4x1,4" \
--data_file "ml20.csv" \
--batch_size 256 \
--n_blocks 4 \
--block_shape "1,4" \
--channel 256 \
--kernel_size 3 \
--store_root "store/nin-ori" \
--gpu 0 \
--iter 50 \
--no_rezero true \
> output.log &