nohup python sas-ori.py \
--name "ml20-b32" \
--data_file "ml20.csv" \
--max_len 20 \
--batch_size 256 \
--num_blocks 32 \
--hidden 256 \
--num_head 2 \
--gpu 7 \
--store_root "store/sas-ori" \
--grad_clip \
> output.log &