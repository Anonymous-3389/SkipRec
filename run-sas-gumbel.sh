nohup python sas-gumbel.py \
--name "ml20-nopre-b32" \
--data_file "ml20.csv" \
--max_len 20 \
--batch_size 256 \
--num_blocks 32 \
--hidden 256 \
--num_head 2 \
--gpu 6 \
--temp 10 \
--store_root "store/sas-gumbel" \
--grad_clip \
> output.log &

# for 2-stage training, a pre-trained model with same hyper-parameters is required
# --use_pre --pre /path/to/model.tfkpt