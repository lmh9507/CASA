export CUDA_VISIBLE_DEVICES=2
model_name=CASA
seq_len=96
root_path=./
model_id_name=ECL

# our method
kernel=3
learning_rate=0.0002

for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./data/LTSF/electricity/ \
    --data_path electricity.csv \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --d_model 512 \
    --kernel $kernel \
    --learning_rate $learning_rate \
    --lradj cosine \
    --train_epochs 15 \
    --batch_size 8 \
    --patience 30 \
    --des 'Exp_'$kernel'_'$stride'_'$stem_kernel_size'_'$learning_rate \
    --itr 1
done
