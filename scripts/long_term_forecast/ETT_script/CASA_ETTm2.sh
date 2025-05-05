model_name=CASA
seq_len=96
root_path=./
model_id_name=ETTm2

# our method
kernel=3
learning_rate=0.0001

for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./data/LTSF/ETT/ \
    --data_path ETTm2.csv \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 512 \
    --kernel $kernel \
    --learning_rate $learning_rate \
    --batch_size 32 \
    --lradj cosine \
    --train_epochs 20 \
    --patience 3 \
    --des 'Exp_'$kernel'_'$stride'_'$stem_kernel_size'_'$learning_rate \
    --itr 1
done