model_name=CASA
seq_len=96
root_path=./
model_id_name=solar

# our method
kernel=3
learning_rate=0.00015

for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./data/LTSF/solar/ \
    --data_path solar_AL.txt \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 4 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --d_model 512 \
    --kernel $kernel \
    --learning_rate $learning_rate \
    --batch_size 32 \
    --lradj cosine \
    --train_epochs 10 \
    --patience 10 \
    --des 'Exp_'$kernel'_'$stride'_'$stem_kernel_size'_'$learning_rate \
    --itr 1
done