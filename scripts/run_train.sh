#sample script to run the model

mkdir -p ../save_dir/pretrain
mkdir -p ../save_dir/sentlv
mkdir -p ../save_dir/bipnmt

MODEL_HOME=$(dirname $(pwd))
DATA_PATH=$MODEL_HOME/data/processed_all-train.pt

pretrain_epoch=$1
bandit_epoch=$((1+$pretrain_epoch))


#out-of-domain model / pretrained model
python ../train.py \
    -data $DATA_PATH \
    -save_dir $MODEL_HOME/save_dir/pretrain \
    -batch_size 64 \
    -eval_batch_size 64 \
    -log_interval 2500 \
    -end_epoch $pretrain_epoch


# sentence level bandit feedback
python ../train.py \
    -data $DATA_PATH \
    -save_dir $MODEL_HOME/save_dir/sentlv \
    -load_from $MODEL_HOME/save_dir/pretrain/model_${pretrain_epoch}.pt \
    -reinforce_lr 0.00001 \
    -batch_size 1 \
    -eval_batch_size 64 \
    -log_interval 1000 \
    -start_reinforce ${bandit_epoch} \
    -end_epoch ${bandit_epoch}


# bip-nmt
python ../train.py \
    -data $DATA_PATH \
    -save_dir $MODEL_HOME/save_dir/bipnmt \ 
    -load_from $MODEL_HOME/save_dir/pretrain/model_${pretrain_epoch}.pt \
    -reinforce_lr 0.00001 \
    -use_bipnmt \
    -mu 0.8 \
    -eps 0.75 \
    -batch_size 1 \
    -eval_batch_size 64 \
    -log_interval 1000 \
    -start_reinforce ${bandit_epoch} \
    -end_epoch ${bandit_epoch}
   
