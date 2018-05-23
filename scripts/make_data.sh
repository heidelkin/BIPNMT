export BANDIT_HOME=$(dirname $(pwd))
export DATA=$BANDIT_HOME/data

src=$1
tgt=$2
lang=${2}-${1}

python ../preprocess.py \
  -train_src $DATA/vocab_train.$lang.$src \
  -train_tgt $DATA/vocab_train.$lang.$tgt \
  -train_xe_src $DATA/sup_train.$lang.$src \
  -train_xe_tgt $DATA/sup_train.$lang.$tgt \
  -train_pg_src $DATA/bandit_train_trun.$lang.$src \
  -train_pg_tgt $DATA/bandit_train_trun.$lang.$tgt \
  -sup_valid_src $DATA/sup_valid.$lang.$src \
  -sup_valid_tgt $DATA/sup_valid.$lang.$tgt \
  -bandit_valid_src $DATA/bandit_valid.$lang.$src \
  -bandit_valid_tgt $DATA/bandit_valid.$lang.$tgt \
  -bandit_test_src $DATA/bandit_test.$lang.$src \
  -bandit_test_tgt $DATA/bandit_test.$lang.$tgt \
  -save_data $DATA/processed_all
