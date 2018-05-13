export BANDIT_HOME=$(dirname $(pwd))
export DATA=$BANDIT_HOME/data

src=$1
tgt=$2
lang=${2}-${1}

export DATA_HOME=$DATA/${lang}
export DATA_PREP=$DATA_HOME/prep

python ../preprocess.py \
  -train_src $DATA_PREP/vocab_train.$lang.$src \
  -train_tgt $DATA_PREP/vocab_train.$lang.$tgt \
  -train_xe_src $DATA_PREP/sup_train.$lang.$src \
  -train_xe_tgt $DATA_PREP/sup_train.$lang.$tgt \
  -train_pg_src $DATA_PREP/bandit_train_trun.$lang.$src \
  -train_pg_tgt $DATA_PREP/bandit_train_trun.$lang.$tgt \
  -sup_valid_src $DATA_PREP/sup_valid.$lang.$src \
  -sup_valid_tgt $DATA_PREP/sup_valid.$lang.$tgt \
  -bandit_valid_src $DATA_PREP/bandit_valid.$lang.$src \
  -bandit_valid_tgt $DATA_PREP/bandit_valid.$lang.$tgt \
  -bandit_test_src $DATA_PREP/bandit_test.$lang.$src \
  -bandit_test_tgt $DATA_PREP/bandit_test.$lang.$tgt \
  -save_data $DATA_HOME/processed_all
