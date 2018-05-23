# sample script to evaluate a trained model

HOME_DIR=$(dirname $(pwd))

# evaluate the pretrained model
python ../train.py \
    -data $HOME_DIR/data/processed_all-train.pt \
    -save_dir $HOME_DIR/save_dir/pretrain \
    -load_from $HOME_DIR/save_dir/pretrain/model_1.pt \
    -eval_batch_size 64 \
    -eval
