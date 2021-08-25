python3 train_dense_encoder.py \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg bert-base-multilingual-uncased \
    --train_file ./data/retriever/movie-train-bilingual-one-positive.json \
    --dev_file ./data/retriever/movie-dev-bilingual-one-positive.json \
    --other_negatives 1 \
    --output_dir results/movieQA_mBERT_b12 \
    --do_lower_case \
    --sequence_length 384 \
    --batch_size 24 \
    --n_gpu 2 \