python3 train_dense_encoder.py \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg bert-base-chinese \
    --train_file ./data/retriever/movie-cn-train.json \
    --dev_file ./data/retriever/movie-cn-dev.json \
    --other_negatives 1 \
    --output_dir results/movieQA_cn_b12 \
    --do_lower_case \
    --sequence_length 384 \
    --batch_size 12 \
