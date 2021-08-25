python3 generate_dense_embeddings.py \
	--model_file ./results/movieQA_mBERT_b24/dpr_biencoder.0.9322 \
	--ctx_file ./data/retriever/movie-dpr-bilingual.tsv \
	--out_file ./ctx_embedding_output/movieQA_mBERT_b24  \
	--do_lower_case \
    --sequence_length 384 \
	--batch_size 2000 \
