python3 generate_dense_embeddings.py \
	--model_file ./results/movieQA_mBERT_bi_1_2/dpr_biencoder.2.38277 \
	--ctx_file ./data/retriever/movie-dpr-bilingual.tsv \
	--out_file ./ctx_embedding_output/movieQA_mBERT_bi_ctxs_1_2  \
	--do_lower_case
