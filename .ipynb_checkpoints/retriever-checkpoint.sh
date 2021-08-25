python3 dense_retriever.py \
	--model_file ./results/movieQA_mBERT_bi_1_2/dpr_biencoder.2.38277 \
	--ctx_file ./data/retriever/movie-dpr-bilingual.tsv \
	--qa_file ./data/retriever/movie-train-cn-questions.tsv \
	--encoded_ctx_file ./ctx_embedding_output/movieQA_mBERT_bi_ctxs_1_2_0.pkl \
	--out_file ./ctx_embedding_output/movieQA_mBERT_bi_train_cn \
	--do_lower_case \
	--n-docs 500
