python3 dense_retriever.py \
	--model_file ./results/msmarco_b24/dpr_biencoder.1.14963 \
	--ctx_file ./data/retriever/msmarco-collections.tsv \
	--qa_file ./data/retriever/msmarco-dev-qa.tsv \
	--encoded_ctx_file ./ctx_embedding_output/msmarco_b24_0.pkl \
	--out_file ./ctx_embedding_output/msmarco_b24 \
	--do_lower_case \
	--n-docs 100
