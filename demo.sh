python3 demo_mixture_cn.py \
    --question "克里斯蒂娜·里奇演了什么电影？" \
	--model_file ./results/movieQA_cn_1_2/dpr_biencoder.2.18106 \
	--ctx_file ./data/retriever/movie-cn-dpr.tsv \
	--encoded_ctx_file ./ctx_embedding_output/movieQA_cn_1_2_0.pkl \
	--do_lower_case \
	--n-docs 10 \
	--alpha 0.5