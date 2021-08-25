#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import argparse
import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator
from opencc import OpenCC
cc = OpenCC('t2s')
import numpy as np
import torch
from torch import Tensor as T
from torch import nn

from collections import OrderedDict
from itertools import repeat

# BM25 preprocessing
from nltk.stem.porter import PorterStemmer
import string
import re
from gensim.summarization import bm25 # !pip install gensim==3.4.0
import jieba
from zhon.hanzi import punctuation

from dpr.data.qa_validation import calculate_matches
from dpr.models import init_biencoder_components
from dpr.options import (
    add_encoder_params,
    setup_args_gpu,
    print_args,
    set_encoder_params_from_state,
    add_tokenizer_params,
    add_cuda_params,
)
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
from dpr.indexer.faiss_indexers import (
    DenseIndexer,
    DenseHNSWFlatIndexer,
    DenseFlatIndexer,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
    ):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index


    def generate_question_vector(self,question):
        with torch.no_grad():
            token_tensor = [self.tensorizer.text_to_tensor(question)]
            q_ids_batch = torch.stack(token_tensor, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
            _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)
            query_vector = out.cpu().split(1, dim=0)
            query_tensor = torch.cat(query_vector, dim=0)
            return query_tensor


    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                batch_token_tensors = [
                    self.tensorizer.text_to_tensor(q)
                    for q in questions[batch_start : batch_start + bsz]
                ]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info("Encoded queries %d", len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info("Total encoded queries tensor %s", query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        return results


def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            yield question, answers


def validate(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str) -> List[List[bool]]:
    match_stats = calculate_matches(
        passages, answers, result_ctx_ids, workers_num, match_type
    )
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits


def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info("Reading data from: %s", ctx_file)
    if ctx_file.endswith(".gz"):
        with gzip.open(ctx_file, "rt") as tsvfile:
            reader = csv.reader(
                tsvfile,
                delimiter="\t",
            )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(
                tsvfile,
                delimiter="\t",
            )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    try:
                        docs[row[0]] = (row[1], row[2])
                    except:
                        pass
    return docs


def print_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    print_topk=5):

    print('==============================================')
    print('==============retrieve result=================')
    for i , q in enumerate(questions):
        print('question : ', q)
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        for k in range(print_topk):
            print(docs[k],scores[k])
    print('==============================================')


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append(
            {
                "question": q,
                "answers": q_answers,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "title": docs[c][1],
                        "text": docs[c][0],
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        )

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector



def process_passage_bm25_cn(passage):
    punctuation_str = punctuation
    passage = passage.lower()
    passage = passage.translate(str.maketrans('', '', string.punctuation))
    passage = passage.translate(str.maketrans('', '', punctuation_str))
    return passage

def build_BM25(all_passages):
    # for chinese BM25 preprocessing
    processed_all_passages = []
    pids = []
    for pid, values in all_passages.items():
        pids.append(pid)
        passage , _ = values
        passage = process_passage_bm25_cn(passage)
        passage_tokenized = list(jieba.cut(passage))
        processed_all_passages.append(passage_tokenized)

    bm25_obj = bm25.BM25(processed_all_passages)
    average_idf = sum(map(lambda k: float(bm25_obj.idf[k]), bm25_obj.idf.keys())) / len(bm25_obj.idf.keys())
    return bm25_obj, average_idf, pids



def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(
        args.encoder_model_type, args, inference_only=True
    )

    encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(
        encoder, None, args.device, args.n_gpu, args.local_rank, args.fp16
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")

    prefix_len = len("question_model.")
    question_encoder_state = {
        key[prefix_len:]: value
        for (key, value) in saved_state.model_dict.items()
        if key.startswith("question_model.")
    }
    model_to_load.load_state_dict(question_encoder_state)
    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)

    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size, args.index_buffer)
    else:
        index = DenseFlatIndexer(vector_size, args.index_buffer)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)

    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    input_paths = glob.glob(ctx_files_pattern)

    index_path = "_".join(input_paths[0].split("_")[:-1])
    if args.save_or_load_index and (
        os.path.exists(index_path) or os.path.exists(index_path + ".index.dpr")
    ):
        retriever.index.deserialize_from(index_path)
    else:
        logger.info("Reading all passages data from files: %s", input_paths)
        retriever.index.index_data(input_paths)

        if args.save_or_load_index:
            retriever.index.serialize(index_path)
    # get questions & answers
    question = args.question
    # t2s
    
    question = cc.convert(question)
    
    questions_tensor = retriever.generate_question_vector(question)
    # question_answers = [["FOR_DEMO"]]

    all_passages = load_passages(args.ctx_file)
    

    # get top k results
    # top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), len(all_passages))
    dpr_ids, dpr_scores = top_ids_and_scores[0][0], top_ids_and_scores[0][1]
    # normalize to 0-1
    dpr_scores = np.array(dpr_scores)
    dpr_scores = (dpr_scores - np.min(dpr_scores)) / (np.max(dpr_scores) - np.min(dpr_scores))
    dpr_results = dict(zip(dpr_ids, dpr_scores))

    # build BM25 and get scores 
    print('--building BM25 Object--')
    bm25_obj, average_idf, bm25_ids = build_BM25(all_passages)
    question_tokenized = list(jieba.cut(process_passage_bm25_cn(question)))
    bm25_scores = bm25_obj.get_scores(question_tokenized, average_idf)
    bm25_scores = np.array(bm25_scores)
    # normalize to 0-1
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
    bm25_results = dict(zip(bm25_ids, bm25_scores))


    mixture_results = {}
    # final score = dpr_score + alpha * bm25_score
    alpha = 0.8
    for key, bm25_score in  bm25_results.items():
        dpr_score = dpr_results[key]
        mixture_results[key] = dpr_score + alpha * bm25_score

    sorted_mixture_results = {k: v for k, v in sorted(mixture_results.items(), key=lambda item: item[1], reverse=True)}

    print(f'Question : {question}')
    for i, (pid, score) in enumerate(sorted_mixture_results.items()):
        bm25_score = bm25_results[pid]
        dpr_score = dpr_results[pid]
        print(f'top {i + 1}')
        print(all_passages[pid][0])
        print(f'BM25 score: {bm25_score} , DPR score : {dpr_score}')
        print('-------')
        if i >= args.n_docs:
            break
    exit(1)


    # if len(all_passages) == 0:
    #     raise RuntimeError(
    #         "No passages data found. Please specify ctx_file param properly."
    #     )


    # print_results(
    #     all_passages,
    #     [question],
    #     top_ids_and_scores,
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument(
        "--question",
        required=True,
        type=str,
        default=None,
        help="Question",
    )
    parser.add_argument(
        "--ctx_file",
        required=True,
        type=str,
        default=None,
        help="All passages file in the tsv format: id \\t passage_text \\t title",
    )
    parser.add_argument(
        "--encoded_ctx_file",
        type=str,
        default=None,
        help="Glob path to encoded passages (from generate_dense_embeddings tool)",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="output .tsv file path to write results to ",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="string",
        choices=["regex", "string"],
        help="Answer matching logic type",
    )
    parser.add_argument(
        "--n-docs", type=int, default=200, help="Amount of top docs to return"
    )
    parser.add_argument(
        "--validation_workers",
        type=int,
        default=16,
        help="Number of parallel processes to validate results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for question encoder forward pass",
    )
    parser.add_argument(
        "--index_buffer",
        type=int,
        default=50000,
        help="Temporal memory data buffer size (in samples) for indexer",
    )
    parser.add_argument(
        "--hnsw_index",
        action="store_true",
        help="If enabled, use inference time efficient HNSW index",
    )
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index"
    )

    args = parser.parse_args()

    assert (
        args.model_file
    ), "Please specify --model_file checkpoint to init model weights"

    setup_args_gpu(args)
    print_args(args)
    main(args)
