import os, csv, argparse, random, json, logging, gc
from collections import defaultdict

import torch
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, RobertaTokenizerFast, AutoModel, GPT2Tokenizer, RobertaForMaskedLM
import pickle
from pyinflect import getAllInflections

import spacy

from tqdm import tqdm
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from datasets import load_dataset # huggingface
from math import exp

blimp_configs = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question', 'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']
        

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

# Tokenizer Parameters
parser.add_argument("--model_name", type=str, help="architecture e.g. gpt2", default="gpt2")
parser.add_argument("--random_seed", type=int, help="random seed of model")
parser.add_argument("--epoch", type=int, help="epoch of the checkpoint", default=0) # not required if evaluating from pretrained
parser.add_argument("--eval_set", type=str, help="evaluation set", default="blimp")
parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="whether to use pretrained model")
parser.add_argument("--not-pretrained", dest="pretrained", action="store_false", help="whether to use pretrained model")
parser.set_defaults(pretrained=False)
args = parser.parse_args()

log_file_name = f"{args.model_name}-{args.random_seed}-epoch={args.epoch}-pretrained={args.pretrained}-evalon{args.eval_set}"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(),logging.FileHandler("logs/" + log_file_name + ".log")])
logging = logging.getLogger(__name__)
logging.info(args)

def load_tokenizer_model(model_name, random_seed, epoch, pretrained=False):
    if pretrained:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        logging.info(f"Loading pretrained gpt2 model")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    else:
        model = GPT2LMHeadModel.from_pretrained(f"model_{random_seed}_epoch{epoch}")
        logging.info(f"Loading model from model_{random_seed}_epoch{epoch}")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def load_eval_dataset(eval_set, blimp_config):
    if eval_set == "blimp":
        dataset = load_dataset("nyu-mll/blimp", blimp_config)["train"]
    return dataset

def evaluate(dataset):
    res = []
    for d in tqdm(dataset):
        good_prob = log_prob(d["sentence_good"])
        bad_prob = log_prob(d["sentence_bad"])
        d["good_prob"] = good_prob
        d["bad_prob"] = bad_prob
        d["correct"] = 1 if good_prob > bad_prob else 0
        res.append(d)
    return res
            

def log_prob(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits

        # Shift logits and labels to align
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log-probability of the correct next token
        # This gives shape (batch_size, seq_len - 1)
        target_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Sum to get total log-probability
        total_log_prob = target_log_probs.sum().item()

    return total_log_prob

        

if __name__ == "__main__":
    tokenizer, model = load_tokenizer_model(args.model_name, args.random_seed, args.epoch, pretrained=args.pretrained)
    for conf in blimp_configs:
        dataset = load_eval_dataset(args.eval_set, conf)
        res = evaluate(dataset)
        res_fn = f"results/blimp_{conf}_{args.random_seed}.csv"
        logging.info(f"Saving results to {res_fn}")
        pd.DataFrame(res).to_csv(res_fn, index=False)