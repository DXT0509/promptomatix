from sentence_transformers import SentenceTransformer, util
import torch
from typing import List
from sentence_transformers.util import cos_sim
import os
from dotenv import load_dotenv
from bert_score import score as bert_score_original
import warnings
import sys
_st_model = SentenceTransformer("all-mpnet-base-v2")
from contextlib import contextmanager
@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def bert_score_silent(*args, **kwargs):
    """BERTScore wrapper that suppresses all stderr output."""
    with suppress_stderr():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bert_score_original(*args, **kwargs)
        
bert_score_metric = bert_score_silent

def bert_score_metric1(cands: List[str], refs: List[str], **kwargs):
    """SentenceTransformer similarity wrapper mimicking BERTScore output signature.
    Returns (P,R,F1) tensors all equal to cosine similarity scores.
    Broadcasting: if lengths differ and one list has length 1, broadcast that element.
    If lengths differ otherwise, pairwise similarities matrix is reduced by max per candidate.
    """
    try:
        if not cands:
            cands = [""]
        if not refs:
            refs = [""]
        # Broadcast single element list
        if len(cands) != len(refs):
            if len(refs) == 1:
                refs = refs * len(cands)
            elif len(cands) == 1:
                cands = cands * len(refs)
        cand_emb = _st_model.encode(cands, convert_to_tensor=True)
        ref_emb = _st_model.encode(refs, convert_to_tensor=True)
        sim_matrix = cos_sim(cand_emb, ref_emb)  # shape (Nc, Nr)
        if sim_matrix.shape[0] == sim_matrix.shape[1]:
            sims_tensor = torch.diag(sim_matrix)
        else:
            # Reduce to a single similarity per candidate (take max over refs)
            sims_tensor = torch.max(sim_matrix, dim=1).values
        return sims_tensor, sims_tensor, sims_tensor
    except Exception:
        sims_tensor = torch.zeros(len(cands) if cands else 1, dtype=torch.float32)
        return sims_tensor, sims_tensor, sims_tensor


# ===== TEST THỰC TẾ =====
pred_answer = 'The quick brown fox jumps over the lazy dog'
gold_answer = 'A brown fox quickly jumps over a lazy dog'
load_dotenv()  # bắt buộc


P, R, F1 = bert_score_metric1(pred_answer, gold_answer)
print("Score:", float(F1.mean()), "\n")