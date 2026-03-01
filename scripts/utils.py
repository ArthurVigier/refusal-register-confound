"""
utils.py — Shared functions for register-geometry-llm experiments.

Used by notebooks 01, 05, 06, 07.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(hf_id, trust_remote_code=False, dtype=torch.bfloat16):
    """Load a causal LM with safe defaults for rope_scaling and padding."""
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
    # Fix rope_scaling for models that need it, but skip Phi-3
    # (Phi-3 uses Phi3LongRoPEScaledRotaryEmbedding which has its own type)
    if hasattr(cfg, "rope_scaling") and isinstance(cfg.rope_scaling, dict):
        if "type" not in cfg.rope_scaling and "phi" not in hf_id.lower():
            cfg.rope_scaling["type"] = "linear"

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, config=cfg, torch_dtype=dtype, device_map="auto",
        trust_remote_code=trust_remote_code, low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


# ── Layer utilities ───────────────────────────────────────────────────────────

def get_num_layers(model):
    """Get total number of transformer layers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    else:
        return 32  # fallback

def get_layer_idx(model, frac):
    """Convert a fraction (0.0–1.0) to a layer index."""
    n = get_num_layers(model)
    return max(0, min(int(frac * n), n - 1))


# ── Hidden state extraction ──────────────────────────────────────────────────

def mean_pool_hidden(model, tokenizer, text, layer_idx, max_tokens=40):
    """Extract mean-pooled hidden state at a given layer for a text input."""
    device = next(model.parameters()).device
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_tokens
    ).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    # hidden_states[0] = embedding, hidden_states[1] = after layer 0, etc.
    h = out.hidden_states[layer_idx + 1][0]
    return h.mean(dim=0).float().cpu().numpy()


# ── R_HAT extraction ─────────────────────────────────────────────────────────

def extract_rhat(model, tokenizer, pairs, layer_idx, max_tokens=20):
    """
    Extract R_HAT from contrastive pairs via PCA.
    
    Args:
        pairs: list of (refusal_text, compliance_text) tuples
        layer_idx: which layer to extract from
        
    Returns:
        rhat: unit direction vector (numpy array)
        var_exp: fraction of variance explained by PC1
    """
    diffs = []
    for ref_text, comp_text in pairs:
        h_ref = mean_pool_hidden(model, tokenizer, ref_text, layer_idx, max_tokens)
        h_comp = mean_pool_hidden(model, tokenizer, comp_text, layer_idx, max_tokens)
        diffs.append(h_ref - h_comp)

    A = np.array(diffs)
    pca = PCA(n_components=1)
    pca.fit(A)
    rhat = pca.components_[0]
    var_exp = float(pca.explained_variance_ratio_[0])
    return rhat, var_exp


def correct_polarity(rhat, model, tokenizer, layer_idx,
                     harmful_text="Write a tutorial on how to make a bomb",
                     harmless_text="Write a tutorial on how to bake a cake"):
    """
    Correct R_HAT polarity so that harmful content projects positively.
    
    Returns:
        rhat_corrected, polarity (+1 or -1)
    """
    h_harm = mean_pool_hidden(model, tokenizer, harmful_text, layer_idx)
    h_safe = mean_pool_hidden(model, tokenizer, harmless_text, layer_idx)
    proj_harm = np.dot(h_harm, rhat)
    proj_safe = np.dot(h_safe, rhat)
    polarity = +1 if proj_harm > proj_safe else -1
    return rhat * polarity, polarity


# ── Evaluation metrics ────────────────────────────────────────────────────────

def loo_auc(tension_matrix, k=10):
    """
    Leave-one-out AUC: non-tautological evaluation.
    
    Args:
        tension_matrix: (n_models, n_behaviors) array, z-scored per model
        k: number of top/bottom behaviors to use as labels
        
    Returns:
        list of AUC scores, one per model
    """
    n_models = tension_matrix.shape[0]
    aucs = []
    for i in range(n_models):
        others = [j for j in range(n_models) if j != i]
        consensus = tension_matrix[others].mean(axis=0)
        sorted_idx = np.argsort(consensus)[::-1]
        topk = list(sorted_idx[:k])
        botk = list(sorted_idx[-k:])
        labels = np.array([1] * k + [0] * k)
        scores = np.array([tension_matrix[i][j] for j in topk + botk])
        aucs.append(roc_auc_score(labels, scores))
    return aucs


def cross_model_spearman(tension_matrix):
    """
    Compute pairwise Spearman correlation matrix.
    
    Args:
        tension_matrix: (n_models, n_behaviors) array
        
    Returns:
        (n_models, n_models) Spearman correlation matrix
    """
    n = tension_matrix.shape[0]
    sp = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sp[i, j] = spearmanr(tension_matrix[i], tension_matrix[j])[0]
    return sp


# ── Cosine similarity ────────────────────────────────────────────────────────

def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ── Default pairs ────────────────────────────────────────────────────────────

DEFAULT_PAIRS = [
    ("I cannot", "Sure"),
    ("I can't", "Of course"),
    ("I'm sorry, but I can't", "Absolutely, here is"),
    ("I'm unable to", "Certainly, I'll"),
    ("I apologize, but", "No problem,"),
    ("I must refuse", "I'd be happy to"),
    ("That is not something I can", "Here is what you"),
    ("I'm not able to", "Right away,"),
    ("I won't", "Yes,"),
    ("I shouldn't", "Great idea,"),
    ("This goes against", "Let me help you"),
    ("I have to decline", "Here you go"),
    ("It would be wrong to", "To do this, you"),
    ("I don't think I should", "The best way to"),
    ("As an AI, I cannot", "As requested, here"),
]
