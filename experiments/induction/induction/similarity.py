"""Ptool orthogonality checking via embedding similarity.

Modes:
  docstring  — cosine sim of docstring embeddings
  functional — cosine sim of actual ptool outputs on sample narratives
  name       — cosine sim of function name embeddings
  combined   — average of all enabled modes
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from .ptool_registry import _INDUCED_INTERFACES, create_ptool_interface

_EMBED_MODEL = None
_FUNCTIONAL_CACHE: dict[str, list[str]] = {}
_SAMPLE_NARRATIVES: list[str] = []


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBED_MODEL


def _get_sample_narratives(n: int = 3) -> list[str]:
    global _SAMPLE_NARRATIVES
    if not _SAMPLE_NARRATIVES:
        from .data import load_murder_cases
        cases = load_murder_cases(n=10, seed=99)
        _SAMPLE_NARRATIVES = [c['narrative'] for c in cases[:n]]
    return _SAMPLE_NARRATIVES


def _embed_similarity(texts_a: list[str], texts_b: list[str]) -> float:
    """Max cosine similarity between text_a and any text in texts_b."""
    if not texts_b:
        return 0.0
    model = _get_embed_model()
    all_texts = texts_a + texts_b
    embeddings = model.encode(all_texts, normalize_embeddings=True)
    a_emb = embeddings[:len(texts_a)]
    b_emb = embeddings[len(texts_a):]
    # Max sim between any a and any b
    sims = a_emb @ b_emb.T
    return float(np.max(sims))


def docstring_similarity(new_ptool: dict, existing_ptools: list[dict]) -> float:
    if not existing_ptools:
        return 0.0
    return _embed_similarity([new_ptool['doc']], [p['doc'] for p in existing_ptools])


def name_similarity(new_ptool: dict, existing_ptools: list[dict]) -> float:
    """Cosine similarity of function names (snake_case + display_name)."""
    if not existing_ptools:
        return 0.0
    new_texts = [f"{new_ptool['func_name']} {new_ptool['display_name']}"]
    existing_texts = [f"{p['func_name']} {p['display_name']}" for p in existing_ptools]
    return _embed_similarity(new_texts, existing_texts)


def functional_similarity(new_ptool: dict, existing_ptools: list[dict]) -> float:
    """Cosine similarity of actual ptool outputs on sample narratives."""
    if not existing_ptools:
        return 0.0

    new_outputs = _get_ptool_outputs(new_ptool)
    new_text = " ".join(new_outputs)

    max_sim = 0.0
    for ep in existing_ptools:
        ep_outputs = _get_ptool_outputs(ep)
        ep_text = " ".join(ep_outputs)
        model = _get_embed_model()
        embeddings = model.encode([new_text, ep_text], normalize_embeddings=True)
        sim = float(embeddings[0] @ embeddings[1])
        max_sim = max(max_sim, sim)
    return max_sim


def _get_ptool_outputs(ptool: dict) -> list[str]:
    """Run ptool on sample narratives with ptool-specific focus args."""
    cache_key = ptool['id']
    if cache_key in _FUNCTIONAL_CACHE:
        return _FUNCTIONAL_CACHE[cache_key]

    iface = _INDUCED_INTERFACES.get(ptool['func_name'])
    if iface is None:
        iface = create_ptool_interface(ptool['func_name'], ptool['doc'])

    narratives = _get_sample_narratives()
    focus_args = []
    if ptool.get('examples'):
        for ex in ptool['examples'][:3]:
            focus_args.append(ex[:100])
    while len(focus_args) < len(narratives):
        focus_args.append(ptool.get('short_desc', 'analyze'))

    outputs = []
    for narrative, focus in zip(narratives, focus_args):
        try:
            result = iface(narrative=narrative, focus=focus)
            outputs.append(str(result)[:500])
        except Exception:
            outputs.append("")

    _FUNCTIONAL_CACHE[cache_key] = outputs
    return outputs


def compute_similarity(new_ptool: dict, existing_ptools: list[dict],
                       mode: str = 'combined') -> tuple[float, dict]:
    """Compute similarity with breakdown.

    mode: 'docstring', 'functional', 'name', 'combined', or comma-separated like 'docstring,name'
    """
    components = mode.split(',') if ',' in mode else [mode]

    if mode == 'combined':
        components = ['docstring', 'functional', 'name']

    scores = {}
    for c in components:
        c = c.strip()
        if c == 'docstring':
            scores['docstring'] = docstring_similarity(new_ptool, existing_ptools)
        elif c == 'functional':
            scores['functional'] = functional_similarity(new_ptool, existing_ptools)
        elif c == 'name':
            scores['name'] = name_similarity(new_ptool, existing_ptools)

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    scores['combined'] = avg
    return avg, scores
