"""Analyze ReAct traces to discover reasoning primitives.

Runs 6 analyses (3 methods × 2 targets):
  Targets: Thoughts, Actions (Do[] descriptions)
  Methods:
    1. LLM-based categorization — ask LLM to label each text with a canonical category
    2. Sentence embeddings + clustering — embed with sentence-transformers, KMeans cluster
    3. Keyword extraction + frequency — TF-IDF keywords, count frequent n-grams

Results saved to experiments/induction/analysis/

Usage:
    source .env && export TOGETHER_API_KEY
    uv run python experiments/induction/analyze_patterns.py
"""

import json
import sys
import re
from pathlib import Path
from collections import Counter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

import typer
from secretagent import config
from secretagent.llm_util import llm

TRACES_DIR = Path(__file__).parent / 'traces'
OUTPUT_DIR = Path(__file__).parent / 'analysis'


def load_traces():
    """Load success traces and extract thoughts and actions."""
    with open(TRACES_DIR / 'success_traces.json') as f:
        traces = json.load(f)

    thoughts = []
    actions = []
    for t in traces:
        for s in t['steps']:
            if s.get('thought') and s['thought'].strip():
                thoughts.append({
                    'case': t['case_name'],
                    'step': s['step'],
                    'text': s['thought'].strip(),
                })
            if s.get('action_type') == 'do' and s.get('action_arg', '').strip():
                actions.append({
                    'case': t['case_name'],
                    'step': s['step'],
                    'text': s['action_arg'].strip(),
                })
    return traces, thoughts, actions


# ============================================================
# Method 1: LLM-based categorization
# ============================================================

def llm_categorize(items: list[dict], label: str, model: str,
                   batch_size: int = 30) -> dict:
    """Ask LLM to categorize each text into a canonical reasoning action type.

    Processes in batches to avoid context overflow and parsing failures.
    """
    all_cat_map = {}

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start:batch_start + batch_size]
        items_text = ""
        for i, item in enumerate(batch):
            global_idx = batch_start + i
            text = item['text'][:300]
            items_text += f"\n[{global_idx}] ({item['case']}, step {item['step']}): {text}\n"

        prompt = f"""You are analyzing {label} from an AI agent solving murder mystery puzzles.
Below are individual {label} the agent produced. Categorize each one into a short,
reusable REASONING ACTION TYPE (3-6 words max).

Rules:
- Use consistent, canonical names (merge synonyms into one category)
- Categories should be general enough to appear across different cases
- Focus on WHAT the agent is doing, not case-specific details
- Output ONLY a JSON array with "index" and "category" fields, nothing else

{label.capitalize()} to categorize:
{items_text}

<answer>
[{{"index": {batch_start}, "category": "your category"}}, ...]
</answer>"""

        response, stats = llm(prompt, model)

        # Try multiple parsing strategies
        json_str = None
        match = re.search(r'<answer>(.*)</answer>', response, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        if not json_str:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
        if not json_str:
            print(f'    WARNING: Could not parse LLM response for batch {batch_start}, skipping')
            continue

        try:
            categories = json.loads(json_str)
            for c in categories:
                all_cat_map[c['index']] = c['category']
        except json.JSONDecodeError:
            print(f'    WARNING: JSON parse failed for batch {batch_start}, skipping')
            continue

    categorized = []
    for i, item in enumerate(items):
        categorized.append({**item, 'category': all_cat_map.get(i, 'unknown')})

    freq = Counter(c['category'] for c in categorized)
    return {
        'method': 'llm_categorization',
        'target': label,
        'categorized': categorized,
        'frequency': freq.most_common(),
        'n_categories': len(freq),
    }


# ============================================================
# Method 2: Sentence embeddings + clustering
# ============================================================

def embedding_cluster(items: list[dict], label: str, n_clusters: int = 4) -> dict:
    """Embed texts with sentence-transformers, cluster with KMeans."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [item['text'] for item in items]
    embeddings = model.encode(texts)

    # Determine k: min of n_clusters and number of items
    k = min(n_clusters, len(texts))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # For each cluster, find the most representative text (closest to centroid)
    clusters = {}
    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_texts = [texts[i] for i in cluster_indices]

        # Find centroid-closest text as representative
        centroid = kmeans.cluster_centers_[cluster_id]
        dists = np.linalg.norm(embeddings[cluster_indices] - centroid, axis=1)
        rep_idx = cluster_indices[np.argmin(dists)]

        # Extract keywords from cluster using TF-IDF within cluster
        if len(cluster_texts) >= 2:
            tfidf = TfidfVectorizer(stop_words='english', max_features=5)
            tfidf.fit(cluster_texts)
            keywords = list(tfidf.get_feature_names_out())
        else:
            keywords = cluster_texts[0].split()[:5]

        clusters[int(cluster_id)] = {
            'size': int(mask.sum()),
            'keywords': keywords,
            'representative': texts[rep_idx],
            'members': [{'case': items[i]['case'], 'step': items[i]['step'],
                         'text': items[i]['text']} for i in cluster_indices],
        }

    categorized = []
    for i, item in enumerate(items):
        categorized.append({
            **item,
            'cluster': int(labels[i]),
            'cluster_keywords': clusters[int(labels[i])]['keywords'],
        })

    return {
        'method': 'embedding_clustering',
        'target': label,
        'n_clusters': k,
        'clusters': clusters,
        'categorized': categorized,
    }


# ============================================================
# Method 3: Keyword extraction + frequency counting
# ============================================================

def keyword_frequency(items: list[dict], label: str) -> dict:
    """Extract keywords via TF-IDF and count frequent n-grams."""
    texts = [item['text'] for item in items]

    # Unigram + bigram TF-IDF
    tfidf = TfidfVectorizer(
        stop_words='english', ngram_range=(1, 3),
        max_features=50, min_df=1)
    tfidf_matrix = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()

    # Get top terms by mean TF-IDF score across documents
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[::-1]
    top_terms = [(feature_names[i], float(mean_scores[i])) for i in top_indices[:20]]

    # Also do raw frequency counting of bigrams/trigrams
    count_vec = CountVectorizer(
        stop_words='english', ngram_range=(2, 3),
        max_features=30, min_df=1)
    count_matrix = count_vec.fit_transform(texts)
    count_names = count_vec.get_feature_names_out()
    total_counts = np.asarray(count_matrix.sum(axis=0)).flatten()
    freq_ngrams = sorted(
        zip(count_names, total_counts.tolist()),
        key=lambda x: x[1], reverse=True)[:15]

    # Per-item: top 3 keywords
    categorized = []
    for i, item in enumerate(items):
        row = tfidf_matrix[i].toarray().flatten()
        top3 = [feature_names[j] for j in row.argsort()[::-1][:3]]
        categorized.append({**item, 'top_keywords': top3})

    return {
        'method': 'keyword_frequency',
        'target': label,
        'top_tfidf_terms': top_terms,
        'frequent_ngrams': freq_ngrams,
        'categorized': categorized,
    }


# ============================================================
# Main
# ============================================================

app = typer.Typer()

@app.command()
def main(
    model: str = typer.Option(
        "together_ai/deepseek-ai/DeepSeek-V3",
        help="LLM model for categorization"),
    n_clusters: int = typer.Option(4, help="Number of clusters for embedding method"),
):
    """Run all 6 pattern analyses on ReAct traces."""
    config.configure(cfg={
        'llm': {'model': model, 'timeout': 300},
        'cachier': {
            'cache_dir': str(TRACES_DIR / 'llm_cache'),
            'enable_caching': True,
        },
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    traces, thoughts, actions = load_traces()
    print(f'Loaded {len(traces)} success traces')
    print(f'  {len(thoughts)} thought steps')
    print(f'  {len(actions)} action steps (Do[...])\n')

    all_results = {}

    # Run all 6 analyses
    for target_name, items in [('thoughts', thoughts), ('actions', actions)]:
        print(f'{"="*60}')
        print(f'Analyzing {target_name.upper()} ({len(items)} items)')
        print(f'{"="*60}\n')

        # Method 1: LLM categorization
        print(f'  [1/3] LLM categorization...')
        llm_result = llm_categorize(items, target_name, model)
        all_results[f'{target_name}_llm'] = llm_result
        print(f'    {llm_result["n_categories"]} categories found:')
        for cat, count in llm_result['frequency'][:8]:
            print(f'      {count:2d}x  {cat}')
        print()

        # Method 2: Embedding clustering
        print(f'  [2/3] Embedding clustering (k={n_clusters})...')
        emb_result = embedding_cluster(items, target_name, n_clusters)
        all_results[f'{target_name}_embedding'] = emb_result
        for cid, cluster in sorted(emb_result['clusters'].items()):
            print(f'    Cluster {cid} ({cluster["size"]} items): '
                  f'keywords={cluster["keywords"]}')
            print(f'      representative: "{cluster["representative"][:80]}..."')
        print()

        # Method 3: Keyword frequency
        print(f'  [3/3] Keyword frequency...')
        kw_result = keyword_frequency(items, target_name)
        all_results[f'{target_name}_keywords'] = kw_result
        print(f'    Top TF-IDF terms:')
        for term, score in kw_result['top_tfidf_terms'][:10]:
            print(f'      {score:.3f}  {term}')
        print(f'    Top n-grams:')
        for ngram, count in kw_result['frequent_ngrams'][:8]:
            print(f'      {count:2.0f}x  {ngram}')
        print()

    # Save all results
    output_file = OUTPUT_DIR / 'pattern_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nAll results saved to {output_file}')

    # Write summary report
    report = generate_report(all_results, thoughts, actions)
    report_file = OUTPUT_DIR / 'report.md'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f'Report saved to {report_file}')


def generate_report(results: dict, thoughts: list, actions: list) -> str:
    """Generate a markdown report summarizing all analyses."""
    lines = [
        "# Inductive Ptool Discovery: Pattern Analysis Report",
        "",
        f"Analyzed **{len(thoughts)} thoughts** and **{len(actions)} actions** "
        f"from successful ReAct traces on MUSR murder mysteries.",
        "",
    ]

    for target in ['thoughts', 'actions']:
        lines.append(f"## {target.capitalize()}")
        lines.append("")

        # LLM
        llm_r = results[f'{target}_llm']
        lines.append("### Method 1: LLM Categorization")
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for cat, count in llm_r['frequency']:
            lines.append(f"| {cat} | {count} |")
        lines.append("")

        # Embedding
        emb_r = results[f'{target}_embedding']
        lines.append("### Method 2: Embedding Clustering")
        lines.append("")
        for cid, cluster in sorted(emb_r['clusters'].items()):
            kw = ", ".join(cluster['keywords'])
            lines.append(f"**Cluster {cid}** ({cluster['size']} items) — keywords: {kw}")
            lines.append(f"> Representative: \"{cluster['representative'][:120]}\"")
            lines.append("")

        # Keywords
        kw_r = results[f'{target}_keywords']
        lines.append("### Method 3: Keyword Frequency")
        lines.append("")
        lines.append("Top TF-IDF terms:")
        lines.append("")
        lines.append("| Term | Score |")
        lines.append("|------|-------|")
        for term, score in kw_r['top_tfidf_terms'][:10]:
            lines.append(f"| {term} | {score:.3f} |")
        lines.append("")
        lines.append("Top n-grams:")
        lines.append("")
        lines.append("| N-gram | Count |")
        lines.append("|--------|-------|")
        for ngram, count in kw_r['frequent_ngrams'][:8]:
            lines.append(f"| {ngram} | {int(count)} |")
        lines.append("")

    lines.extend([
        "## Convergence Across Methods",
        "",
        "All three methods should converge on the same core reasoning primitives. "
        "Look for patterns that appear in LLM categories, embedding clusters, "
        "AND keyword frequency — those are the strongest candidates for ptools.",
    ])

    return "\n".join(lines)


if __name__ == '__main__':
    app()
