"""生成一组用于跑通实验的样例 PDF（无需联网、无需真实论文）。

为什么需要它：`run_experiments.py` 需要先有文献入库才能评测。如果没有现成的
PDF、又连不上 arXiv，可以用这个小脚本造几篇主题相近、细节不同的论文，
保证消融实验（dense vs hybrid vs rerank）能看出差异。

用法：
    python scripts/make_sample_papers.py            # 输出到 data/sample_papers/
    python scripts/make_sample_papers.py --out my_papers
"""

import argparse
import textwrap
from pathlib import Path

import pymupdf as fitz

# 三篇同主题（RAG 检索）但方法/数据不同的论文，便于做对比与消融
PAPERS = [
    {
        "filename": "paper_dense_retrieval.pdf",
        "title": "Dense Passage Retrieval for Academic Search",
        "authors": "A. Author, B. Author",
        "abstract": (
            "We study dense embedding retrieval for academic literature search. "
            "The method encodes queries and passages with a bi-encoder and ranks "
            "passages by cosine similarity. On our benchmark of 200 literature "
            "questions, dense retrieval reaches a hit rate of sixty percent at "
            "top five and a mean reciprocal rank of 0.48."
        ),
        "method": (
            "The bi-encoder uses nomic-embed-text with 768 dimensions. Passages "
            "are chunked to one thousand characters with one hundred fifty "
            "characters of overlap. Retrieval returns the top twenty candidates "
            "from LanceDB cosine search. No lexical matching is applied, so exact "
            "terms such as BM25 or RRF may be missed."
        ),
        "results": (
            "Dense retrieval reports hit rate 60 percent, recall 55 percent and "
            "mean reciprocal rank 0.48. Failure analysis shows that acronym queries "
            "and formula names are frequently missed because the embedding space "
            "does not separate rare tokens well."
        ),
    },
    {
        "filename": "paper_hybrid_rrf.pdf",
        "title": "Hybrid Retrieval with Reciprocal Rank Fusion",
        "authors": "C. Author, D. Author",
        "abstract": (
            "We combine BM25 sparse lexical retrieval with dense embedding "
            "retrieval and fuse both ranked lists using Reciprocal Rank Fusion. "
            "On the same benchmark, hybrid retrieval improves hit rate from sixty "
            "to eighty five percent and raises mean reciprocal rank to 0.71."
        ),
        "method": (
            "The BM25 index is rebuilt in memory from the vector store contents. "
            "Dense candidates come from LanceDB cosine search. RRF uses k equal to "
            "sixty by default and requires no score normalisation. The fused list "
            "keeps twenty candidates before reranking. Chinese text is tokenised "
            "into character bigrams and English text into words."
        ),
        "results": (
            "Hybrid retrieval reports hit rate 85 percent, recall 80 percent and "
            "mean reciprocal rank 0.71. The gain concentrates on acronym and "
            "formula queries, where BM25 recovers exact matches that dense "
            "retrieval misses."
        ),
    },
    {
        "filename": "paper_cross_encoder_rerank.pdf",
        "title": "Cross Encoder Reranking for Literature QA",
        "authors": "E. Author, F. Author",
        "abstract": (
            "We add a cross encoder reranker on top of hybrid retrieval. The "
            "reranker scores each query passage pair jointly and keeps the top "
            "three passages. On our benchmark, reranking raises mean reciprocal "
            "rank from 0.71 to 0.83 while keeping hit rate at eighty five percent."
        ),
        "method": (
            "The reranker is a small cross encoder with a maximum length of 512 "
            "tokens. Candidate passages are reranked and truncated to the top three "
            "before prompt construction. When the local model is unavailable the "
            "system falls back to an LLM scorer that rates passages from zero to "
            "ten in batches of eight."
        ),
        "results": (
            "Reranking reports mean reciprocal rank 0.83 and hit rate 85 percent. "
            "The improvement is largest for multi hop questions where the first "
            "retrieved passage is only partially relevant."
        ),
    },
]

# 干扰文献：主题相关但不回答问题，用来增大 top-k 的区分度
# 没有它们时只有 10 个片段，任何检索都能全命中，消融实验看不出差异
DISTRACTORS = [
    {
        "filename": "distractor_survey.pdf",
        "title": "A Survey of Retrieval Augmented Generation",
        "authors": "G. Author",
        "abstract": (
            "This survey reviews retrieval augmented generation systems. We "
            "categorise prior work by retriever type, fusion strategy and "
            "generation model, and discuss open challenges in evaluation."
        ),
        "method": (
            "We group methods into sparse, dense and hybrid retrievers, and "
            "compare them qualitatively. No new experiments are conducted."
        ),
        "results": (
            "The survey concludes that hybrid retrieval is common but that "
            "evaluation protocols differ widely across papers."
        ),
    },
    {
        "filename": "distractor_eval.pdf",
        "title": "Evaluating Retrieval Quality with Synthetic Questions",
        "authors": "H. Author",
        "abstract": (
            "We study how to evaluate retrieval quality using synthetic questions "
            "generated from passages. We report that synthetic questions inflate "
            "recall compared to human written questions."
        ),
        "method": (
            "Questions are generated by a language model from each passage, and "
            "recall at k is computed against the source passage."
        ),
        "results": (
            "Synthetic questions report recall 95 percent, while human questions "
            "report recall 70 percent, indicating a large optimism bias."
        ),
    },
    {
        "filename": "distractor_chunking.pdf",
        "title": "The Effect of Chunk Size on Retrieval",
        "authors": "I. Author",
        "abstract": (
            "We measure how chunk size affects retrieval quality. Small chunks "
            "improve precision but break long arguments, while large chunks "
            "improve recall but dilute relevance signals."
        ),
        "method": (
            "We sweep chunk sizes from two hundred to two thousand characters "
            "with overlap from zero to two hundred characters."
        ),
        "results": (
            "Chunks of one thousand characters with one hundred fifty overlap "
            "give the best balance of recall and precision."
        ),
    },
    {
        "filename": "distractor_prompting.pdf",
        "title": "Prompt Design for Grounded Question Answering",
        "authors": "J. Author",
        "abstract": (
            "We compare prompt designs for grounded question answering. Prompts "
            "that require citations reduce hallucination but also reduce answer "
            "completeness."
        ),
        "method": (
            "Three prompt variants are compared: plain, citation required and "
            "citation with refusal instruction."
        ),
        "results": (
            "Citation required prompts reduce hallucination by half at the cost of "
            "ten percent completeness."
        ),
    },
    {
        "filename": "distractor_latency.pdf",
        "title": "Latency Aware Reranking for Production Search",
        "authors": "K. Author",
        "abstract": (
            "We study latency of reranking in production search. Cross encoders "
            "add twenty milliseconds per query on commodity hardware but improve "
            "precision substantially."
        ),
        "method": (
            "We benchmark batch sizes from one to sixteen and report p99 latency "
            "for each configuration."
        ),
        "results": (
            "Batching eight passages keeps p99 latency under one hundred "
            "milliseconds while retaining most of the precision gain."
        ),
    },
    {
        # 关键干扰项：大量出现 BM25 / RRF / k 等词，但讲的是别的东西，
        # 专门用来考验「精确术语匹配」——纯向量容易把它和真正讲 RRF 的论文混淆
        "filename": "distractor_bm25_classic.pdf",
        "title": "A Note on Term Weighting for Classic BM25",
        "authors": "L. Author",
        "abstract": (
            "We revisit term weighting in classic BM25 for legacy document "
            "collections. Our note focuses on stopword handling and stemming "
            "rather than on fusion."
        ),
        "method": (
            "We vary the BM25 parameters k1 and b and report how term weighting "
            "changes on an old news corpus. No reciprocal rank fusion is involved."
        ),
        "results": (
            "We find that k1 around one point two and b around zero point seven "
            "five are robust for the legacy corpus, with no effect on modern "
            "retrieval pipelines."
        ),
    },
]


def _write_paper(paper: dict, out_dir: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 90

    page.insert_text((60, y), paper["title"], fontsize=18)
    y += 28
    page.insert_text((60, y), paper["authors"], fontsize=11)
    y += 40

    for heading, body in (
        ("Abstract", paper["abstract"]),
        ("1 Introduction", paper["abstract"]),
        ("2 Method", paper["method"]),
        ("3 Results", paper["results"]),
    ):
        page.insert_text((60, y), heading, fontsize=13)
        y += 20
        for line in textwrap.wrap(body, width=88):
            page.insert_text((60, y), line, fontsize=10.5)
            y += 14
        y += 16
        if y > 760:
            page = doc.new_page(width=595, height=842)
            y = 90

    out_path = out_dir / paper["filename"]
    doc.save(out_path)
    doc.close()
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成实验用样例 PDF")
    parser.add_argument(
        "--out", default=None, help="输出目录（默认 data/sample_papers）"
    )
    parser.add_argument(
        "--with-distractors",
        action="store_true",
        help="额外生成 5 篇干扰文献（让 top-k 检索有区分度，消融实验才看得出差异）",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out) if args.out else root / "data" / "sample_papers"
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = list(PAPERS) + (list(DISTRACTORS) if args.with_distractors else [])
    for paper in papers:
        path = _write_paper(paper, out_dir)
        print(f"已生成：{path}")
    print(f"\n共 {len(papers)} 篇，输出目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
