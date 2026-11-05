"""检索质量评测：hit rate / recall@k / MRR / 关键词命中率。

RAG 系统的效果好坏，八成取决于「检索到没检索到」。这个脚本把「问一个问题，
期望命中哪篇文献」变成可重复的指标，方便做参数对比（换切分大小、换 embedding、
加不加 rerank 前后各跑一次）。

用法（项目根目录执行，并保证 Ollama 与文献已入库）：

    python -m rag.evaluation                                  # 用自带示例数据集
    python -m rag.evaluation --top-k 5 --dump                 # 看每个问题命中了哪些片段
    python -m rag.evaluation --dataset my_questions.json --json report.json

    # 基线 vs 增强 对比（答辩直接用这张表）
    python -m rag.evaluation --mode dense --no-rerank
    python -m rag.evaluation --mode hybrid --rerank cross-encoder

数据集格式（JSON，`expected_sources` 支持包含匹配，可用 source_key / 文件名 / 标题）：

    {
      "cases": [
        {
          "question": "这篇论文的核心贡献是什么？",
          "expected_sources": ["arxiv:2401.00001"],
          "expected_keywords": ["retrieval", "embedding"]
        }
      ]
    }
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rag.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NUM_DOCUMENTS,
)
from rag.knowledge_base import RAGKnowledgeBase
from rag.vector_store import vector_store_available, vector_store_hint

DEFAULT_DATASET = Path(__file__).resolve().parent / "eval_questions.example.json"


@dataclass
class CaseResult:
    question: str
    rank: int = 0                 # 首次命中的名次；0 表示 top-k 内没命中
    matched: int = 0              # 命中的期望来源数
    expected: int = 0             # 期望来源总数
    keyword_hit: float = 0.0      # 检索片段覆盖了多少期望关键词
    precision: float = 0.0        # top-k 里有多少片段来自期望来源（检索精度）
    retrieved: List[str] = field(default_factory=list)
    keywords_expected: int = 0    # 该题标注了多少个期望关键词；0 表示不参与关键词指标


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到评测数据集：{path}")
    # utf-8-sig：兼容 Windows 记事本 / Excel 导出的带 BOM 的 JSON
    with path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)

    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not cases:
        raise ValueError(f"数据集里没有 cases：{path}")

    cleaned: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        question = (case.get("question") or "").strip()
        if not question:
            raise ValueError(f"第 {index} 条 case 缺少 question")
        cleaned.append(
            {
                "question": question,
                "expected_sources": [str(item) for item in case.get("expected_sources") or []],
                "expected_keywords": [str(item) for item in case.get("expected_keywords") or []],
            }
        )
    return cleaned


def _meta(document) -> Dict[str, Any]:
    return dict(getattr(document, "metadata", None) or {})


def _name(document) -> str:
    meta = _meta(document)
    return str(meta.get("name") or getattr(document, "id", None) or "-")


def _candidates(document) -> List[str]:
    meta = _meta(document)
    values = [
        meta.get("source_key"),
        meta.get("source"),
        meta.get("title"),
        _name(document),
    ]
    return [str(value).lower() for value in values if value]


def _matches(document, expected_sources: List[str]) -> bool:
    candidates = _candidates(document)
    for expected in expected_sources:
        needle = expected.strip().lower()
        if needle and any(needle in candidate for candidate in candidates):
            return True
    return False


def _keyword_hit(documents, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    haystack = "\n".join(document.page_content or "" for document in documents).lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in haystack)
    return hits / len(keywords)


def evaluate_cases(
    cases: List[Dict[str, Any]],
    knowledge: RAGKnowledgeBase,
    top_k: int,
    dump: bool = False,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[CaseResult]:
    results: List[CaseResult] = []

    for index, case in enumerate(cases, start=1):
        documents = knowledge.search(case["question"], num_documents=top_k)

        first_rank = 0
        matched_sources = set()
        for rank, document in enumerate(documents, start=1):
            if not _matches(document, case["expected_sources"]):
                continue
            if first_rank == 0:
                first_rank = rank
            meta = _meta(document)
            key = meta.get("source_key") or _name(document) or f"rank-{rank}"
            matched_sources.add(str(key))

        retrieved = []
        for document in documents:
            meta = _meta(document)
            retrieved.append(
                " · ".join(
                    str(part)
                    for part in (meta.get("source"), meta.get("section"), meta.get("page"))
                    if part
                )
                or _name(document)
            )

        result = CaseResult(
            question=case["question"],
            rank=first_rank,
            matched=len(matched_sources),
            expected=len(case["expected_sources"]),
            keyword_hit=_keyword_hit(documents, case["expected_keywords"]),
            precision=(
                sum(1 for document in documents if _matches(document, case["expected_sources"]))
                / len(documents)
                if documents and case["expected_sources"]
                else 0.0
            ),
            retrieved=retrieved,
            keywords_expected=len(case["expected_keywords"]),
        )
        results.append(result)

        if on_progress is not None:
            try:
                on_progress(index, len(cases), result.question)
            except Exception:  # noqa: BLE001 - 进度回调不能拖垮评测
                pass

        if dump:
            print(f"\n[问题] {result.question}")
            if not documents:
                print("  没有检索到任何片段")
            for rank, document in enumerate(documents, start=1):
                meta = _meta(document)
                flag = "*" if _matches(document, case["expected_sources"]) else " "
                print(
                    f"  {flag} #{rank} {meta.get('source', _name(document))} "
                    f"[{meta.get('section', '-')} p.{meta.get('page', '-')}]"
                )
                snippet = (document.page_content or "").replace("\n", " ")[:120]
                print(f"      {snippet}…")

    return results


def summarize(
    results: List[CaseResult],
    top_k: int,
    embeddings_model: str,
    retrieval: Optional[str] = None,
) -> Dict[str, Any]:
    # 负样本（expected_sources 为空）用于「该拒答的能不能拒」，不参与检索指标；
    # 否则每加一条负样本都会把 Hit Rate / MRR 的上限往下拉。
    positives = [result for result in results if result.expected > 0]
    negatives = [result for result in results if result.expected <= 0]
    keyword_cases = [result for result in results if result.keywords_expected > 0]
    hit_cases = [result for result in positives if result.rank > 0]
    positive_count = len(positives)

    recall = (
        sum(min(result.matched / result.expected, 1.0) for result in positives)
        / positive_count
        if positive_count
        else 0.0
    )
    return {
        "top_k": top_k,
        "embeddings_model": embeddings_model,
        "retrieval": retrieval,
        "cases": len(results),
        "positives": positive_count,
        "negatives": len(negatives),
        "keyword_cases": len(keyword_cases),
        "hit_rate": len(hit_cases) / positive_count if positive_count else 0.0,
        "recall": recall,
        "mrr": (
            sum(1.0 / result.rank for result in hit_cases) / positive_count
            if positive_count
            else 0.0
        ),
        "keyword_hit": (
            sum(result.keyword_hit for result in keyword_cases) / len(keyword_cases)
            if keyword_cases
            else 0.0
        ),
        "precision": (
            sum(result.precision for result in positives) / positive_count
            if positive_count
            else 0.0
        ),
    }


def _print_report(results: List[CaseResult], metrics: Dict[str, Any]) -> None:
    print("\n逐条结果")
    print("-" * 72)
    for index, result in enumerate(results, start=1):
        if result.expected <= 0:
            status = "负样本"
        else:
            status = f"命中 @{result.rank}" if result.rank else "未命中"
        coverage = f"{result.matched}/{result.expected}" if result.expected else "-"
        keyword = (
            f"{result.keyword_hit:.0%}" if result.keywords_expected else "—"
        )
        print(f"{index:>2}. [{status}] 来源覆盖 {coverage} ｜ 关键词 {keyword}")
        print(f"    {result.question}")

    print("-" * 72)
    print(
        f"用例数        : {metrics['cases']}"
        f"（正样本 {metrics.get('positives', 0)} / 负样本 {metrics.get('negatives', 0)}）"
    )
    print(
        "说明          : 负样本只用于拒答评估，不参与 Hit Rate / Recall / MRR / 精确率"
    )
    print(f"Hit Rate      : {metrics['hit_rate']:.2%}   （top-{metrics['top_k']} 内命中期望文献的比例）")
    print(f"Recall@{metrics['top_k']}      : {metrics['recall']:.2%}   （期望文献被召回的完整度）")
    print(f"MRR           : {metrics['mrr']:.3f}   （首个命中名次的倒数均值，越接近 1 越好）")
    print(
        f"关键词命中率  : {metrics['keyword_hit']:.2%}   "
        f"（基于 {metrics.get('keyword_cases', 0)} 道标注了关键词的题）"
    )
    if "precision" in metrics:
        print(f"检索精确率    : {metrics['precision']:.2%}   （top-k 中来自期望文献的片段比例）")
    print(f"Embedding     : {metrics['embeddings_model']}")
    if metrics.get("retrieval"):
        print(f"检索配置      : {metrics['retrieval']}")

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 检索质量评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="评测数据集 JSON")
    parser.add_argument(
        "--embeddings", default=DEFAULT_EMBEDDING_MODEL, help="embedding 模型名"
    )
    parser.add_argument(
        "--top-k", type=int, default=max(DEFAULT_NUM_DOCUMENTS, 5), help="每次检索的片段数"
    )
    parser.add_argument("--dump", action="store_true", help="打印每个问题检索到的片段")
    parser.add_argument("--json", type=Path, default=None, help="把结果写成 JSON 报告")
    parser.add_argument(
        "--mode",
        default=None,
        choices=["dense", "hybrid"],
        help="检索模式：dense 纯向量 / hybrid BM25+向量+RRF（默认读 .env）",
    )
    parser.add_argument(
        "--rerank",
        default=None,
        choices=["off", "llm", "cross-encoder"],
        help="重排方式（默认读 .env）",
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="等价于 --rerank off"
    )
    args = parser.parse_args(argv)

    if args.no_rerank:
        args.rerank = "off"

    if not vector_store_available():
        print(f"[错误] 向量库不可用。{vector_store_hint()}")
        return 2

    try:
        cases = load_dataset(args.dataset)
    except (OSError, ValueError) as exc:
        print(f"[错误] {exc}")
        return 2

    knowledge = RAGKnowledgeBase(
        embeddings_model=args.embeddings,
        num_documents=args.top_k,
        retrieval_mode=args.mode,
        rerank_mode=args.rerank,
    )
    try:
        knowledge.ensure_embedder_ready()
    except Exception as exc:  # noqa: BLE001 - 直接把原因反馈给使用者
        print(f"[错误] {exc}")
        return 2

    total_documents = knowledge.count()
    if total_documents == 0:
        print(
            "[错误] 向量库里还没有文献，请先在界面里上传 PDF / 导入 arXiv 论文。"
        )
        return 2
    print(f"知识库片段数：{total_documents} ｜ 用例数：{len(cases)} ｜ top-k：{args.top_k}")

    results = evaluate_cases(cases, knowledge, args.top_k, dump=args.dump)
    retrieval_desc = (
        f"{knowledge.retrieval_mode}"
        f" + rerank:{'off' if knowledge.reranker is None else type(knowledge.reranker).__name__}"
    )
    metrics = summarize(results, args.top_k, args.embeddings, retrieval=retrieval_desc)
    _print_report(results, metrics)

    if args.json:
        payload = {
            "metrics": metrics,
            "cases": [
                {
                    "question": result.question,
                    "rank": result.rank,
                    "matched": result.matched,
                    "expected": result.expected,
                    "keyword_hit": result.keyword_hit,
                    "retrieved": result.retrieved,
                }
                for result in results
            ],
        }
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n报告已写入 {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
