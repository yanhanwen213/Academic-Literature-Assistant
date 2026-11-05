"""实验矩阵：一键跑出「基线 vs 增强」的对比结果，直接产出论文用的表格。

用法（项目根目录执行）：

    # 1) 最常用：库里已有文献，直接跑三组对比
    python -m scripts.run_experiments --dataset rag/eval_questions.example.json

    # 2) 先用 arXiv ID 导入几篇论文，再跑
    python -m scripts.run_experiments --arxiv 1512.03385 2016.xxxxx --dataset my.json

    # 3) 只用本地 PDF
    python -m scripts.run_experiments --pdf papers/a.pdf --pdf papers/b.pdf --dataset my.json

    # 4) 同时扫描拒答阈值（产出「拒答率 vs 误拒率」曲线数据）
    python -m scripts.run_experiments --dataset my.json --refusal-sweep

输出的对比表可以直接贴进论文，JSON 用于画图。
"""

import argparse
import json
import sys
from pathlib import Path

# 允许 `python scripts/run_experiments.py` 直接运行
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Any, Dict, List, Optional, Sequence  # noqa: E402

from rag.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_NUM_DOCUMENTS  # noqa: E402
from rag.evaluation import (  # noqa: E402
    DEFAULT_DATASET,
    evaluate_cases,
    load_dataset,
    summarize,
)
from rag.knowledge_base import RAGKnowledgeBase  # noqa: E402
from rag.rerank import CrossEncoderReranker  # noqa: E402
from rag.vector_store import vector_store_available, vector_store_hint  # noqa: E402

# 默认消融组合：(名称, 检索模式, 重排方式)
DEFAULT_CONFIGS: Sequence[tuple] = (
    ("基线：纯向量", "dense", "off"),
    ("+ 混合检索（BM25+RRF）", "hybrid", "off"),
    ("+ 重排（完整方案）", "hybrid", "cross-encoder"),
)


def _cross_encoder_available() -> bool:
    """cross-encoder 是否真的可用（装了依赖且模型已缓存）。"""
    try:
        probe = CrossEncoderReranker("BAAI/bge-reranker-base", None)
    except Exception:  # noqa: BLE001
        return False
    return probe.available


def _println(text: str = "") -> None:
    print(text)


def ingest(
    knowledge: RAGKnowledgeBase,
    arxiv_ids: Sequence[str],
    pdf_paths: Sequence[str],
) -> None:
    """把待评测的论文导入知识库（重复导入会按 id 覆盖，不会重复计数）。"""
    for arxiv_id in arxiv_ids:
        _println(f"[导入] arXiv:{arxiv_id} …")
        try:
            result = knowledge.add_arxiv(arxiv_id)
        except Exception as exc:  # noqa: BLE001
            _println(f"  [失败] {exc}")
            continue
        _println(f"  已入库 {result.added} 个片段：{result.title or result.source}")

    for pdf_path in pdf_paths:
        path = Path(pdf_path)
        _println(f"[导入] {path.name} …")
        try:
            result = knowledge.add_pdf_bytes(path.read_bytes(), filename=path.name)
        except Exception as exc:  # noqa: BLE001
            _println(f"  [失败] {exc}")
            continue
        _println(f"  已入库 {result.added} 个片段：{result.title or result.source}")


def _build_llm_safe():
    """尝试构造 LLM 用于 LLM 重排；没有 key 时返回 None（不阻断实验）。"""
    try:
        from agent.llm import build_llm
        from agent.config import DEFAULT_LLM_MODEL

        return build_llm(DEFAULT_LLM_MODEL)
    except Exception as exc:  # noqa: BLE001
        _println(f"  [提示] LLM 不可用（{exc}），LLM 重排将退化为不重排。")
        return None


def run_matrix(
    cases: List[Dict[str, Any]],
    embeddings_model: str,
    top_k: int,
    configs: Sequence[tuple],
    dump: bool,
) -> List[Dict[str, Any]]:
    """依次跑每个配置，返回每个配置的完整指标。"""
    rows: List[Dict[str, Any]] = []
    llm = None  # 懒加载：只有配置里真的用到 llm 重排才构造

    for name, mode, rerank in configs:
        _println(f"\n{'=' * 72}\n[配置] {name}（mode={mode}, rerank={rerank}）")

        if rerank == "llm" and llm is None:
            llm = _build_llm_safe()

        knowledge = RAGKnowledgeBase(
            embeddings_model=embeddings_model,
            num_documents=top_k,
            llm=llm,
            retrieval_mode=mode,
            rerank_mode=rerank,
        )

        # cross-encoder 不可用时明确告警，避免「以为开了重排其实没开」
        if rerank == "cross-encoder" and knowledge.reranker is None:
            _println(
                "  [警告] cross-encoder 不可用（未安装 sentence-transformers "
                "或模型未缓存），本组实际为不重排。可改用 --configs 指定 llm。"
            )
        if rerank == "llm" and knowledge.reranker is None:
            _println("  [警告] LLM 重排不可用，本组实际为不重排。")

        import time as _time

        started = _time.perf_counter()
        results = evaluate_cases(cases, knowledge, top_k, dump=dump)
        elapsed = _time.perf_counter() - started
        latency_ms = (elapsed / max(len(cases), 1)) * 1000
        retrieval = f"{mode}+{'off' if knowledge.reranker is None else type(knowledge.reranker).__name__}"
        metrics = summarize(results, top_k, embeddings_model, retrieval=retrieval)
        metrics["config"] = name
        metrics["mode"] = mode
        metrics["rerank"] = rerank
        metrics["latency_ms"] = round(latency_ms, 1)
        rows.append(metrics)

        _println(
            f"  HitRate={metrics['hit_rate']:.2%}  "
            f"Recall@{top_k}={metrics['recall']:.2%}  "
            f"MRR={metrics['mrr']:.3f}  "
            f"精确率={metrics.get('precision', 0.0):.2%}  "
            f"关键词={metrics['keyword_hit']:.2%}  "
            f"平均时延={latency_ms:.1f}ms"
        )
    return rows


def refusal_sweep(
    cases: List[Dict[str, Any]],
    embeddings_model: str,
    top_k: int,
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    """扫描拒答阈值：观察「该答的能答 / 不该答的能拒」。

    说明：这里用评测集里的 expected_sources 作为「是否有标准答案」的标签。
    若有 expected_sources，视为「应该作答」；没有则视为「应该拒答」的负样本。
    """
    from rag.refusal import assess_confidence

    knowledge = RAGKnowledgeBase(
        embeddings_model=embeddings_model,
        num_documents=top_k,
        retrieval_mode="hybrid",
        rerank_mode="off",
    )

    samples: List[tuple] = []
    for case in cases:
        pairs = knowledge.search_with_scores(case["question"], num_documents=top_k)
        documents = [document for document, _ in pairs]
        confidence = assess_confidence(case["question"], documents, scored=pairs or None)
        answerable = bool(case.get("expected_sources"))
        samples.append((confidence.score, answerable))

    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        answered = [score >= threshold for score, _ in samples]
        tp = sum(1 for flag, (_, ans) in zip(answered, samples) if flag and ans)
        fn = sum(1 for flag, (_, ans) in zip(answered, samples) if not flag and ans)
        fp = sum(1 for flag, (_, ans) in zip(answered, samples) if flag and not ans)
        tn = sum(1 for flag, (_, ans) in zip(answered, samples) if not flag and not ans)

        answerable_total = sum(1 for _, ans in samples if ans) or 1
        unanswerable_total = sum(1 for _, ans in samples if not ans) or 1
        rows.append(
            {
                "threshold": threshold,
                "answer_rate": (tp + fp) / len(samples),
                "coverage": tp / answerable_total,          # 该答的答了多少
                "false_refusal": fn / answerable_total,     # 误拒（该答却拒了）
                "correct_refusal": tn / unanswerable_total,  # 该拒的拒了多少
                "false_answer": fp / unanswerable_total,    # 误答（不该答却答了）
            }
        )
    return rows


def print_markdown_matrix(rows: List[Dict[str, Any]]) -> str:
    """把消融结果渲染成 Markdown 表（直接贴论文）。"""
    lines = [
        "| 方案 | Hit Rate | Recall@k | MRR | 检索精确率 | 关键词命中率 | 平均时延(ms) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['hit_rate']:.2%} | {row['recall']:.2%} "
            f"| {row['mrr']:.3f} | {row.get('precision', 0.0):.2%} "
            f"| {row['keyword_hit']:.2%} | {row.get('latency_ms', 0.0):.1f} |"
        )
    return "\n".join(lines)


def print_markdown_refusal(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| 阈值 | 作答率 | 该答的作答率 | 误拒率 | 该拒的拒答率 | 误答率 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']:.2f} | {row['answer_rate']:.2%} "
            f"| {row['coverage']:.2%} | {row['false_refusal']:.2%} "
            f"| {row['correct_refusal']:.2%} | {row['false_answer']:.2%} |"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 消融实验矩阵")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=max(DEFAULT_NUM_DOCUMENTS, 5))
    parser.add_argument("--arxiv", nargs="*", default=[], help="先导入的 arXiv ID 列表")
    parser.add_argument("--pdf", nargs="*", default=[], help="先导入的本地 PDF 路径")
    parser.add_argument(
        "--configs",
        default=None,
        help="自定义配置，形如 'name:mode:rerank,name2:mode2:rerank2'",
    )
    parser.add_argument("--dump", action="store_true", help="打印每问检索到的片段")
    parser.add_argument("--refusal-sweep", action="store_true", help="附带拒答阈值扫描")
    parser.add_argument(
        "--out-dir", type=Path, default=_ROOT / "data" / "experiments",
        help="结果输出目录",
    )
    args = parser.parse_args(argv)

    if not vector_store_available():
        _println(f"[错误] 向量库不可用。{vector_store_hint()}")
        return 2

    try:
        cases = load_dataset(args.dataset)
    except (OSError, ValueError) as exc:
        _println(f"[错误] {exc}")
        return 2

    knowledge = RAGKnowledgeBase(
        embeddings_model=args.embeddings, num_documents=args.top_k
    )
    try:
        knowledge.ensure_embedder_ready()
    except Exception as exc:  # noqa: BLE001
        _println(f"[错误] {exc}")
        return 2

    if args.arxiv or args.pdf:
        ingest(knowledge, args.arxiv, args.pdf)

    total = knowledge.count()
    if total == 0:
        _println("[错误] 知识库里还没有文献，请先 --arxiv/--pdf 导入，或在界面里上传。")
        return 2
    _println(f"\n知识库片段数：{total} ｜ 用例数：{len(cases)} ｜ top-k：{args.top_k}")

    # 解析配置
    if args.configs:
        configs: List[tuple] = []
        for item in args.configs.split(","):
            parts = item.split(":")
            if len(parts) != 3:
                _println(f"[错误] --configs 格式应为 name:mode:rerank，收到：{item}")
                return 2
            configs.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    else:
        configs = list(DEFAULT_CONFIGS)
        if not _cross_encoder_available():
            _println(
                "[提示] cross-encoder 不可用，第三组将改为 LLM 重排"
                "（或用 --no-cross-encoder 手动指定）。"
            )
            configs[-1] = ("+ 重排（完整方案）", "hybrid", "llm")

    rows = run_matrix(cases, args.embeddings, args.top_k, configs, args.dump)

    table = print_markdown_matrix(rows)
    _println("\n" + "=" * 72)
    _println("消融实验结果（Markdown，可直接贴论文）")
    _println("=" * 72)
    _println(table)

    payload: Dict[str, Any] = {
        "embeddings": args.embeddings,
        "top_k": args.top_k,
        "cases": len(cases),
        "library_chunks": total,
        "matrix": rows,
    }

    if args.refusal_sweep:
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
        sweep = refusal_sweep(cases, args.embeddings, args.top_k, thresholds)
        payload["refusal_sweep"] = sweep
        refusal_table = print_markdown_refusal(sweep)
        _println("\n" + "=" * 72)
        _println("拒答阈值扫描（Markdown）")
        _println("=" * 72)
        _println(refusal_table)
        table = table + "\n\n" + refusal_table

    args.out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = args.out_dir / "experiment_report.md"
    json_path = args.out_dir / "experiment_report.json"
    markdown_path.write_text(table, encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _println(f"\n报告已写入：\n  {markdown_path}\n  {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
