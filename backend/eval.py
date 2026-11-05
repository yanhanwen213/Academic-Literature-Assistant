"""评测中心的数据层：数据集管理、跑评测、结果落库、导出。

界面只做渲染，真正的评测逻辑全部复用 `rag/evaluation.py`，避免界面上
和命令行里跑出两套口径。评测结果写进平台库 `eval_runs`，刷新页面、
重启进程都还在，答辩时可以直接拿出来对比。
"""

import csv
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from backend import db
from backend.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

EVAL_DIR = PROJECT_ROOT / "data" / "eval_datasets"
BUILTIN_DIR = PROJECT_ROOT / "rag"
MAX_CASES = 50
MODES = ("hybrid", "dense")
RERANKS = ("off", "cross-encoder", "llm")
DEFAULT_MATRIX = (
    ("纯向量检索", "dense", "off"),
    ("混合检索（BM25+向量+RRF）", "hybrid", "off"),
    ("混合检索 + 重排", "hybrid", "cross-encoder"),
)

_NAME_RE = re.compile(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+")


# ---------------------------------------------------------------- 数据集
def _dataset_paths() -> List[Path]:
    paths = sorted(BUILTIN_DIR.glob("eval_questions*.json"))
    if EVAL_DIR.exists():
        paths.extend(sorted(EVAL_DIR.glob("*.json")))
    return [path for path in paths if path.is_file()]


def load_cases(path: str) -> List[Dict[str, Any]]:
    from rag.evaluation import load_dataset

    cases = load_dataset(Path(path))
    if not cases:
        raise ValueError("数据集里没有可用的题目")
    return cases


def _dataset_meta(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    meta = payload.get("meta") if isinstance(payload, dict) else None
    return meta if isinstance(meta, dict) else {}


def list_datasets() -> List[Dict[str, object]]:
    """列出内置与上传的评测集，解析失败的也列出来并标注原因。"""
    rows: List[Dict[str, object]] = []
    for path in _dataset_paths():
        builtin = path.parent.resolve() == BUILTIN_DIR.resolve()
        meta = _dataset_meta(path)
        generated = bool(meta.get("created_by") or meta.get("mode"))
        row: Dict[str, object] = {
            "名称": path.stem,
            "文件": path.name,
            "来源": "内置" if builtin else ("生成" if generated else "上传"),
            "出题方式": str(meta.get("mode") or ""),
            "题数": 0,
            "大小(KB)": round(path.stat().st_size / 1024, 1),
            "更新时间": time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)
            ),
            "可用": True,
            "问题": "",
            "path": str(path),
        }
        try:
            row["题数"] = len(load_cases(str(path)))
        except Exception as exc:  # noqa: BLE001 - 坏文件也要能看见
            row["可用"] = False
            row["问题"] = str(exc)[:120]
        rows.append(row)
    return rows


def get_dataset(path: str) -> Dict[str, object]:
    for row in list_datasets():
        if Path(str(row["path"])).resolve() == Path(path).resolve():
            return row
    raise ValueError("数据集不存在或不在允许的目录里")


def save_uploaded_dataset(filename: str, data: bytes) -> Dict[str, object]:
    """校验并保存上传的 JSON 数据集，返回保存后的元信息。"""
    if not data:
        raise ValueError("文件内容为空")
    try:
        payload = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"不是合法的 JSON：{exc}") from exc

    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("JSON 里需要非空的 cases 列表")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(filename or "dataset").stem
    cleaned = _NAME_RE.sub("-", stem).strip("-._") or "dataset"
    target = EVAL_DIR / f"{cleaned[:60]}.json"
    target.write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 写入后再用同一套解析器校验，保证列表里显示的一定能跑
    total = len(load_cases(str(target)))
    return {"path": str(target), "name": target.stem, "cases": total}


def template_dataset() -> bytes:
    example = BUILTIN_DIR / "eval_questions.example.json"
    if example.exists():
        return example.read_bytes()
    payload = {
        "cases": [
            {
                "question": "这篇论文的核心贡献是什么？",
                "expected_sources": ["arxiv:2401.00001"],
                "expected_keywords": ["retrieval", "embedding"],
            }
        ]
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


# ---------------------------------------------------------------- 评测执行
def reproduce_command(
    dataset: str,
    mode: str,
    rerank: str,
    top_k: int,
    library: str = "",
) -> str:
    """命令行复现命令；CLI 只作用于默认知识库，其它库需在界面里跑。"""
    parts = [
        "python -m rag.evaluation",
        f'--dataset "{Path(dataset).as_posix()}"',
        f"--mode {mode}",
        f"--rerank {rerank}",
        f"--top-k {int(top_k)}",
    ]
    return " ".join(parts)


def _build_knowledge(
    library: str,
    embeddings_model: str,
    top_k: int,
    mode: str,
    rerank: str,
):
    """构造知识库；单独抽出来是为了测试时可以替换成假实现。"""
    from rag import libraries as libs
    from rag.knowledge_base import RAGKnowledgeBase

    llm = None
    if rerank == "llm":
        from agent.config import DEFAULT_LLM_MODEL
        from agent.llm import build_llm

        llm = build_llm(DEFAULT_LLM_MODEL)

    target = libs.get_library(library)
    return RAGKnowledgeBase(
        embeddings_model=embeddings_model,
        num_documents=top_k,
        llm=llm,
        retrieval_mode=mode,
        rerank_mode=rerank,
        uri=target.lancedb,
    )


def _clean_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {
        key: round(float(metrics.get(key, 0.0) or 0.0), 4)
        for key in ("hit_rate", "recall", "mrr", "keyword_hit", "precision")
    }
    for key in ("positives", "negatives", "keyword_cases"):
        clean[key] = int(metrics.get(key, 0) or 0)
    return clean


def _case_rows(results: Sequence[Any], cases: Sequence[Dict[str, Any]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, (result, case) in enumerate(zip(results, cases), start=1):
        expected = case.get("expected_sources") or []
        rows.append(
            {
                "序号": index,
                "问题": result.question,
                "期望来源": " ｜ ".join(expected[:3]) if expected else "（负样本）",
                "命中名次": result.rank or "未命中",
                "来源覆盖": f"{result.matched}/{result.expected}"
                if result.expected
                else "—",
                "关键词覆盖": round(float(result.keyword_hit) * 100, 1)
                if result.keywords_expected
                else "—",
                "精确率": round(float(result.precision) * 100, 1)
                if result.expected
                else "—",
                "检索片段": " ｜ ".join(result.retrieved[:5]),
            }
        )
    return rows


def run(
    username: str,
    dataset: str,
    library: str,
    mode: str = "hybrid",
    rerank: str = "off",
    top_k: int = 5,
    limit: Optional[int] = None,
    embeddings_model: Optional[str] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """跑一次检索评测并落库，返回 {id, config, metrics, case_rows, ...}。"""
    from rag.config import DEFAULT_EMBEDDING_MODEL
    from rag.evaluation import evaluate_cases, summarize

    if mode not in MODES:
        raise ValueError(f"不支持的检索模式：{mode}")
    if rerank not in RERANKS:
        raise ValueError(f"不支持的重排方式：{rerank}")

    meta = get_dataset(dataset)
    if not meta["可用"]:
        raise ValueError(f"数据集不可用：{meta['问题']}")

    cases = load_cases(dataset)
    if limit:
        cases = cases[: max(1, min(int(limit), MAX_CASES))]
    if not cases:
        raise ValueError("题数为 0")

    model = embeddings_model or DEFAULT_EMBEDDING_MODEL
    knowledge = _build_knowledge(library, model, int(top_k), mode, rerank)
    try:
        knowledge.ensure_embedder_ready()
    except Exception as exc:  # noqa: BLE001 - 把原因原样反馈给界面
        raise RuntimeError(str(exc)) from exc

    chunks = knowledge.count()
    if chunks == 0:
        raise ValueError(f"知识库「{library}」还没有文献，请先上传 PDF 或导入 arXiv")

    started = time.perf_counter()
    results = evaluate_cases(cases, knowledge, int(top_k), on_progress=on_progress)
    duration_ms = int((time.perf_counter() - started) * 1000)

    actual_rerank = "off"
    if knowledge.reranker is not None:
        actual_rerank = type(knowledge.reranker).__name__
    metrics = summarize(
        results,
        int(top_k),
        model,
        retrieval=f"{knowledge.retrieval_mode} + rerank:{actual_rerank}",
    )
    clean = _clean_metrics(metrics)
    cases_payload = [
        {
            "question": result.question,
            "expected_sources": case.get("expected_sources") or [],
            "expected_keywords": case.get("expected_keywords") or [],
            "keywords_expected": result.keywords_expected,
            "rank": result.rank,
            "matched": result.matched,
            "expected": result.expected,
            "keyword_hit": round(float(result.keyword_hit), 4),
            "precision": round(float(result.precision), 4),
            "retrieved": result.retrieved,
        }
        for result, case in zip(results, cases)
    ]
    config = {
        "dataset": Path(dataset).name,
        "dataset_source": meta.get("来源", ""),
        "generation_mode": meta.get("出题方式", ""),
        "library": library,
        "mode": mode,
        "rerank": rerank,
        "actual_rerank": actual_rerank,
        "top_k": int(top_k),
        "cases": len(cases),
        "chunks": chunks,
        "embeddings": model,
        "duration_ms": duration_ms,
        "username": username,
    }

    run_id = _save_run(config, clean, cases_payload) if save else None
    return {
        "id": run_id,
        "config": config,
        "metrics": clean,
        "case_rows": _case_rows(results, cases),
        "cases": cases_payload,
        "reproduce": reproduce_command(dataset, mode, rerank, int(top_k), library),
    }


def run_matrix(
    username: str,
    dataset: str,
    library: str,
    top_k: int = 5,
    limit: Optional[int] = None,
    configs: Sequence[tuple] = DEFAULT_MATRIX,
    on_progress: Optional[Callable[[str, int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """依次跑多组配置（纯向量 / 混合 / 重排），每组各落一条记录。"""

    def _forward(label: str) -> Callable[[int, int, str], None]:
        def _callback(done: int, total: int, question: str) -> None:
            if on_progress is not None:
                on_progress(label, done, total, question)

        return _callback

    rows: List[Dict[str, Any]] = []
    for name, mode, rerank in configs:
        rows.append(
            run(
                username,
                dataset,
                library,
                mode=mode,
                rerank=rerank,
                top_k=top_k,
                limit=limit,
                on_progress=_forward(name),
            )
        )
    return rows


# ---------------------------------------------------------------- 数据集体检
_GENERIC_TITLES = {"sensors", "ieee", "arxiv", "article", "paper", "research"}
_KEYWORD_STOP = {
    "this", "that", "with", "from", "using", "based", "into", "their", "there",
    "these", "those", "have", "been", "were", "will", "would", "which", "what",
    "when", "where", "while", "about", "through", "between", "under", "over",
    "paper", "study", "approach", "method", "results", "result", "propose",
    "proposed", "using", "also", "more", "than", "such", "can", "may", "our",
    "abstract", "introduction", "conclusion", "references", "appendix",
    "related", "background", "discussion", "experiment", "experiments",
}

_NEGATIVE_QUESTIONS = [
    "量子计算机如何实现容错量子门？",
    "全球变暖对海洋环流有哪些影响？",
    "货币政策如何影响通货膨胀？",
    "文艺复兴时期的绘画技法有哪些变化？",
    "蛋白质折叠预测的主要难点是什么？",
    "如何用强化学习训练机械臂抓取物体？",
]


def _doc_meta(document) -> Dict[str, Any]:
    return dict(getattr(document, "metadata", None) or {})


def _source_key(document) -> str:
    meta = _doc_meta(document)
    return str(
        meta.get("source_key")
        or meta.get("source")
        or meta.get("name")
        or ""
    ).strip()


def _known_sources(library: str) -> List[Dict[str, str]]:
    """知识库里每篇文献一个候选（source_key / source / title / name）。"""
    from rag.config import DEFAULT_EMBEDDING_MODEL

    knowledge = _build_knowledge(
        library, DEFAULT_EMBEDDING_MODEL, 5, "dense", "off"
    )
    documents = knowledge.vector_store.all_documents()
    seen = set()
    rows: List[Dict[str, str]] = []
    for document in documents:
        meta = _doc_meta(document)
        key = str(meta.get("source_key") or "")
        name = str(meta.get("name") or "")
        identity = key or name
        if not identity or identity in seen:
            continue
        seen.add(identity)
        rows.append(
            {
                "source_key": key or name,
                "source": str(meta.get("source") or ""),
                "title": str(meta.get("title") or ""),
            }
        )
    return rows


def check_dataset(dataset: str, library: str) -> Dict[str, Any]:
    """体检：数据集里的 expected_sources 能不能在目标知识库里找到。"""
    meta = get_dataset(dataset)
    if not meta["可用"]:
        raise ValueError(f"数据集不可用：{meta['问题']}")
    cases = load_cases(dataset)
    known = _known_sources(library)
    haystacks = [
        " ".join(
            [row["source_key"], row["source"], row["title"]]
        ).lower()
        for row in known
    ]
    rows: List[Dict[str, object]] = []
    missing = 0
    positives = 0
    for index, case in enumerate(cases, start=1):
        expected = case.get("expected_sources") or []
        if not expected:
            rows.append(
                {
                    "序号": index,
                    "问题": case["question"],
                    "期望来源": "（负样本）",
                    "状态": "跳过（不参与检索指标）",
                }
            )
            continue
        positives += 1
        unmatched = [
            item
            for item in expected
            if not any(item.strip().lower() in haystack for haystack in haystacks)
        ]
        if unmatched:
            missing += 1
        rows.append(
            {
                "序号": index,
                "问题": case["question"],
                "期望来源": " ｜ ".join(expected[:3]),
                "状态": "✅ 匹配" if not unmatched else f"❌ 未匹配：{'、'.join(unmatched[:3])}",
            }
        )
    return {
        "rows": rows,
        "known": known[:80],
        "positive": positives,
        "missing": missing,
        "chunks": len(haystacks),
    }


# ---------------------------------------------------------------- 生成数据集
def _doc_label(meta: Dict[str, Any]) -> str:
    title = str(meta.get("title") or "").strip()
    source = str(meta.get("source") or "").strip()
    name = str(meta.get("name") or "").strip()
    if len(title) >= 8 and title.lower() not in _GENERIC_TITLES:
        half = len(title) // 2
        repeated = title[:half].strip() and title[:half].strip() == title[half:].strip()
        if not repeated:
            return title
    return source or title or name or "该文献"


def _title_keywords(meta: Dict[str, Any]) -> List[str]:
    text = f"{meta.get('title') or ''} {meta.get('section') or ''}"
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text)
    picked: List[str] = []
    for word in words:
        lowered = word.lower()
        if lowered in _KEYWORD_STOP or lowered in picked:
            continue
        picked.append(lowered)
        if len(picked) >= 5:
            break
    return picked


def _template_question(meta: Dict[str, Any]) -> str:
    label = _doc_label(meta)
    section = str(meta.get("section") or "").strip()
    if section and section.lower() != "abstract":
        return f"《{label}》的「{section}」部分主要讨论了什么？"
    return f"《{label}》这篇论文解决了什么问题、用了什么方法？"


def _sample_documents(library: str, count: int) -> List[Any]:
    """按文献去重抽样；优先取 Abstract / Introduction 片段。"""
    from rag.config import DEFAULT_EMBEDDING_MODEL

    knowledge = _build_knowledge(
        library, DEFAULT_EMBEDDING_MODEL, 5, "dense", "off"
    )
    documents = knowledge.vector_store.all_documents()
    if not documents:
        raise ValueError(f"知识库「{library}」还没有文献")

    best: Dict[str, tuple] = {}
    for document in documents:
        meta = _doc_meta(document)
        key = _source_key(document)
        if not key:
            continue
        section = str(meta.get("section") or "").lower()
        score = 0 if section in {"abstract", "introduction"} else 1
        current = best.get(key)
        if current is None or score < current[0]:
            best[key] = (score, document)

    picked = [document for _key, (_score, document) in sorted(best.items())][:count]
    if len(picked) < count:
        used = {id(document) for document in picked}
        for document in documents:
            if id(document) in used:
                continue
            picked.append(document)
            if len(picked) >= count:
                break
    return picked


def _llm_question(llm, document) -> Dict[str, Any]:
    text = (getattr(document, "page_content", "") or "")[:1800]
    prompt = (
        "你是学术文献评测集的作者。请根据下面的片段出一道**只能靠该片段回答**的中文问题，"
        "并给出 3~5 个能在片段里找到的关键词（英文或中文原词）。\n"
        '只输出 JSON，不要解释：{"question": "...", "keywords": ["...", "..."]}\n\n'
        f"片段：\n{text}"
    )
    response = llm.invoke(prompt)
    raw = getattr(response, "content", response)
    text_out = str(raw).strip()
    match = re.search(r"\{.*\}", text_out, flags=re.S)
    if not match:
        raise ValueError("模型没有返回 JSON")
    payload = json.loads(match.group(0))
    question = str(payload.get("question") or "").strip()
    if not question:
        raise ValueError("模型返回的问题为空")
    keywords = [
        str(item).strip()
        for item in (payload.get("keywords") or [])
        if str(item).strip()
    ]
    return {"question": question, "keywords": keywords[:6]}


def generate_dataset(
    username: str,
    library: str,
    count: int = 10,
    mode: str = "template",
    negatives: int = 2,
    name: str = "",
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """从当前知识库抽样生成评测集 JSON（模板出题或 LLM 出题）。"""
    if mode not in {"template", "llm"}:
        raise ValueError(f"不支持的生成方式：{mode}")
    total = max(1, min(int(count), 30))
    negative_count = max(0, min(int(negatives), 10))

    documents = _sample_documents(library, total)
    llm = None
    if mode == "llm":
        from agent.config import DEFAULT_LLM_MODEL
        from agent.llm import build_llm

        llm = build_llm(DEFAULT_LLM_MODEL)

    cases: List[Dict[str, Any]] = []
    fallbacks = 0
    for index, document in enumerate(documents, start=1):
        meta = _doc_meta(document)
        expected_source = _source_key(document)
        if not expected_source:
            continue
        question = _template_question(meta)
        keywords = _title_keywords(meta)
        if llm is not None:
            try:
                generated = _llm_question(llm, document)
                question = generated["question"]
                keywords = generated["keywords"] or keywords
            except Exception as exc:  # noqa: BLE001 - 单题失败退回模板
                logger.warning("LLM 出题失败，改用模板：%s", exc)
                fallbacks += 1
        cases.append(
            {
                "question": question,
                "expected_sources": [expected_source],
                "expected_keywords": keywords,
                "meta": {
                    "source": str(meta.get("source") or ""),
                    "section": str(meta.get("section") or ""),
                    "page": str(meta.get("page") or ""),
                },
            }
        )
        if on_progress is not None:
            try:
                on_progress(index, len(documents), question)
            except Exception:  # noqa: BLE001
                pass

    if not cases:
        raise ValueError("没有抽到可用的文献片段")

    for question in _NEGATIVE_QUESTIONS[:negative_count]:
        cases.append(
            {
                "question": question,
                "expected_sources": [],
                "expected_keywords": [],
                "meta": {"kind": "negative"},
            }
        )

    default_name = f"generated-{library}-{time.strftime('%Y%m%d-%H%M%S')}"
    saved = save_generated_dataset(
        name.strip() or default_name,
        cases,
        {
            "library": library,
            "mode": mode,
            "created_by": username,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "positives": len(cases) - negative_count,
            "negatives": negative_count,
            "fallbacks": fallbacks,
        },
    )
    saved.update(
        {
            "positives": len(cases) - negative_count,
            "negatives": negative_count,
            "fallbacks": fallbacks,
            "preview": cases[:3],
        }
    )
    return saved


def save_generated_dataset(
    name: str, cases: List[Dict[str, Any]], meta: Dict[str, Any]
) -> Dict[str, Any]:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stem = _NAME_RE.sub("-", Path(name or "generated").stem).strip("-._") or "generated"
    target = _unique_path(EVAL_DIR / f"{stem[:60]}.json")
    payload = {"meta": meta, "cases": cases}
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        total = len(load_cases(str(target)))
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return {"path": str(target), "name": target.stem, "cases": total}


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 100):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError("同名文件太多，请换个名称")


def dataset_bytes(path: str) -> bytes:
    """下载当前数据集原始 JSON（只允许内置目录与上传目录）。"""
    meta = get_dataset(path)
    return Path(str(meta["path"])).read_bytes()


# ---------------------------------------------------------------- 结果落库
def _save_run(
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    cases: List[Dict[str, Any]],
) -> int:
    db.init_db()
    with db.connect() as connection:
        cursor = connection.execute(
            "INSERT INTO eval_runs "
            "(ts, username, dataset, library, mode, rerank, top_k, cases,"
            " hit_rate, recall, mrr, keyword_hit, precision, duration_ms,"
            " metrics_json, detail_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                config.get("username", ""),
                config.get("dataset", ""),
                config.get("library", ""),
                config.get("mode", ""),
                config.get("actual_rerank", config.get("rerank", "")),
                int(config.get("top_k", 0) or 0),
                int(config.get("cases", 0) or 0),
                float(metrics.get("hit_rate", 0.0)),
                float(metrics.get("recall", 0.0)),
                float(metrics.get("mrr", 0.0)),
                float(metrics.get("keyword_hit", 0.0)),
                float(metrics.get("precision", 0.0)),
                int(config.get("duration_ms", 0) or 0),
                json.dumps({"config": config, "metrics": metrics}, ensure_ascii=False),
                json.dumps(cases, ensure_ascii=False),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid or 0)


def _row_to_summary(row) -> Dict[str, object]:
    try:
        payload = json.loads(row["metrics_json"] or "{}")
    except ValueError:
        payload = {}
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    return {
        "ID": row["id"],
        "时间": row["ts"],
        "操作人": row["username"],
        "数据集": row["dataset"],
        "数据集来源": config.get("dataset_source") or "—",
        "出题方式": {"template": "模板", "llm": "LLM"}.get(
            str(config.get("generation_mode") or ""), "—"
        ),
        "知识库": row["library"],
        "模式": row["mode"],
        "重排": row["rerank"],
        "top-k": row["top_k"],
        "题数": row["cases"],
        "Hit Rate": round(float(row["hit_rate"]) * 100, 1),
        "Recall": round(float(row["recall"]) * 100, 1),
        "MRR": round(float(row["mrr"]), 3),
        "关键词": round(float(row["keyword_hit"]) * 100, 1),
        "精确率": round(float(row["precision"]) * 100, 1),
        "耗时(s)": round(int(row["duration_ms"]) / 1000, 1),
    }


def list_runs(limit: int = 50) -> List[Dict[str, object]]:
    db.init_db()
    with db.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM eval_runs ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [_row_to_summary(row) for row in rows]


def get_run(run_id: int) -> Dict[str, Any]:
    db.init_db()
    with db.connect() as connection:
        row = connection.execute(
            "SELECT * FROM eval_runs WHERE id = ?", (int(run_id),)
        ).fetchone()
    if row is None:
        raise ValueError(f"评测记录不存在：{run_id}")
    try:
        payload = json.loads(row["metrics_json"] or "{}")
    except ValueError:
        payload = {}
    try:
        cases = json.loads(row["detail_json"] or "[]")
    except ValueError:
        cases = []
    return {
        "summary": _row_to_summary(row),
        "config": payload.get("config", {}),
        "metrics": payload.get("metrics", {}),
        "cases": cases,
    }


def delete_runs(run_ids: Sequence[int]) -> int:
    """删除指定的评测记录，返回实际删除条数。"""
    ids = sorted({int(item) for item in run_ids})
    if not ids:
        return 0
    db.init_db()
    placeholders = ",".join("?" for _ in ids)
    with db.connect() as connection:
        cursor = connection.execute(
            f"DELETE FROM eval_runs WHERE id IN ({placeholders})", ids
        )
        connection.commit()
        return int(cursor.rowcount or 0)


def clear_runs() -> int:
    """清空全部评测历史，返回删除条数。"""
    db.init_db()
    with db.connect() as connection:
        cursor = connection.execute("DELETE FROM eval_runs")
        connection.commit()
        return int(cursor.rowcount or 0)


def runs_csv(rows: List[Dict[str, object]]) -> bytes:
    buffer = io.StringIO()
    fields = [
        "ID", "时间", "操作人", "数据集", "数据集来源", "出题方式", "知识库",
        "模式", "重排", "top-k", "题数", "Hit Rate", "Recall", "MRR", "关键词",
        "精确率", "耗时(s)",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fields})
    return buffer.getvalue().encode("utf-8-sig")


def cases_csv(cases: List[Dict[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["问题", "期望来源", "命中名次", "命中来源数", "期望来源数", "关键词覆盖", "精确率", "检索片段"]
    )
    for case in cases:
        keyword = (
            f"{float(case.get('keyword_hit', 0)) * 100:.1f}%"
            if int(case.get("keywords_expected", 0) or 0)
            else "—"
        )
        writer.writerow(
            [
                case.get("question", ""),
                " ｜ ".join(case.get("expected_sources") or []) or "（负样本）",
                case.get("rank") or "未命中",
                case.get("matched", 0),
                case.get("expected", 0),
                keyword,
                f"{float(case.get('precision', 0)) * 100:.1f}%"
                if int(case.get("expected", 0) or 0)
                else "—",
                " ｜ ".join(case.get("retrieved", [])[:5]),
            ]
        )
    return buffer.getvalue().encode("utf-8-sig")


def matrix_markdown(rows: Sequence[Dict[str, Any]]) -> str:
    """把消融结果渲染成 Markdown 表，方便直接贴进 README / PPT。"""
    lines = [
        "| 配置 | Hit Rate | Recall | MRR | 关键词命中 | 检索精确率 | 平均耗时 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        metrics = row.get("metrics", {})
        config = row.get("config", {})
        label = f"{config.get('mode', '')} + {config.get('actual_rerank', 'off')}"
        lines.append(
            f"| {label} | {metrics.get('hit_rate', 0) * 100:.1f}% | "
            f"{metrics.get('recall', 0) * 100:.1f}% | {metrics.get('mrr', 0):.3f} | "
            f"{metrics.get('keyword_hit', 0) * 100:.1f}% | "
            f"{metrics.get('precision', 0) * 100:.1f}% | "
            f"{int(config.get('duration_ms', 0)) / 1000:.1f}s |"
        )
    return "\n".join(lines)
