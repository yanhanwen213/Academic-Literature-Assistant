"""检查 Semantic Scholar 的 key 是否生效、当前实际能承受多快的请求频率。

用法：
    python scripts/check_s2.py [--probe 4] [--gap 1.5]

它只做 3~5 次 `/paper/search` 请求（每次间隔 gap 秒），把每次的状态码、
耗时和 429 的响应体打印出来。用来判断：
- key 没被读到  → 每次都是 429（S2 的匿名共享池，几乎必挂）
- key 被别处共用/额度吃紧 → 间隔 1.5~2s 仍然间断 429
- key 正常 → 全部 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def probe(url: str, key: str) -> tuple[int, float, str]:
    headers = {"User-Agent": "AcademicLiteratureAssistant/1.0 (diagnose)"}
    if key:
        headers["x-api-key"] = key
    request = urllib.request.Request(url, headers=headers)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read(80)
            return response.status, time.monotonic() - started, ""
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read(160).decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001
            pass
        return exc.code, time.monotonic() - started, body
    except Exception as exc:  # noqa: BLE001
        return 0, time.monotonic() - started, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="诊断 Semantic Scholar 限流")
    parser.add_argument("--probe", type=int, default=4, help="探测次数，默认 4")
    parser.add_argument("--gap", type=float, default=1.5, help="每次间隔秒数，默认 1.5")
    args = parser.parse_args()

    _load_env()
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    base = os.getenv(
        "SEMANTIC_SCHOLAR_API", "https://api.semanticscholar.org/graph/v1"
    )
    url = f"{base}/paper/search?query=lrfvnet&limit=1&fields=paperId"

    print(f"key: {'已配置（%d 字符）' % len(key) if key else '未配置（匿名共享池，几乎必 429）'}")
    print(f"探测: {args.probe} 次 / 间隔 {args.gap}s / {url}\n")

    codes: list[int] = []
    for index in range(args.probe):
        if index:
            time.sleep(args.gap)
        status, elapsed, body = probe(url, key)
        codes.append(status)
        extra = f"  body={body[:90]}" if body else ""
        print(f"  #{index + 1}  HTTP {status}  {elapsed:.2f}s{extra}")

    ok = sum(1 for code in codes if code == 200)
    print()
    if ok == len(codes):
        print("结论：全部成功，当前间隔可以放心使用。")
    elif ok == 0:
        print(
            "结论：全部 429。\n"
            "  1) 先确认 key 已写进 .env 且被读到了（上面显示「已配置」才对）；\n"
            "  2) key 是账号级的 1 请求/秒，如果别的脚本/另一台机器也在用同一个 key，\n"
            "     两边加起来就会超；\n"
            "  3) 免费 key 在高峰期确实会被共享限流，等几分钟或调大间隔。"
        )
    else:
        print(
            f"结论：{ok}/{len(codes)} 成功，说明 key 有效但额度吃紧（被共享或在高峰期）。\n"
            "  把间隔调到 2.5s 以上会比较稳：在 .env 里写 SEMANTIC_SCHOLAR_MIN_INTERVAL=2.5"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
