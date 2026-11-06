"""管理后台界面（仅管理员可见）。

六个页签：概览 / 用户与权限 / 知识库运维 / 评测 / 审计日志 / 系统健康。
界面只做渲染和交互，数据全部来自 `backend/`，便于单独测试。
"""

import csv
import io
from typing import Dict, List

import pandas as pd
import streamlit as st

from backend import admin as admin_backend
from backend import audit
from backend import eval as eval_backend
from backend import users
from backend.config import AUDIT_KEEP_DAYS
from rag import libraries as libs


def _accounts_csv(rows: List[tuple]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["用户名", "初始密码"])
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def _drop_deleted_eval_state(run_ids: List[int]) -> None:
    """删除历史后，同步清掉页面上缓存的「本次结果 / 矩阵 / 详情选择」。"""
    deleted = {int(item) for item in run_ids}
    last = st.session_state.get("eval_last")
    if isinstance(last, dict) and int(last.get("id") or 0) in deleted:
        st.session_state.pop("eval_last", None)
    matrix = st.session_state.get("eval_matrix") or []
    if any(int(row.get("id") or 0) in deleted for row in matrix if isinstance(row, dict)):
        st.session_state.pop("eval_matrix", None)
    st.session_state.pop("eval_run_pick", None)
    st.session_state.pop("eval_delete_ids", None)


def render_admin(me: Dict[str, object]) -> None:
    st.subheader("管理后台")
    st.caption(
        f"当前账号：{me.get('username')}（{users.ROLE_LABELS.get(str(me.get('role')), '-')}）"
        " ｜ 所有敏感操作都会写入审计日志"
    )

    tabs = st.tabs(
        ["📊 概览", "👥 用户与权限", "📚 知识库运维", "🧪 评测", "🧾 审计日志", "🩺 系统健康"]
    )
    with tabs[0]:
        _render_overview()
    with tabs[1]:
        _render_users(me)
    with tabs[2]:
        _render_libraries(me)
    with tabs[3]:
        _render_eval(me)
    with tabs[4]:
        _render_audit()
    with tabs[5]:
        _render_health()


# ---------------------------------------------------------------- 概览
def _render_overview() -> None:
    st.markdown("### 📊 运营概览")
    st.caption("账号、提问量、响应耗时与今日成本的整体视图。")
    data = admin_backend.overview()
    columns = st.columns(5)
    columns[0].metric("账号数", f"{data['users_active']}/{data['users_total']}")
    columns[1].metric("今日提问", data["qa_today"])
    columns[2].metric(
        "平均响应", f"{data['avg_latency_ms'] / 1000:.1f}s" if data["avg_latency_ms"] else "—"
    )
    columns[3].metric("失败请求", data["qa_failed"])
    cost = data.get("cost_today")
    columns[4].metric(
        "今日 tokens（估）",
        f"{data['tokens_today']:,}",
        delta=f"≈ ¥{cost}" if cost else None,
        delta_color="off",
    )

    st.markdown("**最近 7 天提问量**")
    trend = pd.DataFrame(data["trend"], columns=["日期", "提问数"])
    st.bar_chart(trend, x="日期", y="提问数", height=220)

    st.markdown("**在线运行质量（近 7 天）**")
    quality = data.get("quality") or {}
    if quality.get("graded"):
        q_cols = st.columns(4)
        q_cols[0].metric(
            "引用可溯源率", f"{(quality.get('grounded_rate') or 0) * 100:.1f}%"
        )
        q_cols[1].metric(
            "平均置信度", f"{quality.get('avg_confidence') or 0:.2f}"
        )
        q_cols[2].metric("平均工具步数", f"{quality.get('avg_tool_steps') or 0:.1f}")
        q_cols[3].metric("拒答率", f"{(quality.get('refusal_rate') or 0) * 100:.1f}%")
        st.caption(
            f"统计自 {quality['graded']} 条带质量字段的提问"
            f"（近 7 天共 {quality['total']} 条，旧记录不参与）。"
        )
    else:
        st.caption("还没有带质量字段的提问记录，先去「💬 对话」问几次再回来看。")

    st.markdown("**知识库概况**")
    rows = data["libraries"]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("还没有知识库。")

    with st.expander("最近提问明细", expanded=False):
        recent = admin_backend.recent_questions(50)
        if recent:
            st.dataframe(recent, use_container_width=True, hide_index=True)
        else:
            st.caption("还没有提问记录。")


# ---------------------------------------------------------------- 用户
def _user_rows() -> List[Dict[str, object]]:
    rows = []
    for user in users.list_users():
        rows.append(
            {
                "用户名": user.username,
                "姓名": user.display_name,
                "角色": users.ROLE_LABELS.get(user.role, user.role),
                "每日配额": "不限" if not user.quota_daily else user.quota_daily,
                "今日已用": users.today_count(user.username),
                "状态": "启用" if user.active else "停用",
                "知识库": user.library_slug or "（未分配）",
                "最近登录": user.last_login or "—",
            }
        )
    return rows


def _render_users(me: Dict[str, object]) -> None:
    st.markdown("### 👥 用户与权限")
    st.caption("管理账号、角色、启用状态与每日提问额度。")
    st.markdown("**账号列表**")
    st.dataframe(_user_rows(), use_container_width=True, hide_index=True)

    names = [user.username for user in users.list_users()]
    with st.expander("新建账号", expanded=False):
        with st.form("create_user_form", clear_on_submit=True):
            cols = st.columns(2)
            username = cols[0].text_input("用户名", placeholder="3~32 位字母/数字/下划线")
            display_name = cols[1].text_input("显示名称", placeholder="例如：张三")
            role = cols[0].selectbox(
                "角色",
                options=list(users.ROLES),
                format_func=lambda value: users.ROLE_LABELS.get(value, value),
            )
            quota = cols[1].number_input(
                "每日提问配额（0 = 不限）", min_value=0, max_value=10000, value=200, step=10
            )
            library_slug = st.selectbox(
                "分配知识库",
                options=[""] + [lib.slug for lib in libs.list_libraries() if not lib.is_default],
                format_func=lambda value: "（不分配）"
                if not value
                else next((l.name for l in libs.list_libraries() if l.slug == value), value),
            )
            password = st.text_input("初始密码（留空自动生成）", type="password")
            if st.form_submit_button("创建账号", type="primary"):
                try:
                    _user, plain = users.create_user(
                        username,
                        password=password or None,
                        display_name=display_name,
                        role=role,
                        quota_daily=int(quota),
                        library_slug=library_slug,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"创建失败：{exc}")
                else:
                    audit.log(
                        str(me.get("username")), str(me.get("role")), "user_create",
                        target=_user.username, detail=f"角色 {role}",
                    )
                    st.success(f"已创建 {_user.username}，初始密码：`{plain}`（请立即告知本人）")

    with st.expander("批量创建访客账号（答辩 / 试用）", expanded=False):
        st.caption(
            "生成 `guest01`、`guest02`… 这样的账号，角色固定为访客：只能提问（受每日配额限制），"
            "不能导入或删除文献。密码只在创建后显示一次，请立即下载 CSV 或抄走。"
        )
        with st.form("bulk_guest_form"):
            cols = st.columns(3)
            prefix = cols[0].text_input("用户名前缀", value="guest", max_chars=20)
            count = cols[1].number_input(
                "数量", min_value=1, max_value=50, value=5, step=1
            )
            quota = cols[2].number_input(
                "每个账号的每日配额（0 = 不限）",
                min_value=0,
                max_value=1000,
                value=30,
                step=10,
            )
            shared = st.text_input("统一密码（留空 = 每人随机）", type="password")
            if st.form_submit_button("批量创建", type="primary"):
                try:
                    created = users.create_guests(
                        prefix, int(count), shared or None, int(quota)
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"创建失败：{exc}")
                else:
                    audit.log(
                        str(me.get("username")), str(me.get("role")), "user_create",
                        target=f"{prefix.strip() or 'guest'}*",
                        detail=f"批量创建 {len(created)} 个访客（配额 {int(quota)}/天）",
                    )
                    st.session_state["bulk_guest_result"] = [
                        (user.username, plain) for user, plain in created
                    ]
                    st.success(f"已创建 {len(created)} 个访客账号，密码见下方表格。")

    bulk_result = st.session_state.get("bulk_guest_result") or []
    if bulk_result:
        st.markdown("**本次创建的账号（密码只显示这一次）**")
        st.dataframe(
            [{"用户名": name, "初始密码": plain} for name, plain in bulk_result],
            use_container_width=True,
            hide_index=True,
        )
        _dl_col, _clear_col = st.columns([1, 1])
        _dl_col.download_button(
            "下载账号密码 CSV",
            data=_accounts_csv(bulk_result),
            file_name="guest_accounts.csv",
            mime="text/csv",
            key="bulk_guest_csv",
        )
        if _clear_col.button("清除本次结果（密码不再显示）", key="bulk_guest_clear"):
            st.session_state.pop("bulk_guest_result", None)
            st.rerun()

    if not names:
        return

    st.markdown("**账号维护**")
    target = st.selectbox("选择账号", options=names, key="admin_user_target")
    user = users.get_user(target)
    if user is None:
        return
    st.caption(
        f"角色：{users.ROLE_LABELS.get(user.role, user.role)} ｜ "
        f"状态：{'启用' if user.active else '停用'} ｜ "
        f"配额：{'不限' if not user.quota_daily else user.quota_daily}/天 ｜ "
        f"知识库：{user.library_slug or '未分配'}"
    )

    action = st.radio(
        "操作",
        options=["重置密码", "修改角色", "调整配额", "启用/停用", "分配知识库", "删除账号"],
        horizontal=True,
        key="admin_user_action",
    )

    if action == "重置密码":
        new_pwd = st.text_input("新密码（留空自动生成）", type="password", key="admin_new_pwd")
        if st.button("确认重置", type="primary", key="admin_reset_btn"):
            try:
                plain = users.set_password(target, new_pwd or None)
            except Exception as exc:  # noqa: BLE001
                st.error(f"重置失败：{exc}")
            else:
                audit.log(str(me.get("username")), str(me.get("role")), "user_reset", target=target)
                st.success(f"已重置，新密码：`{plain}`")

    elif action == "修改角色":
        role = st.selectbox(
            "新角色",
            options=list(users.ROLES),
            index=users.ROLES.index(user.role),
            format_func=lambda value: users.ROLE_LABELS.get(value, value),
            key="admin_role_value",
        )
        if st.button("保存角色", type="primary", key="admin_role_btn"):
            try:
                users.set_role(target, role)
            except Exception as exc:  # noqa: BLE001
                st.error(f"修改失败：{exc}")
            else:
                audit.log(
                    str(me.get("username")), str(me.get("role")), "user_role",
                    target=target, detail=f"{user.role} → {role}",
                )
                st.success("已更新（该用户下次操作即生效）")

    elif action == "调整配额":
        quota = st.number_input(
            "每日提问配额（0 = 不限）",
            min_value=0,
            max_value=10000,
            value=int(user.quota_daily),
            step=10,
            key="admin_quota_value",
        )
        if st.button("保存配额", type="primary", key="admin_quota_btn"):
            users.set_quota(target, int(quota))
            audit.log(
                str(me.get("username")), str(me.get("role")), "user_quota",
                target=target, detail=str(quota),
            )
            st.success("已更新")

    elif action == "启用/停用":
        if st.button(
            "停用账号" if user.active else "启用账号",
            type="primary",
            key="admin_active_btn",
        ):
            try:
                users.set_active(target, not user.active)
            except Exception as exc:  # noqa: BLE001
                st.error(f"操作失败：{exc}")
            else:
                audit.log(
                    str(me.get("username")), str(me.get("role")), "user_active",
                    target=target, detail="停用" if user.active else "启用",
                )
                st.success("已更新")
                st.rerun()

    elif action == "分配知识库":
        options = [""] + [lib.slug for lib in libs.list_libraries() if not lib.is_default]
        index = options.index(user.library_slug) if user.library_slug in options else 0
        slug = st.selectbox(
            "知识库",
            options=options,
            index=index,
            format_func=lambda value: "（不分配）"
            if not value
            else next((l.name for l in libs.list_libraries() if l.slug == value), value),
            key="admin_assign_slug",
        )
        if st.button("保存分配", type="primary", key="admin_assign_btn"):
            users.assign_library(target, slug)
            audit.log(
                str(me.get("username")), str(me.get("role")), "assign_library",
                target=target, detail=slug or "（不分配）",
            )
            st.success("已更新")

    else:
        st.warning("删除账号不可恢复，其历史会话仍会保留在库中。")
        confirmed = st.checkbox("我确认删除该账号", key="admin_delete_user_confirm")
        if st.button(
            "删除账号", type="primary", disabled=not confirmed, key="admin_delete_user_btn"
        ):
            try:
                users.delete_user(target, actor=str(me.get("username")))
            except Exception as exc:  # noqa: BLE001
                st.error(f"删除失败：{exc}")
            else:
                audit.log(
                    str(me.get("username")), str(me.get("role")), "user_delete", target=target
                )
                st.success("已删除")
                st.rerun()


# ---------------------------------------------------------------- 知识库
def _render_libraries(me: Dict[str, object]) -> None:
    st.markdown("### 📚 知识库运维")
    st.caption("查看库规模、可见性，并执行新建 / 重命名 / 删除等维护操作。")
    rows = admin_backend.library_metrics()
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("新建知识库", expanded=False):
        with st.form("admin_create_library"):
            name = st.text_input("名称", max_chars=libs.NAME_MAX_LEN)
            note = st.text_input("备注（可选）")
            owner = st.text_input("归属用户名（留空 = 共享库）")
            visibility = st.radio(
                "可见性", options=["private", "shared"], horizontal=True,
                format_func=lambda v: "私有（仅本人与管理员）" if v == "private" else "共享（所有人可见）",
            )
            if st.form_submit_button("创建", type="primary"):
                try:
                    created = libs.create_library(
                        name, note=note, owner=owner.strip(), visibility=visibility
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"创建失败：{exc}")
                else:
                    audit.log(
                        str(me.get("username")), str(me.get("role")), "create_library",
                        target=created.name, detail=f"owner={created.owner or '-'}",
                    )
                    st.success(f"已创建 {created.name}（{created.lancedb}）")

    st.markdown("**归属与可见性**")
    names = [library.name for library in libs.list_libraries()]
    if names:
        selected = st.selectbox("知识库", options=names, key="admin_lib_target")
        library = libs.get_library(selected)
        col1, col2, col3 = st.columns([1.2, 1, 1])
        owner = col1.text_input(
            "归属用户名（留空 = 共享）", value=library.owner, key="admin_lib_owner"
        )
        visibility = col2.selectbox(
            "可见性",
            options=["shared", "private"],
            index=0 if library.visibility == "shared" else 1,
            format_func=lambda v: "共享" if v == "shared" else "私有",
            key="admin_lib_visibility",
        )
        if col3.button("保存", type="primary", key="admin_lib_save"):
            try:
                libs.set_library_access(selected, owner.strip(), visibility)
            except Exception as exc:  # noqa: BLE001
                st.error(f"保存失败：{exc}")
            else:
                audit.log(
                    str(me.get("username")), str(me.get("role")), "assign_library",
                    target=selected, detail=f"owner={owner.strip() or '-'} / {visibility}",
                )
                st.success("已保存")

        st.markdown("**抓取引用关系**")
        st.caption("对该库所有在库论文重新抓取 references / cited-by，按 1 请求/秒排队。")
        if st.button("开始抓取", key="admin_refresh_graph"):
            bar = st.progress(0.0, text="准备中…")

            def _progress(done: int, total: int, title: str) -> None:
                bar.progress(
                    min(done / total, 1.0) if total else 1.0,
                    text=f"{done}/{total}：{(title or '')[:30]}",
                )

            try:
                summary = admin_backend.refresh_library_graph(selected, on_progress=_progress)
            except Exception as exc:  # noqa: BLE001
                st.error(f"抓取失败：{exc}")
            else:
                bar.empty()
                audit.log(
                    str(me.get("username")), str(me.get("role")), "refresh_graph",
                    target=selected, detail=str(summary),
                )
                st.success(
                    f"完成 {summary['papers']} 篇：references {summary['references']}、"
                    f"citations {summary['citations']}（PDF 兜底 {summary['local_references']}）"
                )

        if not library.is_default:
            st.markdown("**删除知识库**")
            st.caption("连同向量目录、引用图与抽取库一起删除，不可恢复（会先自动备份注册表）。")
            confirm = st.checkbox("我确认删除该知识库", key="admin_lib_delete_confirm")
            if st.button(
                "删除知识库",
                type="primary",
                disabled=not confirm,
                key="admin_lib_delete_btn",
            ):
                try:
                    libs.delete_library(selected)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"删除失败：{exc}")
                else:
                    audit.log(
                        str(me.get("username")), str(me.get("role")), "delete_library",
                        target=selected,
                    )
                    st.success("已删除")

    st.divider()
    st.markdown("**备份与恢复**")
    if st.button("创建备份（平台库 + 会话库 + 注册表）", key="admin_backup_btn"):
        created = admin_backend.create_backup()
        audit.log(
            str(me.get("username")), str(me.get("role")), "backup", detail=", ".join(created)
        )
        st.success("已备份：" + ("、".join(created) if created else "没有可备份的文件"))

    backups = admin_backend.list_backups()
    if backups:
        st.dataframe(
            [{k: v for k, v in row.items() if k != "path"} for row in backups],
            use_container_width=True,
            hide_index=True,
        )
        pick = st.selectbox(
            "下载备份文件", options=[row["文件"] for row in backups], key="admin_backup_pick"
        )
        target = next(row["path"] for row in backups if row["文件"] == pick)
        with open(target, "rb") as handle:
            st.download_button(
                "下载选中备份",
                data=handle.read(),
                file_name=pick,
                key="admin_backup_download",
            )

        with st.expander("从备份恢复（危险）", expanded=False):
            uploaded = st.file_uploader(
                "上传备份文件（只接受 platform-*.db / assistant-*.db / libraries-*.json）",
                type=["db", "json"],
                key="admin_restore_upload",
            )
            confirmed = st.checkbox("我确认用该备份覆盖当前数据", key="admin_restore_confirm")
            if uploaded is not None and st.button(
                "执行恢复", type="primary", disabled=not confirmed, key="admin_restore_btn"
            ):
                try:
                    path = admin_backend.restore_from_bytes(uploaded.name, uploaded.getvalue())
                except Exception as exc:  # noqa: BLE001
                    st.error(f"恢复失败：{exc}")
                else:
                    audit.log(
                        str(me.get("username")), str(me.get("role")), "restore",
                        target=uploaded.name, detail=path,
                    )
                    st.success(f"已恢复：{path}。建议重启应用以确保连接重新加载。")
    else:
        st.caption("还没有备份。点上方按钮创建。")


# ---------------------------------------------------------------- 审计
def _render_audit() -> None:
    st.markdown("### 🧾 审计日志")
    st.caption("按用户、操作与时间筛选敏感操作记录。")
    user_names = ["（全部）"] + [user.username for user in users.list_users()]
    action_values = ["（全部）"] + list(audit.ACTIONS.keys())

    col1, col2, col3 = st.columns([1.2, 1.2, 1])
    username = col1.selectbox("用户", options=user_names, key="audit_user")
    action = col2.selectbox(
        "动作",
        options=action_values,
        format_func=lambda v: audit.ACTIONS.get(v, v),
        key="audit_action",
    )
    limit = col3.number_input("条数", min_value=20, max_value=2000, value=200, step=20)

    rows = audit.recent(
        limit=int(limit),
        username=None if username == "（全部）" else username,
        action=None if action == "（全部）" else action,
    )
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button(
            "导出 CSV",
            data=audit.to_csv(rows).encode("utf-8-sig"),
            file_name="audit.csv",
            mime="text/csv",
            key="audit_export",
        )
    else:
        st.caption("没有符合条件的日志。")

    st.divider()
    st.caption(f"默认保留 {AUDIT_KEEP_DAYS} 天，可手动清理更早的记录。")
    if st.button("清理过期日志", key="audit_purge_btn"):
        removed = audit.purge_old()
        st.success(f"已清理 {removed} 条")


# ---------------------------------------------------------------- 健康
def _render_health() -> None:
    st.markdown("### 🩺 系统健康")
    st.caption("检查向量库、模型、存储与外部服务连通性。")
    items = admin_backend.health()
    st.dataframe(
        [{"检查项": row["检查项"], "状态": row["状态"], "说明": row["说明"]} for row in items],
        use_container_width=True,
        hide_index=True,
    )
    if st.button("重新检查", key="health_refresh_btn"):
        st.rerun()
    ok = sum(1 for row in items if row["ok"])
    st.caption(f"{ok}/{len(items)} 项正常。Semantic Scholar 的 429 计数只统计本次进程。")


# ---------------------------------------------------------------- 评测
def _render_eval(me: Dict[str, object]) -> None:
    st.markdown("### 🧪 评测工作台")
    st.caption(
        "离线检索评测：数据集 × 检索配置 → 指标落库，可复现、可导出。"
        "建议 20 题以内；混合检索 + 重排的耗时通常是纯向量的数倍。"
    )

    flash = st.session_state.pop("eval_flash", None)
    pending = st.session_state.pop("eval_pending_dataset", None)
    if pending:
        st.session_state["eval_dataset"] = pending
    if flash:
        st.success(flash)

    recent = eval_backend.list_runs(1)
    st.markdown("#### 评测概览")
    if recent:
        last = recent[0]
        st.caption(
            f"最近一次 #{last['ID']} · {last['时间']} · {last['数据集']} · "
            f"{last['模式']}/{last['重排']} · top-{last['top-k']} · {last['题数']} 题"
        )
        m_cols = st.columns(5)
        m_cols[0].metric("Hit Rate", f"{last['Hit Rate']}%")
        m_cols[1].metric("Recall@k", f"{last['Recall']}%")
        m_cols[2].metric("MRR", f"{last['MRR']}")
        m_cols[3].metric("关键词命中", f"{last['关键词']}%")
        m_cols[4].metric("检索精确率", f"{last['精确率']}%")
    else:
        st.info("还没有评测记录。选择数据集并运行一次评测后，这里会显示最新指标。")
    st.divider()

    st.markdown("#### 1. 数据集")
    with st.container(border=True):
        st.markdown("**数据集管理**")
        st.caption("上传自定义评测集，或下载模板按字段填写。")
        st.download_button(
            "下载模板 JSON",
            data=eval_backend.template_dataset(),
            file_name="eval_questions_template.json",
            mime="application/json",
            key="eval_template_download",
        )
        uploaded = st.file_uploader(
            "上传评测集（字段：question / expected_sources / expected_keywords）",
            type=["json"],
            key="eval_upload",
        )
        save_name = st.text_input(
            "保存名称（留空使用文件名）", key="eval_upload_name"
        )
        if uploaded is not None and st.button("保存数据集", key="eval_upload_save"):
            try:
                saved = eval_backend.save_uploaded_dataset(
                    save_name.strip() or uploaded.name, uploaded.getvalue()
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"保存失败：{exc}")
            else:
                audit.log(
                    str(me.get("username")), str(me.get("role")), "eval_upload",
                    target=saved["name"], detail=f"{saved['cases']} 题",
                )
                st.session_state["eval_pending_dataset"] = saved["path"]
                st.session_state["eval_flash"] = (
                    f"已保存 {saved['name']}（{saved['cases']} 题），已自动选中。"
                )
                st.rerun()

    datasets = eval_backend.list_datasets()
    if not datasets:
        st.warning("没有找到数据集：上传 JSON，或确认 `rag/eval_questions*.json` 存在。")
        return

    labels: Dict[str, str] = {}
    for row in datasets:
        flag = "" if row["可用"] else "（不可用）"
        mode_label = {"llm": " · LLM 出题", "template": " · 模板出题"}.get(
            str(row.get("出题方式") or ""), ""
        )
        labels[str(row["path"])] = (
            f"{row['名称']} · {row['题数']} 题 · {row['来源']}{mode_label}{flag}"
        )
    dataset = st.selectbox(
        "评测数据集", options=list(labels), format_func=labels.get, key="eval_dataset"
    )
    meta = next(row for row in datasets if str(row["path"]) == dataset)
    if not meta["可用"]:
        st.error(f"数据集不可用：{meta['问题']}")
        return
    st.caption(
        f"{meta['文件']} ｜ 更新于 {meta['更新时间']} ｜ {meta['大小(KB)']} KB ｜ "
        f"共 {meta['题数']} 题"
    )

    st.markdown("#### 2. 检索配置")
    with st.container(border=True):
        st.caption(
            "评测只读检索结果，不会调用大模型生成回答；`llm` 重排会调用模型，费用更高。"
        )
        library_names = [library.name for library in libs.list_libraries()]
        cols = st.columns([1.5, 1, 1, 0.8, 0.8])
        library = cols[0].selectbox("知识库", options=library_names, key="eval_library")
        mode = cols[1].selectbox(
            "检索模式", options=list(eval_backend.MODES), key="eval_mode"
        )
        rerank = cols[2].selectbox(
            "重排", options=list(eval_backend.RERANKS), key="eval_rerank"
        )
        top_k = cols[3].number_input(
            "top-k", min_value=1, max_value=20, value=5, step=1, key="eval_topk"
        )
        default_limit = min(int(meta["题数"]) or 1, eval_backend.MAX_CASES, 20)
        limit = cols[4].number_input(
            "题数上限",
            min_value=1,
            max_value=eval_backend.MAX_CASES,
            value=default_limit,
            step=1,
            key="eval_limit",
        )

    action_cols = st.columns([1, 1, 3])
    if action_cols[0].button("🔍 体检数据集", key="eval_check_btn"):
        with st.spinner("读取知识库来源清单（首次会稍慢）…"):
            try:
                report = eval_backend.check_dataset(dataset, library)
            except Exception as exc:  # noqa: BLE001
                st.error(f"体检失败：{exc}")
            else:
                st.session_state["eval_check"] = {
                    "dataset": dataset,
                    "library": library,
                    **report,
                }
    action_cols[1].download_button(
        "下载数据集 JSON",
        data=eval_backend.dataset_bytes(dataset),
        file_name=f"{meta['名称']}.json",
        mime="application/json",
        key="eval_dataset_download",
    )

    check = st.session_state.get("eval_check")
    if check and check.get("dataset") == dataset and check.get("library") == library:
        if check["positive"] and check["missing"]:
            st.warning(
                f"{check['missing']}/{check['positive']} 道正样本的期望来源在"
                f"「{library}」里找不到，指标会偏低甚至为 0。"
            )
        elif check["positive"]:
            st.success(
                f"全部 {check['positive']} 道正样本的期望来源都能在「{library}」中找到。"
            )
        st.dataframe(check["rows"], use_container_width=True, hide_index=True)
        if check["known"]:
            with st.expander("库里可用的来源（可直接复制到 expected_sources）"):
                st.dataframe(check["known"], use_container_width=True, hide_index=True)
        else:
            st.info("知识库里还没有可识别的来源，请先导入文献。")

    with st.expander("🪄 生成评测集（从当前知识库自动出题）", expanded=False):
        st.caption(
            "按文献抽样出题，`expected_sources` 自动写成库里的 source_key，"
            "生成后可直接运行评测。模板模式零成本、题面较简单（适合冒烟/演示）；"
            "LLM 模式每题一次模型调用，题面更自然，费用与耗时随题数线性增长。"
        )
        gen_cols = st.columns([1, 1.3, 1, 1.6])
        gen_count = gen_cols[0].number_input(
            "题数", min_value=1, max_value=30, value=10, step=1, key="gen_count"
        )
        gen_mode = gen_cols[1].selectbox(
            "出题方式",
            options=["template", "llm"],
            format_func=lambda value: "模板（零成本，题面较简单）"
            if value == "template"
            else "LLM（更自然，需 Key）",
            key="gen_mode",
        )
        gen_neg = gen_cols[2].number_input(
            "负样本", min_value=0, max_value=5, value=2, step=1, key="gen_neg"
        )
        gen_name = gen_cols[3].text_input(
            "保存名称（留空自动命名）", key="gen_name"
        )
        gen_actions = st.columns([1, 1.4])
        save_now = gen_actions[0].button("生成并保存", key="gen_btn")
        run_now = gen_actions[1].button(
            "生成并运行（记一条历史）", type="primary", key="gen_run_btn"
        )
        st.caption(
            "「生成并运行」会用当前知识库/模式/重排/top-k 立即跑完整评测，"
            "并写入历史（来源=生成、出题方式=模板/LLM）；评测全部生成的题，不受上方题数上限影响。"
        )
        if save_now or run_now:
            bar = st.progress(0.0, text="抽样文献…")

            def _gen_progress(done: int, total: int, question: str) -> None:
                bar.progress(
                    min(done / total, 1.0) if total else 1.0,
                    text=f"出题 {done}/{total}：{(question or '')[:36]}",
                )

            try:
                generated = eval_backend.generate_dataset(
                    str(me.get("username")),
                    library,
                    count=int(gen_count),
                    mode=gen_mode,
                    negatives=int(gen_neg),
                    name=gen_name,
                    on_progress=_gen_progress,
                )
            except Exception as exc:  # noqa: BLE001
                bar.empty()
                st.error(f"生成失败：{exc}")
            else:
                audit.log(
                    str(me.get("username")), str(me.get("role")), "eval_upload",
                    target=str(generated["name"]),
                    detail=f"自动生成 {generated['cases']} 题（{gen_mode}）",
                )
                if run_now:
                    bar.progress(0.0, text="评测中…")

                    def _eval_progress(done: int, total: int, question: str) -> None:
                        bar.progress(
                            min(done / total, 1.0) if total else 1.0,
                            text=f"评测 {done}/{total}：{(question or '')[:36]}",
                        )

                    try:
                        result = eval_backend.run(
                            str(me.get("username")),
                            generated["path"],
                            library,
                            mode=mode,
                            rerank=rerank,
                            top_k=int(top_k),
                            limit=None,
                            on_progress=_eval_progress,
                        )
                    except Exception as exc:  # noqa: BLE001
                        bar.empty()
                        st.session_state["eval_pending_dataset"] = generated["path"]
                        st.session_state["eval_flash"] = (
                            f"已生成 {generated['name']}，但评测失败：{exc}"
                        )
                        st.rerun()
                    else:
                        bar.empty()
                        audit.log(
                            str(me.get("username")), str(me.get("role")), "eval_run",
                            target=str(generated["name"]),
                            detail=(
                                f"生成并运行 · {mode}/{result['config']['actual_rerank']} · "
                                f"{result['config']['cases']} 题 · "
                                f"Hit {result['metrics']['hit_rate'] * 100:.1f}%"
                            ),
                        )
                        st.session_state["eval_last"] = result
                        st.session_state["eval_pending_dataset"] = generated["path"]
                        fallback_note = (
                            f"（{generated['fallbacks']} 题 LLM 出题失败，已退回模板）"
                            if generated.get("fallbacks")
                            else ""
                        )
                        st.session_state["eval_flash"] = (
                            f"已生成并完成评测：{generated['name']}（正样本 "
                            f"{generated['positives']} / 负样本 {generated['negatives']}）"
                            f" · Hit {result['metrics']['hit_rate'] * 100:.1f}%"
                            f" · 已写入历史 #{result['id']}{fallback_note}"
                        )
                        st.rerun()
                else:
                    bar.empty()
                    st.session_state["eval_pending_dataset"] = generated["path"]
                    fallback_note = (
                        f"（{generated['fallbacks']} 题 LLM 出题失败，已退回模板）"
                        if generated.get("fallbacks")
                        else ""
                    )
                    st.session_state["eval_flash"] = (
                        f"已生成 {generated['name']}：正样本 {generated['positives']} / "
                        f"负样本 {generated['negatives']}，已自动选中。{fallback_note}"
                    )
                    st.rerun()

    st.caption(
        f"当前配置：{meta['名称']} · {library} · {mode}/{rerank} · "
        f"top-{top_k} · 最多 {limit} 题"
    )
    st.markdown("#### 3. 运行评测")
    if st.button("▶ 运行评测", type="primary", key="eval_run_btn"):
        bar = st.progress(0.0, text="初始化知识库与 embedding…")

        def _progress(done: int, total: int, question: str) -> None:
            bar.progress(
                min(done / total, 1.0) if total else 1.0,
                text=f"{done}/{total}：{(question or '')[:36]}",
            )

        try:
            result = eval_backend.run(
                str(me.get("username")),
                dataset,
                library,
                mode=mode,
                rerank=rerank,
                top_k=int(top_k),
                limit=int(limit),
                on_progress=_progress,
            )
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            st.error(f"评测失败：{exc}")
        else:
            bar.empty()
            config = result["config"]
            audit.log(
                str(me.get("username")), str(me.get("role")), "eval_run",
                target=str(meta["名称"]),
                detail=(
                    f"{config['mode']}/{config['actual_rerank']} · "
                    f"{config['cases']} 题 · Hit {result['metrics']['hit_rate'] * 100:.1f}%"
                ),
            )
            st.session_state["eval_last"] = result
            st.success(
                f"完成：{config['cases']} 题 / {config['duration_ms'] / 1000:.1f}s，"
                f"已保存为记录 #{result['id']}"
            )

    _render_eval_result(st.session_state.get("eval_last"))

    st.markdown("#### 4. 消融对比")
    with st.expander("消融对比（纯向量 / 混合 / 混合+重排）", expanded=False):
        st.caption(
            "依次跑 3 组配置，耗时约为单次评测的 3 倍，每组各存一条记录；"
            "结果可直接生成 Markdown 表贴进 README / PPT。"
        )
        if st.button("运行消融矩阵", key="eval_matrix_btn"):
            bar = st.progress(0.0, text="准备中…")

            def _matrix_progress(
                label: str, done: int, total: int, question: str
            ) -> None:
                bar.progress(
                    min(done / total, 1.0) if total else 1.0,
                    text=f"{label} {done}/{total}：{(question or '')[:24]}",
                )

            try:
                rows = eval_backend.run_matrix(
                    str(me.get("username")),
                    dataset,
                    library,
                    top_k=int(top_k),
                    limit=int(limit),
                    on_progress=_matrix_progress,
                )
            except Exception as exc:  # noqa: BLE001
                bar.empty()
                st.error(f"消融评测失败：{exc}")
            else:
                bar.empty()
                st.session_state["eval_matrix"] = rows
                audit.log(
                    str(me.get("username")), str(me.get("role")), "eval_run",
                    target=str(meta["名称"]),
                    detail=f"消融矩阵 {len(rows)} 组 · {int(limit)} 题",
                )
                st.success(f"完成 {len(rows)} 组配置，结果已分别落库。")

        matrix_rows = st.session_state.get("eval_matrix")
        if matrix_rows:
            first_config = matrix_rows[0]["config"]
            st.caption(
                f"数据集：{first_config['dataset']} ｜ 知识库：{first_config['library']} ｜ "
                f"top-{first_config['top_k']} ｜ {first_config['cases']} 题"
            )
            if any(
                row["config"].get("rerank") not in (None, "off")
                and row["config"].get("actual_rerank") == "off"
                for row in matrix_rows
            ):
                st.warning(
                    "部分配置的重排不可用，实际按「不重排」跑完；对比前先确认 "
                    "cross-encoder 已安装/已缓存，或改用 `llm` 重排。"
                )
            st.dataframe(
                [
                    {
                        "配置": f"{row['config']['mode']} + {row['config']['actual_rerank']}",
                        "Hit Rate": f"{row['metrics']['hit_rate'] * 100:.1f}%",
                        "Recall": f"{row['metrics']['recall'] * 100:.1f}%",
                        "MRR": round(row["metrics"]["mrr"], 3),
                        "关键词命中": f"{row['metrics']['keyword_hit'] * 100:.1f}%",
                        "检索精确率": f"{row['metrics']['precision'] * 100:.1f}%",
                        "耗时(s)": round(row["config"]["duration_ms"] / 1000, 1),
                    }
                    for row in matrix_rows
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.code(eval_backend.matrix_markdown(matrix_rows), language="markdown")

    st.divider()
    st.markdown("#### 5. 历史记录")
    runs = eval_backend.list_runs(200)
    if not runs:
        st.caption("还没有评测记录。")
        return
    st.dataframe(runs, use_container_width=True, hide_index=True)
    st.caption(
        f"共 {len(runs)} 条记录（最多显示最近 200 条）；"
        "删除/清空只影响历史记录，不会删除数据集 JSON。"
    )
    st.download_button(
        "导出历史 CSV",
        data=eval_backend.runs_csv(runs),
        file_name="eval_runs.csv",
        mime="text/csv",
        key="eval_runs_csv",
    )
    run_map = {int(row["ID"]): row for row in runs}
    with st.expander("管理历史记录（删除）", expanded=False):
        st.caption("删除只影响历史记录，不会删除数据集文件；删除后不可恢复。")
        selected_ids = st.multiselect(
            "选择要删除的记录",
            options=list(run_map),
            format_func=lambda rid: (
                f"#{rid} · {run_map[int(rid)]['数据集']} · "
                f"{run_map[int(rid)]['模式']}/{run_map[int(rid)]['重排']} · "
                f"Hit {run_map[int(rid)]['Hit Rate']}%"
            ),
            key="eval_delete_ids",
        )
        confirm = st.checkbox("我确认删除选中的记录", key="eval_delete_confirm")
        if st.button(
            "删除选中",
            type="primary",
            disabled=not (selected_ids and confirm),
            key="eval_delete_btn",
        ):
            try:
                removed = eval_backend.delete_runs(selected_ids)
            except Exception as exc:  # noqa: BLE001
                st.error(f"删除失败：{exc}")
            else:
                audit.log(
                    str(me.get("username")), str(me.get("role")), "eval_delete",
                    target="eval_runs",
                    detail=f"删除 {removed} 条：{', '.join(str(i) for i in selected_ids)}",
                )
                _drop_deleted_eval_state(selected_ids)
                st.rerun()

        st.divider()
        confirm_all = st.checkbox(
            "我确认清空全部评测历史", key="eval_clear_confirm"
        )
        if st.button(
            "清空全部历史",
            disabled=not confirm_all,
            key="eval_clear_btn",
        ):
            try:
                removed = eval_backend.clear_runs()
            except Exception as exc:  # noqa: BLE001
                st.error(f"清空失败：{exc}")
            else:
                audit.log(
                    str(me.get("username")), str(me.get("role")), "eval_clear",
                    target="eval_runs", detail=f"清空 {removed} 条",
                )
                st.session_state.pop("eval_last", None)
                st.session_state.pop("eval_matrix", None)
                st.session_state.pop("eval_run_pick", None)
                st.session_state.pop("eval_delete_ids", None)
                st.rerun()

    picked = st.selectbox(
        "查看某次详情",
        options=list(run_map),
        format_func=lambda rid: (
            f"#{rid} · {run_map[int(rid)]['数据集']} · "
            f"{run_map[int(rid)]['模式']}/{run_map[int(rid)]['重排']} · "
            f"Hit {run_map[int(rid)]['Hit Rate']}%"
        ),
        key="eval_run_pick",
    )
    detail = eval_backend.get_run(int(picked))
    summary = detail["summary"]
    st.caption(
        f"#{summary['ID']} ｜ {summary['数据集']}（{summary['数据集来源']}） ｜ "
        f"{summary['时间']} ｜ {summary['知识库']} ｜ "
        f"top-{summary['top-k']} ｜ {summary['题数']} 题 ｜ {summary['耗时(s)']}s"
    )
    m = detail["metrics"]
    st.caption(
        f"正样本 {m.get('positives', 0)} / 负样本 {m.get('negatives', 0)} ｜ "
        f"关键词题 {m.get('keyword_cases', 0)}"
    )
    d_cols = st.columns(5)
    d_cols[0].metric("Hit Rate", f"{m.get('hit_rate', 0) * 100:.1f}%")
    d_cols[1].metric("Recall@k", f"{m.get('recall', 0) * 100:.1f}%")
    d_cols[2].metric("MRR", f"{m.get('mrr', 0):.3f}")
    d_cols[3].metric("关键词命中", f"{m.get('keyword_hit', 0) * 100:.1f}%")
    d_cols[4].metric("检索精确率", f"{m.get('precision', 0) * 100:.1f}%")
    if detail["cases"]:
        st.dataframe(
            [
                {
                    "问题": case.get("question", ""),
                    "期望来源": " ｜ ".join(case.get("expected_sources") or [])
                    or ("（负样本）" if not case.get("expected") else "—"),
                    "命中名次": case.get("rank") or "未命中",
                    "来源覆盖": f"{case.get('matched', 0)}/{case.get('expected', 0)}",
                    "关键词覆盖": f"{float(case.get('keyword_hit', 0)) * 100:.1f}%"
                    if int(case.get("keywords_expected", 0) or 0)
                    else "—",
                    "检索片段": " ｜ ".join(case.get("retrieved", [])[:5]),
                }
                for case in detail["cases"]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "导出该次逐题 CSV",
            data=eval_backend.cases_csv(detail["cases"]),
            file_name=f"eval_run_{summary['ID']}_cases.csv",
            mime="text/csv",
            key="eval_detail_csv",
        )


def _render_eval_result(result) -> None:
    if not result:
        return
    config = result["config"]
    metrics = result["metrics"]
    st.markdown("**本次结果**")
    st.caption(
        f"{config['dataset']} ｜ {config['library']} ｜ "
        f"{config['mode']} + {config['actual_rerank']} ｜ top-{config['top_k']} ｜ "
        f"{config['cases']} 题 ｜ {config['duration_ms'] / 1000:.1f}s ｜ "
        f"{config['chunks']} 个片段"
    )
    counts = (
        f"正样本 {metrics.get('positives', 0)} / 负样本 {metrics.get('negatives', 0)}"
        f"（负样本不参与 Hit / Recall / MRR）｜ "
        f"关键词题 {metrics.get('keyword_cases', 0)}"
    )
    st.caption(counts)
    if config.get("rerank") not in (None, "off") and config.get("actual_rerank") == "off":
        st.warning(
            "所选重排不可用，本次实际未重排（检查 sentence-transformers 是否安装、"
            "模型是否已缓存，或改用 `llm` 重排）。"
        )
    if not metrics.get("positives"):
        st.warning("这个数据集没有正样本（expected_sources 全为空），无法计算检索指标。")
    elif metrics.get("hit_rate", 0) == 0:
        st.info(
            "所有题都未命中：先用上方「体检数据集」确认 expected_sources 是否"
            "与库里的 source_key / 标题一致，或换用「生成评测集」自动出题。"
        )
    cols = st.columns(5)
    previous = None
    current_id = int(result.get("id") or 0)
    if current_id:
        try:
            for row in eval_backend.list_runs(3):
                if int(row["ID"]) != current_id:
                    previous = row
                    break
        except Exception:  # noqa: BLE001 - 对比失败不影响看本次结果
            previous = None

    def _pct_delta(value: float, key: str):
        return None if previous is None else round(value * 100 - float(previous[key]), 1)

    def _num_delta(value: float, key: str):
        return None if previous is None else round(float(value) - float(previous[key]), 3)

    cols[0].metric(
        "Hit Rate", f"{metrics['hit_rate'] * 100:.1f}%",
        delta=_pct_delta(metrics["hit_rate"], "Hit Rate"),
    )
    cols[1].metric(
        "Recall@k", f"{metrics['recall'] * 100:.1f}%",
        delta=_pct_delta(metrics["recall"], "Recall"),
    )
    cols[2].metric(
        "MRR", f"{metrics['mrr']:.3f}",
        delta=_num_delta(metrics["mrr"], "MRR"),
    )
    cols[3].metric(
        "关键词命中", f"{metrics['keyword_hit'] * 100:.1f}%",
        delta=_pct_delta(metrics["keyword_hit"], "关键词"),
    )
    cols[4].metric(
        "检索精确率", f"{metrics['precision'] * 100:.1f}%",
        delta=_pct_delta(metrics["precision"], "精确率"),
    )
    if previous is not None:
        st.caption(f"Δ 对比上一次评测 #{previous['ID']}（{previous['时间']}）")
    st.dataframe(result["case_rows"], use_container_width=True, hide_index=True)
    st.download_button(
        "下载逐题 CSV",
        data=eval_backend.cases_csv(result["cases"]),
        file_name="eval_cases.csv",
        mime="text/csv",
        key="eval_last_csv",
    )
    st.caption("命令行复现（CLI 作用于默认知识库；其它库请在界面里跑）：")
    st.code(result["reproduce"], language="bash")
