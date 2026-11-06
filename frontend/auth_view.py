"""登录页：只负责渲染，口令校验交给 `backend.auth`。

登录状态存在 `st.session_state["auth"]` 里；每次 rerun 由 app.py 回数据库
复核一次角色/启用状态，所以这里不需要维护额外的会话表。
"""

import base64
from pathlib import Path

import streamlit as st

from title_orbits import orbit_title, orbit_style

from backend import auth as auth_backend


@st.cache_data(show_spinner=False)
def _asset_base64(path_str: str) -> str:
    """缓存大图的 base64，避免每次 rerun 都重新读盘 + 编码。"""
    return base64.b64encode(Path(path_str).read_bytes()).decode("ascii")


def render_login() -> None:
    st.markdown(orbit_style(), unsafe_allow_html=True)
    from urllib.parse import quote

    drawing = """<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000"><defs><linearGradient id="p" x2="1" y2="1"><stop stop-color="#ddd0f6"/><stop offset="1" stop-color="#f7dce9"/></linearGradient><linearGradient id="paper" x2="1" y2="1"><stop stop-color="#fff"/><stop offset="1" stop-color="#ede4fa"/></linearGradient><pattern id="dots" width="32" height="32" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#af96ce" opacity=".18"/></pattern></defs><rect width="1600" height="1000" fill="url(#dots)"/><g fill="none" stroke="#bda6dd" opacity=".4"><ellipse cx="790" cy="510" rx="730" ry="290" transform="rotate(-24 790 510)"/><ellipse cx="790" cy="510" rx="760" ry="320" transform="rotate(-24 790 510)"/><path d="M0 780Q400 420 700 650T1600 350" stroke-dasharray="5 10"/></g><g transform="translate(1320 120) scale(0.62) rotate(12 140 160)"><rect x="20" y="22" width="230" height="290" rx="20" fill="#bca4d6" opacity=".14"/><rect width="230" height="290" rx="18" fill="url(#paper)" stroke="#d9c9eb"/><rect x="30" y="35" width="45" height="8" rx="4" fill="#aa8dcb"/><path d="M30 70H190M30 85H155" stroke="#9371b8" stroke-width="7" stroke-linecap="round"/><path d="M30 115H195M30 130H185M30 145H195M30 160H145" stroke="#d7c8e7" stroke-width="4" stroke-linecap="round"/><rect x="30" y="186" width="170" height="70" rx="8" fill="#eee5f7"/><path d="M45 240L75 220 102 230 135 200 180 211" fill="none" stroke="#b193d0" stroke-width="3"/><circle cx="135" cy="200" r="5" fill="#e8a9c5"/></g><g transform="translate(1090 690) scale(0.85)"><circle cx="90" cy="100" r="115" fill="url(#p)" opacity=".4"/><g stroke="#b49acc" stroke-width="2"><path d="M0 100L80 30 170 65 145 175 40 200 0 100 145 175 80 30 40 200M0 100L170 65" fill="none"/></g><g fill="#f9f6ff" stroke="#ac8dca" stroke-width="2"><circle cy="100" r="15"/><circle cx="80" cy="30" r="23"/><circle cx="170" cy="65" r="13"/><circle cx="145" cy="175" r="25"/><circle cx="40" cy="200" r="12"/></g><circle cx="80" cy="30" r="9" fill="#c0a5dd"/><circle cx="145" cy="175" r="10" fill="#ecc0d4"/></g><g fill="#ae8bce" opacity=".5"><path d="M310 180v20m-10-10h20M1290 740v24m-12-12h24M1450 220v14m-7-7h14" stroke="#ae8bce" stroke-width="2"/><circle cx="210" cy="720" r="5"/><circle cx="1180" cy="200" r="4"/><circle cx="1400" cy="800" r="3"/></g></svg>"""
    st.markdown("""<style>
    [data-testid="stAppViewContainer"] {background:#f7f7fa;font-family:"Inter","Microsoft YaHei",sans-serif;}
    [data-testid="stHeader"] {background:transparent;}
    .stMainBlockContainer {max-width:1200px;padding-top:clamp(30px,7vh,80px);}
    [data-testid="stForm"] {background:white;border:1px solid #e9e5ef;border-radius:22px;padding:40px;box-shadow:0 16px 55px #3020400b;animation:appear .5s ease-out;}
    .brand {text-align:center;margin-bottom:28px;}
    .brand .logo {display:inline-flex;align-items:center;justify-content:center;width:60px;height:60px;background:#eee9f5;color:#82699f;border-radius:15px;font-size:30px;margin-bottom:16px;}
    .brand h1 {color:#302a3c;font-size:32px;letter-spacing:2px;padding:0;margin:0 0 12px;}
    .brand p {color:#958d9f;font-size:14px;line-height:1.8;}
    [data-testid="stForm"] [data-baseweb="input"],[data-testid="stFormSubmitButton"] button,[data-testid="stExpander"] {position:relative;isolation:isolate;overflow:hidden;background:#f9fafb;border:1px solid #737373;border-radius:8px;transition:border-color .3s,box-shadow .3s;}
    [data-testid="stForm"] [data-baseweb="input"]::before,[data-testid="stFormSubmitButton"] button::before,[data-testid="stExpander"]::before {content:"";position:absolute;z-index:-1;pointer-events:none;width:48px;height:48px;right:0;top:0;border-radius:50%;background:#8b5cf6;filter:blur(16px);box-shadow:-60px 20px 10px 10px #f9b0b9;opacity:.5;transition:opacity .3s,transform .3s;}
    [data-testid="stForm"] [data-baseweb="input"]:hover::before,[data-testid="stFormSubmitButton"] button:hover::before,[data-testid="stExpander"]:hover::before {opacity:.75;transform:translateX(-6px);}
    [data-testid="stForm"] [data-baseweb="input"]:focus-within {border-color:#8b5cf6;box-shadow:0 0 0 2px #8b5cf626;}
    [data-testid="stForm"] [data-baseweb="base-input"],[data-testid="stForm"] input {background:transparent!important;}
    [data-testid="stForm"] input {min-height:52px;font-size:15px;font-weight:600;color:#171717;}
    [data-testid="stForm"] input::placeholder {color:#8b6aa9;font-weight:400;opacity:1;}
    [data-testid="stFormSubmitButton"] button {color:#6d28d9;min-height:54px;margin-top:16px;font-weight:700;}
    [data-testid="stFormSubmitButton"] button:hover {background:#f9fafb;color:#6d28d9;border-color:#8b5cf6;box-shadow:0 3px 12px #8b5cf61a;}
    [data-testid="stExpander"] {color:#73528d;}
    [data-testid="stExpander"] summary svg {color:#f9a8d4;transition:transform .3s;}
    [data-testid="stExpander"] summary:hover svg {transform:rotate(45deg);}
    .foot {text-align:center;color:#aaa3b3;font-size:12px;line-height:2;margin:25px 0;}
    @keyframes appear {from {opacity:0;transform:translateY(10px);}to {opacity:1;transform:translateY(0);}}
    @media(prefers-reduced-motion:reduce) {* {animation:none!important;transition:none!important;}}
    @media(max-width:640px) {[data-testid="stForm"] {padding:24px 20px;}.stMainBlockContainer {padding-top:30px;}}
    [data-testid="stAppViewContainer"] {background-image:url("data:image/svg+xml,ARTWORK"),radial-gradient(ellipse at 12% 20%,#e8dcfa 0%,transparent 48%),radial-gradient(ellipse at 90% 75%,#f7dce9 0%,transparent 48%),linear-gradient(135deg,#f8f5fc,#f1eef9 55%,#fbf3f7);background-size:cover;background-position:center;background-attachment:fixed;}
    [data-testid="stAppViewContainer"]::before {content:"";position:fixed;left:8%;top:12%;width:240px;height:240px;border-radius:50%;background:#d3b5f333;filter:blur(45px);pointer-events:none;animation:ambient-drift 12s ease-in-out infinite alternate;}
    [data-testid="stForm"] {background:linear-gradient(145deg,#ffffffed,#fcf8ffe6);backdrop-filter:blur(18px);border:1px solid #ffffff;box-shadow:0 24px 70px #76549b20,0 2px 8px #ffffff80 inset;border-radius:24px;}
    .brand .logo {background:linear-gradient(140deg,#ede1fc,#f9dfe9);box-shadow:0 6px 18px #a278c826;border:1px solid #ffffff;}
    .brand h1 {color:#493160;}.brand p {color:#937da5;}
    .foot {color:#927ca5;text-shadow:0 1px 8px #fff;}
    [data-testid="stExpander"] {background:#fffafddd;border-color:#cfbbdf;backdrop-filter:blur(12px);}
    @keyframes ambient-drift {from {transform:translate(0,0) scale(1);}to {transform:translate(75px,45px) scale(1.25);}}
    @media(max-width:850px) {[data-testid="stAppViewContainer"] {background-size:auto 100%;}[data-testid="stAppViewContainer"]::before {width:130px;height:130px;}}
    </style>""".replace("ARTWORK", quote(drawing)), unsafe_allow_html=True)
    reader_image = _asset_base64(
        str(Path(__file__).parent / "assets" / "login_reader.png")
    )
    st.markdown("""<style>
    .login-reader-art {position:fixed;left:0;bottom:0;width:clamp(460px,56vw,880px);height:auto;z-index:0;pointer-events:none;user-select:none;opacity:.92;filter:drop-shadow(0 12px 24px #84639c18);transform-origin:50% 90%;animation:reader-breathe 7s ease-in-out infinite;}
    @keyframes reader-breathe {0%,100% {transform:translateY(0) rotate(-.4deg);}50% {transform:translateY(-8px) rotate(.4deg);}}
    @media(max-width:900px) {.login-reader-art {width:46vw;opacity:.75;}}
    @media(max-width:640px) {.login-reader-art {width:220px;opacity:.4;}}
    @media(prefers-reduced-motion:reduce) {.login-reader-art {animation:none;}}
    </style>""" + '<img class="login-reader-art" src="data:image/png;base64,' + reader_image + '" alt="" aria-hidden="true" />', unsafe_allow_html=True)
    _left, middle, _right = st.columns([1, 1.9, 1])
    with middle:
        st.markdown('<div class="foot">YOUR RESEARCH COMPANION</div>', unsafe_allow_html=True)


        generated = st.session_state.get("_bootstrap_password")
        if generated:
            st.warning(
                "首次启动，已自动创建管理员账号：\n\n"
                f"**用户名**：`{st.session_state.get('_bootstrap_user', 'admin')}`　"
                f"**密码**：`{generated}`\n\n"
                "登录后请到「管理 → 用户与权限」立即修改密码。"
            )

        with st.form("login_form"):
            st.markdown('<div class="brand"><div class="logo">▤</div><h1>' + orbit_title() + '</h1><p>让阅读更深入，让研究更进一步</p></div>', unsafe_allow_html=True)
            username = st.text_input("账号", placeholder="请输入您的账号", key="login_username", autocomplete="username")
            password = st.text_input(
                "密码", placeholder="请输入您的密码", type="password", key="login_password", autocomplete="current-password"
            )
            submitted = st.form_submit_button("登录", use_container_width=True, type="primary")

        if submitted:
            status, user = auth_backend.login(username, password)
            if status == "ok" and user is not None:
                st.session_state["auth"] = auth_backend.snapshot(user)
                st.session_state.pop("_bootstrap_password", None)
                st.rerun()
            elif status == "locked":
                remaining = int(auth_backend.locked_seconds(username)) + 1
                st.error(f"尝试次数过多，请 {remaining} 秒后再试。")
            elif status == "inactive":
                st.error("该账号已被停用，请联系管理员。")
            elif status == "empty":
                st.warning("请输入用户名和密码。")
            else:
                st.error("用户名或密码不正确。")

        left_help, right_help = st.columns(2)
        with left_help:
            with st.expander("忘记密码？"):
                st.caption("请联系管理员在「管理 → 用户与权限」中重置密码。")
        with right_help:
            with st.expander("注册账号"):
                st.caption("账号由管理员统一分配，请联系课题组或平台管理员申请开通。")
        st.markdown('<div class="foot">专注于文献，连接每一个研究灵感<br>论文问答 · 文献检索 · 引用分析</div>', unsafe_allow_html=True)