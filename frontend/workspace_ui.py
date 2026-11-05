"""Shared presentation components for the research workspace."""
import streamlit as st

from title_orbits import orbit_title, orbit_style


def render_console_title():
    """Shattered type presentation adapted for the Streamlit sidebar."""
    shapes = (
        ("polygon(0 0,55% 0,42% 48%,0 34%)", -17, -16, -18),
        ("polygon(55% 0,100% 0,100% 40%,42% 48%)", 19, -13, 20),
        ("polygon(0 34%,42% 48%,54% 100%,0 100%)", -15, 17, -12),
        ("polygon(42% 48%,100% 40%,100% 100%,54% 100%)", 18, 18, 16),
    )
    letters = []
    for index, char in enumerate("控制台"):
        fragments = ''.join(
            f'<span class="console-shard" style="--clip:{clip};--dx:{dx}px;'
            f'--dy:{dy}px;--turn:{turn}deg;--lag:{index * 70 + part * 25}ms">{char}</span>'
            for part, (clip, dx, dy, turn) in enumerate(shapes)
        )
        letters.append('<span class="console-letter"><span class="console-base">'
                       + char + '</span>' + fragments + '</span>')
    markup = ("""<style>
    html,body {margin:0;background:transparent;overflow:hidden;}
    .console-shatter-title {
      display:flex;align-items:center;width:max-content;max-width:100%;
      position:relative;margin:0 0 1rem;padding:8px 0;
      font-family:Inter,"Microsoft YaHei",sans-serif;font-size:32px;
      font-weight:750;line-height:1.35;letter-spacing:.035em;color:#604676;
      perspective:600px;outline-offset:5px;border-radius:6px;cursor:pointer;
    }
    .console-letter {position:relative;display:inline-block;}
    .console-base {opacity:0;}
    .console-shard {
      position:absolute;inset:0;clip-path:var(--clip);color:inherit;
      transform-origin:center;backface-visibility:hidden;
      animation:console-shard-assemble 1.15s cubic-bezier(.16,1,.3,1) both;
      animation-delay:var(--lag);transition:transform .65s cubic-bezier(.16,1,.3,1),opacity .4s,color .4s;
      transition-delay:var(--lag);pointer-events:none;
    }
    .console-shatter-title:hover .console-shard,
    .console-shatter-title:focus-visible .console-shard {
      transform:translate3d(var(--dx),var(--dy),14px) rotate(var(--turn));
      color:#9b76b5;opacity:.85;animation:none;
    }
    .console-shatter-title.is-shattered .console-shard {
      animation:console-click-shatter 1.5s cubic-bezier(.2,.7,.2,1) both;
      animation-delay:var(--lag);
    }
    @keyframes console-click-shatter {
      0%,100% {transform:translate3d(0,0,0) rotate(0);opacity:1;}
      35% {transform:translate3d(calc(var(--dx) * 2),calc(var(--dy) * 1.5),55px) rotate(calc(var(--turn) * 2));opacity:.3;}
    }
    .console-shatter-title:focus-visible {outline:2px solid #b291c9;}
    @keyframes console-shard-assemble {
      from {opacity:0;transform:translate3d(var(--dx),var(--dy),45px) rotate(var(--turn));filter:blur(2px);}
      to {opacity:1;transform:translate3d(0,0,0) rotate(0);filter:blur(0);}
    }
    @media(prefers-reduced-motion:reduce) {
      .console-shard {display:none;animation:none;transition:none;}
      .console-base {opacity:1;}
    }
    </style>""" + '<h1 class="console-shatter-title" tabindex="0" aria-label="控制台">'
        + '<span aria-hidden="true" style="display:flex">' + ''.join(letters)
        + '</span></h1>')
    markup += """<script>
    const title = document.querySelector('.console-shatter-title');
    let reset;
    function shatter() {
      if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
      clearTimeout(reset);
      title.classList.remove('is-shattered');
      void title.offsetWidth;
      title.classList.add('is-shattered');
      reset = setTimeout(() => title.classList.remove('is-shattered'), 2000);
    }
    title.addEventListener('click', shatter);
    title.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); shatter(); }
    });
    </script>"""
    import streamlit.components.v1 as components
    with st.sidebar:
        components.html(markup, height=90, scrolling=False)


def render_workspace_style():
    st.markdown("""
<style>
:root {--accent:#82699f;--ink:#302b40;--muted:#8c859b;--line:#e8e4ef;}
[data-testid="stAppViewContainer"] {background:#f8f7fb;color:var(--ink);font-family:Inter,"Microsoft YaHei",sans-serif;}
[data-testid="stHeader"] {background:#f8f7fbdd;}
.stMainBlockContainer {max-width:1440px;padding-top:2.8rem;padding-bottom:3rem;}
h1,h2,h3 {color:var(--ink);letter-spacing:-.025em;}
h1 {font-size:2.1rem!important;} h2 {font-size:1.6rem!important;} h3 {font-size:1.15rem!important;}
[data-testid="stSidebar"] {background:#f0edf5;border-right:1px solid var(--line);}
[data-testid="stSidebar"] .stMainBlockContainer {padding-top:2rem;}
[data-testid="stSidebar"] hr {margin:1.5rem 0;border-color:#ddd7e7;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:6px;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:9px 12px;border-radius:10px;transition:background .2s;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#e8e1f0;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#e3daee;color:#674b87;font-weight:600;}
[data-baseweb="input"],[data-baseweb="select"]>div,[data-baseweb="textarea"] {border-radius:10px!important;border-color:var(--line)!important;background:#fff!important;}
[data-baseweb="input"]:focus-within,[data-baseweb="textarea"]:focus-within {box-shadow:0 0 0 3px #82699f15;}
.stButton button,.stDownloadButton button,[data-testid="stFormSubmitButton"] button {border-radius:9px;border-color:#ded7e8;transition:transform .18s,box-shadow .18s;}
.stButton button:hover,.stDownloadButton button:hover {border-color:var(--accent);color:#71558e;box-shadow:0 4px 12px #4332560b;transform:translateY(-1px);}
button[kind="primary"] {background:#82699f!important;border-color:#82699f!important;color:white!important;}
[data-testid="stExpander"],[data-testid="stForm"],[data-testid="stDataFrame"] {background:white;border:1px solid var(--line);border-radius:13px;overflow:hidden;}
[data-testid="stExpander"] summary {padding-top:12px;padding-bottom:12px;}
[data-testid="stMetric"] {padding:20px;background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 4px 16px #40304f04;}
[data-testid="stMetricValue"] {color:#72578f;}
[data-baseweb="tab-list"] {gap:18px;border-bottom:1px solid var(--line);margin-bottom:16px;}
[data-baseweb="tab"] {background:transparent;padding:10px 6px;}
[data-baseweb="tab-highlight"] {background:#82699f;}
[data-testid="stChatMessage"] {background:white;border:1px solid var(--line);border-radius:16px;padding:22px;margin-bottom:16px;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {background:#f0ebf7;}
[data-testid="stBottom"]>div {background:linear-gradient(transparent,#f8f7fb 22%);}
[data-testid="stChatInput"] {border:1px solid #ded5e9;border-radius:16px;background:white;box-shadow:0 8px 30px #49345f09;}
[data-testid="stAlert"] {border-radius:12px;}
[data-testid="stFileUploader"] {border-radius:12px;}
.workspace-hero {position:relative;overflow:hidden;padding:30px 34px;margin-bottom:22px;border:1px solid #e5deee;border-radius:20px;background:linear-gradient(115deg,#fff 15%,#f0eaf8 100%);}
.workspace-hero .eyebrow {color:#9178ac;font-size:10px;letter-spacing:3px;font-weight:600;margin-bottom:12px;}
.workspace-hero h1 {margin:0;padding:0;font-size:30px!important;}
.workspace-hero p {color:#82758f;font-size:14px;margin:12px 0 0;}
.workspace-hero:after {content:"";position:absolute;right:45px;top:-60px;width:210px;height:210px;border:1px solid #b8a5ce40;border-radius:50%;box-shadow:0 0 0 28px #b8a5ce0d,0 0 0 56px #b8a5ce08;animation:research-float 9s ease-in-out infinite;pointer-events:none;}
.research-welcome {padding:38px 30px;text-align:center;margin:22px 0 28px;background:#ffffffa6;border:1px solid var(--line);border-radius:18px;}
.research-welcome .symbol {font-size:30px;color:#9a82b4;display:inline-block;animation:research-float 6s ease-in-out infinite;}
.research-welcome h3 {margin:14px 0 8px;font-weight:600;}
.research-welcome p {font-size:14px;color:#91869e;line-height:1.9;}
.research-steps {display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin-top:20px;}
.research-steps span {background:#f3eff8;color:#857095;font-size:12px;padding:8px 16px;border-radius:30px;}
@keyframes research-float {0%,100% {transform:translateY(0);}50% {transform:translateY(7px);}}
@media(prefers-reduced-motion:reduce) {*,*:before,*:after {animation:none!important;transition:none!important;}}
@media(max-width:700px) {.workspace-hero {padding:24px;}.workspace-hero h1 {font-size:25px!important;}.workspace-hero:after {right:-140px;}.stMainBlockContainer {padding-top:1.5rem;}}

/* Sidebar navigation: consistent line icons and spacious rows. */
[data-testid="stSidebar"] .st-key-logout_btn {padding:8px 0 26px;margin-bottom:12px;border-bottom:1px solid #e0d7e9;}
[data-testid="stSidebar"] .st-key-logout_btn button {
    min-height:42px;border:1px solid #e6d5df;border-radius:10px;
    background:#fcf6f8;color:#a06c80;box-shadow:none;
}
[data-testid="stSidebar"] .st-key-logout_btn button:hover {background:#f8e9ee;border-color:#d6a8b8;color:#98536c;transform:none;}


/* Lavender animated-gradient navigation, inspired by dexter-st / Uiverse.io. */
[data-testid="stSidebar"] .st-key-workspace_navigation [role="radiogroup"] {
    display:flex;align-items:stretch;gap:16px;padding:12px 5px 18px;
}
[data-testid="stSidebar"] .st-key-workspace_navigation [role="radiogroup"] label {
    --nav-radius:32px;position:relative;isolation:isolate;overflow:hidden;
    display:flex;align-items:center;justify-content:center;
    box-sizing:border-box;width:100%!important;min-width:0;max-width:none!important;
    min-height:50px;margin:0;padding:12px 24px!important;
    border:2px solid #ffffffdd;border-radius:var(--nav-radius);
    background:linear-gradient(120deg,#f7efff,#dfcdf0 52%,#f3e3f5);
    color:#77588e;box-shadow:inset 0 0 10px 3px #ad8ac02b,0 4px 12px #80618d10;
    transition:box-shadow .3s,color .3s;
}
[data-testid="stSidebar"] .st-key-workspace_navigation label > div:first-child {
    position:absolute!important;opacity:0!important;width:1px!important;height:1px!important;
    overflow:hidden;pointer-events:none;
}
[data-testid="stSidebar"] .st-key-workspace_navigation label > div:last-child {
    position:relative;z-index:2;width:100%;margin:0!important;text-align:center;
    text-shadow:0 0 4px #fff;transition:transform .3s;
}
[data-testid="stSidebar"] .st-key-workspace_navigation label p {
    font-size:14px;line-height:22px;font-weight:600;letter-spacing:1px;margin:0;
}
[data-testid="stSidebar"] .st-key-workspace_navigation label::before {
    content:"";position:absolute;pointer-events:none;z-index:-1;
    left:-25%;top:50%;width:150%;aspect-ratio:1;
    background:radial-gradient(ellipse at 65% 180%,#fff,#d1b8e8,#fff,#e9cfe9,#fff,#d6c0eb,#fff,#d1b8e8);
    transform-origin:center;mix-blend-mode:soft-light;
    animation:nav-lavender-rotate 8s linear infinite;
}
[data-testid="stSidebar"] .st-key-workspace_navigation label::after {
    content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;z-index:1;
    background:radial-gradient(ellipse at 50% 30%,#ffffff99,transparent 70%);
    box-shadow:inset 0 -4px 4px #8b6a9d20,inset 0 4px 4px #ffffffa6;
    animation:nav-lavender-pulse 3s ease-in-out infinite;
}
[data-testid="stSidebar"] .st-key-workspace_navigation label:hover {
    color:#68437f;background:linear-gradient(120deg,#f8f1ff,#decaef,#f1dff3);
    box-shadow:inset 0 0 10px 3px #ad8ac02b,0 6px 16px #80618d1c;
}
[data-testid="stSidebar"] .st-key-workspace_navigation label:hover > div:last-child {transform:scale(1.06);}
[data-testid="stSidebar"] .st-key-workspace_navigation label:active > div:last-child {transform:scale(.95);}
[data-testid="stSidebar"] .st-key-workspace_navigation label:has(input:checked) {
    color:#67447f;border-color:#fcf8ff;
    background:linear-gradient(120deg,#f3e8ff,#d7bee9 52%,#eed6ed);
    box-shadow:inset 0 0 10px 3px #a780be3b,0 5px 14px #80618d20;
}
[data-testid="stSidebar"] .st-key-workspace_navigation label:focus-within {outline:2px solid #b89bcd;outline-offset:4px;}
@keyframes nav-lavender-rotate {from {transform:translateY(-50%) rotate(0);}to {transform:translateY(-50%) rotate(360deg);}}
@keyframes nav-lavender-pulse {0%,100% {opacity:1;}50% {opacity:.35;}}
@media(prefers-reduced-motion:reduce) {
    [data-testid="stSidebar"] .st-key-workspace_navigation label::before,
    [data-testid="stSidebar"] .st-key-workspace_navigation label::after {animation:none;}
    [data-testid="stSidebar"] .st-key-workspace_navigation label,
    [data-testid="stSidebar"] .st-key-workspace_navigation label > div:last-child {transition:none;}
}

/* Logout-only treatment adapted from dexter-st / Uiverse.io. */
[data-testid="stSidebar"] .st-key-logout_visual {
    position:relative;padding:14px 17px;margin:4px 0 24px;
    background:transparent;transition:background .35s;border-radius:4px;
}
[data-testid="stSidebar"] .st-key-logout_visual:has(button:hover) {background:#e9ddf580;transition-delay:.8s;}
[data-testid="stSidebar"] .st-key-logout_visual button {
    position:relative;min-height:44px;width:100%;padding:12px 20px;
    background-color:#e2d2f0;background-image:linear-gradient(transparent,#9773ad20);
    border:none;border-radius:30% / 200%;color:#715486;
    box-shadow:0 0 0 1px #9475ad30,0 1px 1px #49305505,0 5px 4px #49305508,0 12px 9px #4930550b,0 20px 15px #4930550e;
    transition:background-color .2s,transform .2s,box-shadow .2s,border-radius .3s;
}
[data-testid="stSidebar"] .st-key-logout_visual button p {font-size:14px;font-weight:600;}
[data-testid="stSidebar"] .st-key-logout_visual button:hover {
    background-color:#fffaff;transform:scale(1.05);border-radius:10% / 200%;color:#715486;
}
[data-testid="stSidebar"] .st-key-logout_visual button:active {
    background-color:#dfcbed;transform:scale(.98);border-radius:20% / 200%;
}
[data-testid="stSidebar"] .st-key-logout_visual::before {
    content:"";position:absolute;inset:0;pointer-events:none;opacity:0;
    background:radial-gradient(circle,#a38ab8 0 3px,transparent 3.2px) left top/6px 6px no-repeat,
      radial-gradient(circle,#a38ab8 0 3px,transparent 3.2px) right top/6px 6px no-repeat,
      radial-gradient(circle,#a38ab8 0 3px,transparent 3.2px) right bottom/6px 6px no-repeat,
      radial-gradient(circle,#a38ab8 0 3px,transparent 3.2px) left bottom/6px 6px no-repeat;
    transform:scale(.6);transition:transform .35s,opacity .35s;
}
[data-testid="stSidebar"] .st-key-logout_visual::after {
    content:"";position:absolute;inset:3px;pointer-events:none;
    border:1px dashed #ad97be;opacity:0;clip-path:inset(0 100% 100% 0);
    transition:clip-path .84s ease .28s,opacity .2s;
}
[data-testid="stSidebar"] .st-key-logout_visual:has(button:hover)::before {opacity:1;transform:scale(1);}
[data-testid="stSidebar"] .st-key-logout_visual:has(button:hover)::after {opacity:.65;clip-path:inset(0);}
[data-testid="stSidebar"] .st-key-logout_visual button:focus-visible {outline:2px solid #ae8cc5;outline-offset:4px;}

/* Lavender treatment for sidebar settings and import controls. */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {color:#685478;font-size:13px;font-weight:500;}
[data-testid="stSidebar"] .react-aria-ComboBox > [role="group"],
[data-testid="stSidebar"] [data-testid="stTextInputRootElement"],
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] {
    min-height:44px;border:1px solid #dcd0e8!important;border-radius:13px!important;
    background:linear-gradient(120deg,#fff,#faf5ff)!important;
    box-shadow:inset 0 1px 0 #fff,0 3px 12px #76568f08;
    transition:border-color .25s,box-shadow .25s,background .25s;
}
[data-testid="stSidebar"] .react-aria-ComboBox > [role="group"]:hover,
[data-testid="stSidebar"] [data-testid="stTextInputRootElement"]:hover,
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover,
[data-testid="stSidebar"] [data-baseweb="input"]:hover {
    border-color:#bba0cf!important;box-shadow:0 5px 16px #87639f15;
}
[data-testid="stSidebar"] .react-aria-ComboBox > [role="group"]:focus-within,
[data-testid="stSidebar"] [data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stSidebar"] [data-baseweb="select"]:focus-within > div,
[data-testid="stSidebar"] [data-baseweb="input"]:focus-within {
    border-color:#a580bf!important;box-shadow:0 0 0 3px #a580bf21,0 5px 18px #87639f12;
}
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] input {background:transparent!important;color:#51415f;}
[data-testid="stSidebar"] input::placeholder {color:#a092ac;font-size:12px;}
[data-testid="stSidebar"] .react-aria-ComboBox svg,
[data-testid="stSidebar"] [data-baseweb="select"] svg {color:#9b7db0;transition:transform .25s;}
[data-testid="stSidebar"] .react-aria-ComboBox:has(input[aria-expanded="true"]) svg,
[data-testid="stSidebar"] [data-baseweb="select"]:focus-within svg {transform:rotate(180deg);}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background:linear-gradient(115deg,#ffffffeb,#f7effcde);border:1px solid #ddd0e9;
    border-radius:14px;box-shadow:0 4px 14px #76568f08;
    transition:border-color .25s,box-shadow .25s;
}
[data-testid="stSidebar"] [data-testid="stExpander"]:hover {
    border-color:#b99bce;box-shadow:0 6px 18px #87639f16;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    position:relative;min-height:46px;padding:12px 14px;color:#685078;
    background:linear-gradient(110deg,#ffffff70,#ecdef53d);transition:background .25s;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary::after {
    content:"";position:absolute;inset:0;pointer-events:none;
    background:linear-gradient(105deg,transparent 25%,#fff9 48%,transparent 72%);
    transform:translateX(-110%);opacity:0;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover::after {animation:sidebar-control-glint .8s ease-out;}
[data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary {
    background:linear-gradient(110deg,#f5eafa,#eadef36b);border-bottom:1px solid #e6d9ef;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {color:#a180b7;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background:linear-gradient(140deg,#ffffffef,#f2e8fae6);border:1px dashed #c0a4d4;
    border-radius:16px;padding:14px;transition:border-color .25s,box-shadow .25s,background .25s;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:focus-within {
    border-color:#976db5;background:linear-gradient(140deg,#fff,#edddf9);
    box-shadow:0 0 0 3px #a580bf16,0 8px 22px #87639f12;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button,
[data-testid="stSidebar"] .stButton:not(.st-key-logout_btn) button {
    border:1px solid #d2bee2;border-radius:11px;color:#75518e;
    background:linear-gradient(120deg,#fff,#f0e4fa);min-height:38px;
    box-shadow:inset 0 1px 0 #fff,0 3px 10px #76568f08;
    transition:background .2s,box-shadow .2s,border-color .2s;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover,
[data-testid="stSidebar"] .stButton:not(.st-key-logout_btn) button:hover {
    background:linear-gradient(120deg,#faf5ff,#e9d8f5);border-color:#ad8bc5;
    color:#684282;box-shadow:0 5px 15px #87639f1c;transform:none;
}
[data-testid="stSidebar"] button:focus-visible {outline:2px solid #b191c9;outline-offset:3px;}
[data-testid="stSidebar"] hr {border-color:#dacbe6;opacity:.8;}
@keyframes sidebar-control-glint {0% {transform:translateX(-110%);opacity:0;}25% {opacity:.7;}100% {transform:translateX(110%);opacity:0;}}
@media(prefers-reduced-motion:reduce) {
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-testid="stExpander"],
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
    [data-testid="stSidebar"] button {transition:none!important;}
    [data-testid="stSidebar"] [data-testid="stExpander"] summary::after {animation:none!important;}
}
</style>
""", unsafe_allow_html=True)


def render_workspace_header():
    st.markdown(orbit_style(), unsafe_allow_html=True)
    st.markdown('''<div class="workspace-hero"><div class="eyebrow">LITERATURE ASSISTANT / RESEARCH WORKSPACE</div><h1>''' + orbit_title() + '''</h1><p>从一篇论文出发，连接更广阔的学术世界。</p></div>''', unsafe_allow_html=True)


def render_chat_welcome():
    st.markdown('''<div class="research-welcome"><span class="symbol" aria-hidden="true">✧</span><h3>让每一次阅读，都成为研究的起点</h3><p>上传文献后，探索核心观点、梳理研究方法，或追溯论据与引用。<br>在下方输入问题，开启你的文献研究。</p><div class="research-steps"><span>01 · 上传与整理</span><span>02 · 提问与精读</span><span>03 · 关联与发现</span></div></div>''', unsafe_allow_html=True)


def render_page_background(page):
    """Page-specific, non-interactive academic scenery."""
    from urllib.parse import quote

    if page == "⚙ 管理后台":
        _render_admin_landscape()
        return

    scenes = {
        "💬 对话": ("#eee5fb", "#f8e4ef", '<path d="M35 70Q35 40 65 40H230Q260 40 260 70V160Q260 190 230 190H115L70 225V190H65Q35 190 35 160Z"/><path d="M80 90H210M80 120H180M80 150H145"/><circle cx="310" cy="245" r="30"/><path d="M298 245H322M310 233V257"/>'),
        "📚 文献库": ("#e9e0f6", "#fae9ef", '<g transform="rotate(-10 180 160)"><rect x="75" y="40" width="175" height="230" rx="15"/><path d="M105 85H210M105 110H190M105 150H220M105 175H220M105 200H180"/></g><path d="M265 95L305 90 335 270 295 275Z"/><path d="M40 300H345"/>'),
        "🕸 引用图谱": ("#e9e2fa", "#f6e0ee", '<path d="M60 160L155 65 285 110 265 260 120 285 60 160 265 260 155 65 120 285M60 160L285 110"/><circle cx="60" cy="160" r="17"/><circle cx="155" cy="65" r="24"/><circle cx="285" cy="110" r="15"/><circle cx="265" cy="260" r="27"/><circle cx="120" cy="285" r="13"/>'),
        "⚙ 管理后台": ("#e9e3f4", "#f4e6ef", '<rect x="40" y="60" width="290" height="220" rx="20"/><path d="M40 110H330M85 155H145M85 185H130M85 220H160"/><path d="M200 235V185M235 235V150M270 235V170"/><circle cx="65" cy="85" r="3"/><circle cx="80" cy="85" r="3"/>'),
    }
    purple, pink, drawing = scenes.get(page, scenes["💬 对话"])
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 340"><g fill="none" stroke="#ab8bc7" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' + drawing + '</g></svg>'
    artwork = quote(svg)
    st.markdown("""<style>
    [data-testid="stAppViewContainer"] {
        background:radial-gradient(ellipse at 95% 12%,PURPLE,transparent 55%),radial-gradient(ellipse at 20% 92%,PINK,transparent 52%),#faf8fd;
    }
    [data-testid="stHeader"] {background:#faf8fdcc;backdrop-filter:blur(12px);}
    [data-testid="stAppViewContainer"]::before {
        content:"";position:fixed;right:1.5vw;bottom:7vh;width:clamp(180px,23vw,360px);height:320px;
        background:url("data:image/svg+xml,ARTWORK") center/contain no-repeat;
        opacity:.12;pointer-events:none;animation:page-art-drift 14s ease-in-out infinite;
    }
    [data-testid="stAppViewContainer"]::after {
        content:"";position:fixed;right:12vw;top:17vh;width:120px;height:120px;
        border:1px solid #bda2d424;border-radius:50%;
        box-shadow:0 0 0 26px #bda2d408,0 0 0 55px #bda2d406;
        pointer-events:none;animation:page-orbit 18s ease-in-out infinite;
    }
    [data-testid="stSidebar"] {background:linear-gradient(165deg,#f0eaf7f5,#f7eef5f5);}
    [data-testid="stMain"] {position:relative;z-index:1;}
    [data-testid="stBottom"]>div {background:linear-gradient(transparent,#faf8fdf5 28%);}
    @keyframes page-art-drift {0%,100% {transform:translateY(0) rotate(-2deg);}50% {transform:translateY(-12px) rotate(2deg);}}
    @keyframes page-orbit {0%,100% {transform:translate(0,0) scale(1);opacity:.5;}50% {transform:translate(-24px,16px) scale(1.08);opacity:1;}}
    @media(max-width:700px) {[data-testid="stAppViewContainer"]::before {width:160px;height:180px;opacity:.07;}[data-testid="stAppViewContainer"]::after {display:none;}}
    @media(prefers-reduced-motion:reduce) {[data-testid="stAppViewContainer"]::before,[data-testid="stAppViewContainer"]::after {animation:none;}}
    </style>""".replace("PURPLE", purple).replace("PINK", pink).replace("ARTWORK", artwork), unsafe_allow_html=True)

    if page == "💬 对话":
        import base64
        from pathlib import Path
        backdrop = base64.b64encode(
            (Path(__file__).parent / "assets" / "chat_lace_background.png").read_bytes()
        ).decode("ascii")
        st.markdown(
            """<style>
            [data-testid="stAppViewContainer"] {
                background-color:#f1e8fa;
                background-image:radial-gradient(ellipse at 8% 55%,#c8a4ec80,transparent 62%),linear-gradient(115deg,#eee1f970,#e8d9f540),url("data:image/png;base64,""" + backdrop + '"' + """);
                background-position:right top;background-size:cover;background-attachment:fixed;
                background-repeat:no-repeat;
            }
            [data-testid="stAppViewContainer"]::before,
            [data-testid="stAppViewContainer"]::after {display:none;}
            [data-testid="stHeader"] {background:transparent;}
            [data-testid="stChatMessage"] {background:#ffffffed;border-color:#e9e1ef;}
            .workspace-hero {background:linear-gradient(110deg,#ffffffed,#ffffff70);}
            .workspace-hero::after {display:none;}
            [data-testid="stBottom"]>div {background:linear-gradient(transparent,#f1e8faf5 28%);}
            @media(max-width:700px) {
                [data-testid="stAppViewContainer"] {background-size:auto 100vh;background-position:right top;}
            }
            </style>""", unsafe_allow_html=True,
        )

        st.markdown("""
        <style>
        .chat-butterfly-scene {
            position:fixed;right:3.5vw;top:82px;width:180px;height:150px;
            perspective:700px;pointer-events:none;z-index:2;
            animation:butterfly-hover 9s ease-in-out infinite;
        }
        .chat-butterfly-scene svg {width:100%;height:100%;overflow:visible;
            transform:rotate(-18deg) rotateX(14deg);
            filter:drop-shadow(5px 12px 8px #70548f25);}
        .butterfly-wing {transform-box:view-box;transform-origin:100px 80px;
            animation:butterfly-left 3.6s ease-in-out infinite;}
        .butterfly-wing.right {animation-name:butterfly-right;}
        @keyframes butterfly-left {0%,100% {transform:rotateY(-14deg);}50% {transform:rotateY(-65deg);}}
        @keyframes butterfly-right {0%,100% {transform:rotateY(14deg);}50% {transform:rotateY(65deg);}}
        @keyframes butterfly-hover {0%,100% {transform:translateY(0) rotate(-3deg);}50% {transform:translateY(-11px) rotate(3deg);}}
        @media(max-width:800px) {.chat-butterfly-scene {width:110px;height:100px;right:12px;top:72px;opacity:.6;}}
        @media(prefers-reduced-motion:reduce) {.chat-butterfly-scene,.butterfly-wing {animation:none!important;}}
        </style>
        <div class="chat-butterfly-scene" aria-hidden="true">
        <svg viewBox="0 0 200 160" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="chat-wing" x1="0" y1="0" x2="1" y2="1">
            <stop stop-color="#fbf5ff"/><stop offset=".3" stop-color="#d5bfe9"/>
            <stop offset=".65" stop-color="#ac87c6"/><stop offset="1" stop-color="#ead6ef"/>
          </linearGradient>
          <radialGradient id="chat-wing-shine"><stop stop-color="#fff" stop-opacity=".85"/><stop offset="1" stop-color="#eee1f7" stop-opacity="0"/></radialGradient>
        </defs>
        <g class="butterfly-wing" opacity=".86">
          <path d="M99 81C79 25 21 2 13 24C6 42 25 67 50 80C8 73 25 121 46 126C67 135 87 104 99 81Z" fill="url(#chat-wing)" stroke="#b79aca" stroke-width="1"/>
          <path d="M97 79L25 30M96 81L38 55M94 84L43 107M64 63L55 39M72 73L42 70M74 96L58 116" stroke="#fff" stroke-opacity=".6" fill="none"/>
          <ellipse cx="45" cy="46" rx="20" ry="15" fill="url(#chat-wing-shine)"/>
          <path d="M26 27L31 33M37 33L42 39M47 40L52 46M40 99L44 104M48 111L52 115" stroke="#fff8" stroke-width="3" stroke-linecap="round"/>
        </g>
        <g class="butterfly-wing right" opacity=".86">
          <path d="M101 81C121 25 179 2 187 24C194 42 175 67 150 80C192 73 175 121 154 126C133 135 113 104 101 81Z" fill="url(#chat-wing)" stroke="#b79aca" stroke-width="1"/>
          <path d="M103 79L175 30M104 81L162 55M106 84L157 107M136 63L145 39M128 73L158 70M126 96L142 116" stroke="#fff" stroke-opacity=".6" fill="none"/>
          <ellipse cx="155" cy="46" rx="20" ry="15" fill="url(#chat-wing-shine)"/>
          <path d="M174 27L169 33M163 33L158 39M153 40L148 46M160 99L156 104M152 111L148 115" stroke="#fff8" stroke-width="3" stroke-linecap="round"/>
        </g>
        <path d="M98 66Q100 58 102 66L103 93L100 113L97 93Z" fill="#9372ab"/>
        <path d="M99 67Q91 49 85 52M101 67Q109 49 115 52" stroke="#9372ab" stroke-width="1.3" fill="none"/>
        </svg></div>
        """, unsafe_allow_html=True)

        left_butterflies = []
        for index in range(3):
            left_butterflies.append(f'''<div class="chat-left-butterfly butterfly-{index}" aria-hidden="true"><svg viewBox="0 0 200 160" focusable="false"><defs><linearGradient id="left-wing-{index}" x2="1" y2="1"><stop stop-color="#fff5ff"/><stop offset=".45" stop-color="#c39be1"/><stop offset="1" stop-color="#9669bd"/></linearGradient></defs><g class="butterfly-wing"><path d="M99 81C79 25 21 2 13 24C6 42 25 67 50 80C8 73 25 121 46 126C67 135 87 104 99 81Z" fill="url(#left-wing-{index})" stroke="#a887c3"/><path d="M97 79L25 30M96 81L38 55M94 84L43 107M64 63L55 39M74 96L58 116" stroke="#fff9" fill="none"/><ellipse cx="43" cy="46" rx="12" ry="9" fill="#ffffff55"/></g><g class="butterfly-wing right"><path d="M101 81C121 25 179 2 187 24C194 42 175 67 150 80C192 73 175 121 154 126C133 135 113 104 101 81Z" fill="url(#left-wing-{index})" stroke="#a887c3"/><path d="M103 79L175 30M104 81L162 55M106 84L157 107M136 63L145 39M126 96L142 116" stroke="#fff9" fill="none"/><ellipse cx="157" cy="46" rx="12" ry="9" fill="#ffffff55"/></g><path d="M98 66Q100 58 102 66L103 93L100 113L97 93Z" fill="#806097"/><path d="M99 67Q91 49 85 52M101 67Q109 49 115 52" stroke="#806097" stroke-width="1.3" fill="none"/></svg></div>''')
        st.markdown("""<style>
        .chat-left-garden {position:fixed;left:calc(300px + 1.5vw);top:35vh;width:155px;height:410px;pointer-events:none;z-index:0;}
        .chat-left-garden::before {content:"";position:absolute;inset:-60px;background:radial-gradient(ellipse,#d4b4ef50,transparent 70%);filter:blur(20px);}
        .chat-left-butterfly {position:absolute;perspective:650px;filter:drop-shadow(3px 7px 7px #84619920);animation:chat-left-flight 13s ease-in-out infinite;}
        .chat-left-butterfly svg {display:block;width:100%;height:100%;overflow:visible;transform:rotate(-24deg);}
        .chat-left-butterfly.butterfly-0 {width:105px;left:6px;top:24px;opacity:.72;}
        .chat-left-butterfly.butterfly-1 {width:65px;left:70px;top:165px;opacity:.57;animation-duration:17s;animation-delay:-6s;}
        .chat-left-butterfly.butterfly-2 {width:48px;left:12px;top:285px;opacity:.45;animation-duration:15s;animation-delay:-10s;}
        .chat-left-butterfly.butterfly-1 svg {transform:rotate(22deg);}
        .chat-left-butterfly .butterfly-wing {animation-duration:2.8s;}
        .chat-left-butterfly.butterfly-1 .butterfly-wing {animation-duration:2.2s;}
        @keyframes chat-left-flight {0%,100% {transform:translate3d(0,0,0) rotate(-5deg);}35% {transform:translate3d(18px,-24px,0) rotate(7deg);}70% {transform:translate3d(-7px,-10px,0) rotate(-9deg);}}
        @media(max-width:1200px) {.chat-left-garden {left:310px;width:100px;opacity:.55;}.chat-left-butterfly.butterfly-0 {width:75px;}.chat-left-butterfly.butterfly-1 {left:34px;width:46px;}.chat-left-butterfly.butterfly-2 {display:none;}}
        @media(max-width:1000px) {.chat-left-garden {display:none;}}
        @media(prefers-reduced-motion:reduce) {.chat-left-garden * {animation:none!important;}}
        </style><div class="chat-left-garden" aria-hidden="true">""" + ''.join(left_butterflies) + '</div>', unsafe_allow_html=True)

    if page == "📚 文献库":
        import base64
        from pathlib import Path
        floral = base64.b64encode(
            (Path(__file__).parent / "assets" / "library_floral_background.jpg").read_bytes()
        ).decode("ascii")
        floral_url = 'url("data:image/jpeg;base64,' + floral + '")'
        st.markdown("""<style>
        [data-testid="stAppViewContainer"] {
            background:linear-gradient(135deg,#faf7ff,#f5eef9 55%,#fcf5fa);
        }
        [data-testid="stAppViewContainer"]::before {
            content:"";display:block;position:fixed;inset:-18px;right:-18px;bottom:-18px;
            width:auto;height:auto;z-index:0;border:0;border-radius:0;
            background-image:linear-gradient(100deg,#faf7fff5 0%,#faf7ffbd 35%,#faf7ff55 100%),FLORAL_URL;
            background-size:cover;background-position:center;
            opacity:.65;transform-origin:90% 20%;pointer-events:none;
            animation:library-bloom-sway 18s ease-in-out infinite;
        }
        [data-testid="stAppViewContainer"]::after {display:none;}
        [data-testid="stHeader"] {background:#faf7ffb3;backdrop-filter:blur(12px);}
        .workspace-hero {
            background:linear-gradient(115deg,#ffffffed,#f5eafaad);
            border-color:#e1d3eb;box-shadow:0 10px 35px #80608f09;
        }
        .workspace-hero::after {display:none;}
        [data-testid="stMain"] [data-testid="stMetric"] {
            position:relative;overflow:hidden;background:linear-gradient(135deg,#fffffff7,#fcf8ffeb);
            border:1px solid #e5d8ee;border-radius:16px;padding:22px 24px;
            box-shadow:0 6px 22px #74578a09;transition:transform .25s,box-shadow .25s;
        }
        [data-testid="stMain"] [data-testid="stMetric"]::after {
            content:"";position:absolute;right:-18px;top:-20px;width:70px;height:70px;
            border-radius:50%;background:#ebdcef55;pointer-events:none;
        }
        [data-testid="stMain"] [data-testid="stMetric"]:hover {
            transform:translateY(-3px);box-shadow:0 10px 25px #74578a12;
        }
        [data-testid="stMain"] [data-testid="stMetricValue"] {color:#87669f;}
        [data-testid="stMain"] [data-testid="stDataFrame"] {
            background:white;border:1px solid #dfd3e9;border-radius:15px;
            box-shadow:0 8px 28px #74578a0c;
        }
        [data-testid="stMain"] [data-testid="stExpander"] {
            background:#fffffff2;border-color:#e4d8ed;border-radius:13px;
        }
        [data-testid="stMain"] [data-baseweb="select"]>div {background:#fffffff5!important;border-color:#e1d4eb!important;}
        [data-testid="stMain"] h3 {color:#5d426f;}
        @keyframes library-bloom-sway {
            0%,100% {transform:translate(0,0) rotate(-.25deg) scale(1.015);}
            50% {transform:translate(-7px,5px) rotate(.25deg) scale(1.025);}
        }
        @media(max-width:700px) {
            [data-testid="stAppViewContainer"]::before {opacity:.4;background-position:75% center;}
            [data-testid="stMain"] [data-testid="stMetric"] {padding:16px;}
        }
        @media(prefers-reduced-motion:reduce) {
            [data-testid="stAppViewContainer"]::before {animation:none;}
            [data-testid="stMain"] [data-testid="stMetric"] {transition:none;}
        }
        </style>""".replace("FLORAL_URL", floral_url), unsafe_allow_html=True)

        flowers = []
        for index, (top, right, size, tilt) in enumerate([
            (0, -8, 207, -23), (16, -35, 259, 17), (34, -4, 184, -16),
            (49, -40, 238, 24), (68, -12, 196, -27), (85, -30, 249, 11),
        ]):
            petals = ''.join('<i style="--petal-angle:' + str(n * 72) + 'deg"></i>' for n in range(5))
            flowers.append(
                '<div class="library-sculpted-flower" style="--flower-top:' + str(top)
                + '%;--flower-right:' + str(right) + 'px;--flower-size:' + str(size)
                + 'px;--flower-tilt:' + str(tilt) + 'deg;--flower-delay:-' + str(index * 1.4)
                + 's">' + petals + '<b></b></div>'
            )
        natural_flowers = base64.b64encode(
            (Path(__file__).parent / "assets" / "library_natural_blossoms.png").read_bytes()
        ).decode("ascii")
        flowers = ['<img class="library-natural-blossoms" src="data:image/png;base64,'
                   + natural_flowers + '" alt="" />']
        drifts = ''.join('<span class="library-drifting-petal" style="--drift-delay:-'
                         + str(n * 2.3) + 's;--drift-left:' + str(15 + (n * 19) % 75)
                         + '%"></span>' for n in range(7))
        st.markdown("""<style>
        .library-flower-border {position:fixed;right:0;top:110px;bottom:25px;
            width:145px;perspective:850px;pointer-events:none;z-index:2;}
        .library-sculpted-flower {position:absolute;top:var(--flower-top);right:var(--flower-right);
            width:var(--flower-size);height:var(--flower-size);transform-style:preserve-3d;
            filter:drop-shadow(5px 9px 5px #71518e30);
            animation:library-flower-turn 8s ease-in-out infinite;animation-delay:var(--flower-delay);}
        .library-sculpted-flower i {position:absolute;left:32%;top:2%;width:38%;height:57%;
            border-radius:58% 48% 52% 48%;transform-origin:50% 85%;
            transform:rotateZ(var(--petal-angle)) rotateX(-24deg) translateZ(9px);
            background:radial-gradient(ellipse at 30% 22%,#fff 3%,#f7efff 30%,#e1cbee 65%,#b79bcf 100%);
            box-shadow:inset 3px 4px 6px #ffffffed,inset -3px -5px 8px #9a73b54a,2px 4px 5px #87629e24;
            border:1px solid #ffffff99;}
        .library-sculpted-flower b {position:absolute;left:43%;top:43%;width:18%;height:18%;
            border-radius:50%;background:radial-gradient(circle at 30% 25%,#fff8dd,#e8c7d9 50%,#b48dbd);
            box-shadow:0 3px 5px #9675a84d;transform:translateZ(18px);}
        .library-drifting-petal {position:absolute;left:var(--drift-left);top:-30px;width:14px;height:22px;
            border-radius:65% 25% 65% 30%;background:linear-gradient(135deg,#fff,#e3cdee 65%,#c6a4d7);
            box-shadow:inset 2px 2px 3px #fff,2px 4px 5px #87629e20;
            animation:library-petal-fall 17s linear infinite;animation-delay:var(--drift-delay);opacity:0;}
        @keyframes library-flower-turn {
            0%,100% {transform:rotateZ(var(--flower-tilt)) rotateY(-14deg) rotateX(8deg) translateY(0);}
            50% {transform:rotateZ(calc(var(--flower-tilt) + 5deg)) rotateY(15deg) rotateX(-7deg) translateY(-6px);}
        }
        @keyframes library-petal-fall {0% {top:0;opacity:0;transform:rotateY(0) rotateZ(0);}
            12%,80% {opacity:.65;}100% {top:100%;opacity:0;transform:translateX(-22px) rotateY(360deg) rotateZ(150deg);}}
        @media(min-width:1200px) {[data-testid="stMain"] .stMainBlockContainer {padding-right:155px;}}
        @media(max-width:1199px) {.library-flower-border {width:80px;opacity:.65;right:-28px;}}
        @media(max-width:700px) {.library-flower-border {display:none;}}
        @media(prefers-reduced-motion:reduce) {.library-sculpted-flower,.library-drifting-petal {animation:none;}.library-drifting-petal {display:none;}}

        /* Larger, less symmetrical petals with restrained natural shading. */
        .library-flower-border {width:350px;top:70px;bottom:0;}
        .library-sculpted-flower {filter:drop-shadow(3px 7px 6px #80648e22);}
        .library-sculpted-flower i {
            border-radius:66% 41% 57% 36% / 46% 61% 40% 55%;
            background:radial-gradient(ellipse at 36% 24%,#fffdfa 8%,#f5edf8 43%,#e5d6ec 77%,#c8b3d3);
            box-shadow:inset 2px 2px 5px #ffffff99,inset -2px -3px 8px #a68cb520,1px 3px 6px #8d729b18;
            border:1px solid #f7f1fa80;
        }
        .library-sculpted-flower i:nth-child(2) {width:41%;height:54%;border-radius:48% 60% 42% 55%;transform:rotateZ(calc(var(--petal-angle) + 5deg)) rotateX(-16deg) translateZ(7px);}
        .library-sculpted-flower i:nth-child(3) {width:35%;height:59%;border-radius:56% 42% 58% 45%;transform:rotateZ(calc(var(--petal-angle) - 4deg)) rotateX(-30deg) translateZ(11px);}
        .library-sculpted-flower i:nth-child(4) {height:53%;top:5%;border-radius:45% 59% 47% 62%;transform:rotateZ(calc(var(--petal-angle) + 3deg)) rotateX(-20deg) translateZ(8px);}
        .library-sculpted-flower i:nth-child(5) {width:40%;height:56%;border-radius:65% 40% 52% 46%;transform:rotateZ(calc(var(--petal-angle) - 3deg)) rotateX(-26deg) translateZ(10px);}
        .library-sculpted-flower b {width:13%;height:13%;left:45%;top:45%;
            background:radial-gradient(circle at 35% 32%,#fff3da,#e7cfc8 60%,#bfa1b7);
            box-shadow:0 2px 4px #9675a82b;}
        .library-drifting-petal {width:18px;height:27px;background:linear-gradient(135deg,#fffafc,#e9dcec 70%,#cdb7d7);border-radius:72% 28% 62% 38%;}
        @media(min-width:1200px) {[data-testid="stMain"] .stMainBlockContainer {padding-right:340px;}}
        @media(max-width:1199px) {.library-flower-border {width:130px;right:-125px;opacity:.5;}}
        @media(max-width:700px) {.library-flower-border {display:none;}}

        .library-natural-blossoms {position:absolute;right:-10px;top:-68%;
            width:960px;height:265%;object-fit:fill;object-position:right center;
            filter:drop-shadow(-5px 9px 7px #79608d29);
            transform-origin:80% 18%;animation:natural-blossom-sway 10s ease-in-out infinite;
            pointer-events:none;}
        @keyframes natural-blossom-sway {
            0%,100% {transform:perspective(900px) rotateY(-5deg) rotateZ(12deg);}
            50% {transform:perspective(900px) rotateY(5deg) rotateZ(14deg) translateY(-5px);}
        }
        @media(max-width:1199px) {.library-natural-blossoms {width:700px;right:-110px;}}
        @media(prefers-reduced-motion:reduce) {.library-natural-blossoms {animation:none;}}


        /* Quiet floral underlay: content is above decoration. */
        .library-flower-border {
            top:0;right:0;bottom:0;width:42vw;z-index:0;
            opacity:.58;overflow:hidden;perspective:1200px;
        }
        .library-natural-blossoms {
            top:-10%;right:-125px;width:640px;height:125%;
            object-fit:fill;transform-origin:80% 80%;
            filter:saturate(.85) contrast(1.08) brightness(1.02) drop-shadow(-3px 6px 5px #76558e20);
            animation:natural-blossom-sway 8s ease-in-out infinite;
        }
        .library-drifting-petal {display:none;}
        @keyframes natural-blossom-sway {
            0%,100% {transform:perspective(1100px) rotateY(-6deg) rotate(-23deg) translateY(0);}
            50% {transform:perspective(1100px) rotateY(7deg) rotate(-19deg) translateY(-12px);}
        }
        [data-testid="stMain"] {position:relative;z-index:1;}
        [data-testid="stSidebar"] {z-index:3;}
        @media(min-width:1200px) {[data-testid="stMain"] .stMainBlockContainer {padding-right:2rem;}}
        @media(max-width:1199px) {
            .library-flower-border {width:40vw;right:0;opacity:.4;}
            .library-natural-blossoms {width:480px;right:-180px;}
        }
        @media(max-width:700px) {.library-flower-border {display:none;}}
        @media(prefers-reduced-motion:reduce) {.library-natural-blossoms {animation:none;transform:rotate(-21deg);}}
        </style><div class="library-flower-border" aria-hidden="true">"""
        + ''.join(flowers) + drifts + '</div>', unsafe_allow_html=True)

    if page in ("🕸 引用图谱", "⚙ 管理后台"):
        is_graph = page == "🕸 引用图谱"
        if is_graph:
            drawing = '<g fill="none" stroke="#ac8cc6" stroke-width="1.2"><path d="M70 110L180 65 300 155 220 270 95 250 70 110 220 270M180 65L220 270M95 250L300 155"/><circle cx="70" cy="110" r="12"/><circle cx="180" cy="65" r="18"/><circle cx="300" cy="155" r="14"/><circle cx="220" cy="270" r="22"/><circle cx="95" cy="250" r="10"/></g><g fill="#bda2d2"><circle cx="180" cy="65" r="5"/><circle cx="220" cy="270" r="6"/></g>'
        else:
            drawing = '<g fill="none" stroke="#ad94c3" stroke-width="1.2"><rect x="50" y="65" width="270" height="210" rx="20"/><path d="M50 110H320M80 140H145M80 165H130M80 190H145"/><rect x="180" y="170" width="22" height="70" rx="5"/><rect x="220" y="145" width="22" height="95" rx="5"/><rect x="260" y="125" width="22" height="115" rx="5"/><circle cx="75" cy="88" r="3"/><circle cx="90" cy="88" r="3"/></g>'
        decor = '<svg viewBox="0 0 370 340" xmlns="http://www.w3.org/2000/svg">' + drawing + '</svg>'
        st.markdown("""<style>
        [data-testid="stAppViewContainer"] {background:radial-gradient(ellipse at 100% 15%,#e8dcf5,transparent 52%),radial-gradient(ellipse at 20% 100%,#f6e5ef,transparent 50%),#f9f6fc;}
        [data-testid="stAppViewContainer"]::before,
        [data-testid="stAppViewContainer"]::after {display:none;}
        .workspace-ambient {position:fixed;right:0;bottom:6vh;width:290px;height:290px;z-index:0;pointer-events:none;opacity:.25;}
        .workspace-ambient svg {width:100%;height:100%;animation:ambient-illustration 12s ease-in-out infinite;}
        .workspace-ambient::before {content:"";position:absolute;inset:-50px;background:radial-gradient(ellipse,#cbb1e22b,transparent 70%);border-radius:50%;}
        .workspace-ambient .ambient-dot {position:absolute;width:5px;height:5px;border-radius:50%;background:#b396c9;animation:ambient-dot 7s ease-in-out infinite;}
        .workspace-ambient .ambient-dot:nth-of-type(1) {top:0;left:30px;}
        .workspace-ambient .ambient-dot:nth-of-type(2) {top:30%;right:15px;animation-delay:-2s;}
        .workspace-ambient .ambient-dot:nth-of-type(3) {bottom:0;left:40%;animation-delay:-4s;}
        [data-testid="stMain"] {position:relative;z-index:1;}
        .workspace-hero {background:linear-gradient(110deg,#ffffffed,#f0e7f8c9);border-color:#e0d4eb;box-shadow:0 10px 32px #80608f08;}
        [data-testid="stMain"] [data-testid="stMetric"] {background:linear-gradient(140deg,#fffffffa,#f7f0fbeF);border-color:#e4d7ee;box-shadow:0 5px 20px #78558e08;transition:box-shadow .25s,transform .25s;}
        [data-testid="stMain"] [data-testid="stMetric"]:hover {transform:translateY(-2px);box-shadow:0 9px 25px #78558e12;}
        [data-testid="stMain"] [data-testid="stExpander"],
        [data-testid="stMain"] [data-testid="stDataFrame"] {background:#fffffff5;border-color:#e2d6ed;border-radius:14px;}
        [data-testid="stMain"] [data-testid="stGraphVizChart"] {background:#fffffff2;border:1px solid #e2d6ed;border-radius:16px;padding:14px;box-shadow:0 8px 28px #78558e09;}
        [data-baseweb="tab-list"] {background:#ffffff80;padding:5px 10px;border-radius:12px;}
        [data-baseweb="tab"][aria-selected="true"] {color:#805c9c;}
        @keyframes ambient-illustration {0%,100% {transform:translateY(0) rotate(-2deg);}50% {transform:translateY(-10px) rotate(2deg);}}
        @keyframes ambient-dot {0%,100% {opacity:.25;transform:translateY(0);}50% {opacity:.8;transform:translateY(-14px);}}
        @media(max-width:800px) {.workspace-ambient {width:180px;height:180px;opacity:.13;}}
        @media(prefers-reduced-motion:reduce) {.workspace-ambient * {animation:none!important;}[data-testid="stMain"] [data-testid="stMetric"] {transition:none;}}
        </style><div class="workspace-ambient" aria-hidden="true">""" + decor
        + '<i class="ambient-dot"></i><i class="ambient-dot"></i><i class="ambient-dot"></i></div>', unsafe_allow_html=True)

    if page == "🕸 引用图谱":
        import base64
        from pathlib import Path
        sky = base64.b64encode((Path(__file__).parent / "assets" / "graph_hydrangea_sky.png").read_bytes()).decode("ascii")
        sky_url = 'url("data:image/png;base64,' + sky + '")'
        petals = ''.join('<i style="--x:' + str(56 + n * 6) + '%;--delay:-' + str(n * 3.1) + 's"></i>' for n in range(7))
        st.markdown("""<style>
        [data-testid="stAppViewContainer"] {
            background-color:#f3eafa;
            background-image:linear-gradient(110deg,#f5edfbf2 0%,#f0e4f9e8 55%,#eee0f5d9 100%),SKY_URL;
            background-size:cover;background-position:right center;background-attachment:fixed;
        }
        .workspace-ambient {display:none;}
        [data-testid="stAppViewContainer"]::before {
            content:"";display:block;position:fixed;inset:0;width:auto;height:auto;
            border:0;border-radius:0;pointer-events:none;z-index:0;opacity:.55;
            background:radial-gradient(ellipse at 15% 70%,#ffffff99,transparent 40%),radial-gradient(ellipse at 60% 30%,#ffffff66,transparent 38%);
            animation:graph-cloud-light 24s ease-in-out infinite;
        }
        [data-testid="stAppViewContainer"]::after {display:none;}
        [data-testid="stHeader"] {background:#f8f5fc70;backdrop-filter:blur(8px);}
        .workspace-hero {background:linear-gradient(110deg,#ffffffed,#f3edfa9e);border-color:#ffffffb3;backdrop-filter:blur(8px);}
        .workspace-hero::after {display:none;}
        [data-testid="stMain"] [data-testid="stGraphVizChart"],
        [data-testid="stMain"] [data-testid="stExpander"],
        [data-testid="stMain"] [data-testid="stDataFrame"],
        [data-testid="stMain"] [data-testid="stMetric"] {background:#fffffff2;border-color:#e2d8ee;box-shadow:0 8px 28px #66538110;}
        [data-testid="stMain"] [data-baseweb="select"]>div {background:#ffffffed!important;}
        .graph-sky-petals {position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;}
        .graph-sky-petals i {position:absolute;left:var(--x);top:-30px;width:9px;height:15px;
            background:linear-gradient(135deg,#fff,#eddef5);border-radius:70% 30% 60% 40%;
            box-shadow:1px 2px 4px #8c71a21a;opacity:0;
            animation:graph-sky-petal 22s linear infinite;animation-delay:var(--delay);}
        @keyframes graph-cloud-light {0%,100% {transform:translateX(-1%);opacity:.4;}50% {transform:translateX(3%);opacity:.7;}}
        @keyframes graph-sky-petal {0% {transform:translate(0,-20px) rotate(0);opacity:0;}12%,80% {opacity:.6;}100% {transform:translate(-110px,105vh) rotate(280deg);opacity:0;}}
        @media(max-width:700px) {[data-testid="stAppViewContainer"] {background-position:75% center;}.graph-sky-petals {opacity:.4;}}
        @media(prefers-reduced-motion:reduce) {[data-testid="stAppViewContainer"]::before {animation:none;}.graph-sky-petals {display:none;}}

        .graph-wind-flowers {position:fixed;right:-12px;top:-12px;bottom:-12px;
            width:38vw;pointer-events:none;z-index:0;overflow:hidden;
            background-image:linear-gradient(110deg,#f0e4f980,#f0e4f93b),SKY_URL;
            background-size:auto 100%;background-position:right center;
            opacity:.52;transform-origin:92% 95%;
            mask-image:linear-gradient(to right,transparent,#000 45%);
            animation:graph-flower-breeze 11s ease-in-out infinite;
        }
        @keyframes graph-flower-breeze {
            0%,100% {transform:perspective(1200px) rotateZ(-.5deg) rotateY(-1deg) translateX(0);}
            25% {transform:perspective(1200px) rotateZ(.6deg) rotateY(1.2deg) translateX(-4px);}
            55% {transform:perspective(1200px) rotateZ(-.8deg) rotateY(-1.4deg) translateX(2px);}
            78% {transform:perspective(1200px) rotateZ(.3deg) rotateY(.8deg) translateX(-2px);}
        }
        @media(max-width:700px) {.graph-wind-flowers {width:45vw;opacity:.3;}}
        @media(prefers-reduced-motion:reduce) {.graph-wind-flowers {animation:none;}}
        </style><div class="graph-wind-flowers" aria-hidden="true"></div><div class="graph-sky-petals" aria-hidden="true">""".replace("SKY_URL", sky_url) + petals + '</div>', unsafe_allow_html=True)


def _render_admin_landscape():
    """Admin-only photographic scenery with gentle CSS motion."""
    import base64
    from pathlib import Path

    scene = base64.b64encode(
        (Path(__file__).parent / "assets" / "admin_lavender_dusk.png").read_bytes()
    ).decode("ascii")
    st.markdown('''<style>
    [data-testid="stAppViewContainer"] {
      background-color:#f2ecf8;
      background-image:linear-gradient(110deg,#f4edfa48,#eee4f630),url("data:image/png;base64,SCENE");
      background-position:center center,center bottom;
      background-size:cover,cover;
      background-attachment:fixed,fixed;
      background-repeat:no-repeat;
      color:#3f3452;
    }
    [data-testid="stAppViewContainer"]::before,[data-testid="stAppViewContainer"]::after {display:none;}
    [data-testid="stHeader"] {background:#f7f2fbbd;backdrop-filter:blur(12px);}
    [data-testid="stSidebar"] {background:linear-gradient(165deg,#f4eefbef,#e7dcf2f5);border-right:1px solid #d8cbe7;}
    [data-testid="stMain"] {position:relative;z-index:1;}
    .workspace-hero {background:linear-gradient(110deg,#fffaffeb,#f1e8f9a8);border-color:#ffffffd9;box-shadow:0 12px 40px #6950810b;backdrop-filter:blur(8px);}
    .workspace-hero::after {background:#fffaf078;border-color:#ffffff80;}
    .workspace-hero h1,[data-testid="stMain"] h3 {color:#58416e;}
    [data-testid="stMain"] [data-testid="stCaptionContainer"] {color:#695778;}
    [data-testid="stMain"] [data-testid="stTabs"] {background:#fcfaffdf;border:1px solid #ffffffd9;border-radius:22px;padding:18px 22px 24px;box-shadow:0 16px 48px #6950810d;}
    [data-baseweb="tab-list"] {gap:8px;padding:6px;background:#eee5f7;border:1px solid #e0d3ed;border-radius:14px;flex-wrap:wrap;}
    [data-baseweb="tab"] {height:auto;min-height:44px;padding:10px 18px;border-radius:10px;color:#6a547e;}
    [data-baseweb="tab"][aria-selected="true"] {background:#fffcff;color:#684989;box-shadow:0 3px 12px #73528e14;font-weight:600;}
    [data-baseweb="tab-highlight"],[data-baseweb="tab-border"] {display:none;}
    [data-testid="stMain"] [data-testid="stMetric"] {min-height:124px;background:linear-gradient(135deg,#fff,#f1e9fa);border-color:#e3d7ed;border-radius:16px;box-shadow:0 6px 20px #73528e08;transition:transform .2s,box-shadow .2s;}
    [data-testid="stMain"] [data-testid="stMetric"]:hover {transform:translateY(-3px);box-shadow:0 10px 24px #73528e17;}
    [data-testid="stMetricValue"] {color:#795799;}
    [data-testid="stMetricLabel"] {color:#71617f;}
    [data-testid="stMain"] [data-testid="stExpander"],[data-testid="stMain"] [data-testid="stForm"],[data-testid="stMain"] [data-testid="stDataFrame"],[data-testid="stMain"] [data-testid="stVegaLiteChart"] {background:#fffcffed;border:1px solid #e2d5ed;border-radius:14px;}
    [data-testid="stMain"] [data-testid="stForm"] {padding:18px;}
    [data-testid="stMain"] button[kind="primary"] {background:#8865a8!important;border-color:#8865a8!important;}
    [data-testid="stMain"] button:focus-visible {outline:3px solid #b79ace;outline-offset:3px;}
    .admin-scenery {position:fixed;inset:0;overflow:hidden;pointer-events:none;z-index:0;}
    .admin-mist {position:absolute;right:-12%;width:85%;height:150px;border-radius:50%;background:radial-gradient(ellipse,#ffffffb0,transparent 68%);animation:admin-mist-drift 32s ease-in-out infinite alternate;}
    .admin-mist.one {top:47%;}
    .admin-mist.two {top:72%;right:20%;opacity:.7;animation-delay:-16s;animation-duration:40s;}
    .admin-birds {display:none;position:absolute;top:24%;right:7%;width:150px;color:#9077a8;opacity:.65;animation:admin-bird-glide 24s ease-in-out infinite alternate;}
    .admin-ripple {display:none;position:absolute;right:10%;bottom:10%;width:220px;height:28px;border:1px solid #fff9;border-radius:50%;animation:admin-water-ripple 8s ease-out infinite;}
    .admin-ripple.second {right:22%;bottom:6%;width:140px;animation-delay:-4s;}
    @keyframes admin-mist-drift {from {transform:translateX(-5%);}to {transform:translateX(9%);}}
    @keyframes admin-bird-glide {from {transform:translate(0,0);}to {transform:translate(-90px,-14px);}}
    @keyframes admin-water-ripple {0% {transform:scale(.6);opacity:0;}30% {opacity:.7;}100% {transform:scale(1.5);opacity:0;}}
    @media(max-width:700px) {[data-testid="stMain"] [data-testid="stTabs"] {padding:10px;border-radius:16px;}[data-baseweb="tab-list"] {gap:4px;}[data-baseweb="tab"] {padding:8px 10px;font-size:13px;}.admin-birds {width:100px;top:18%;}.admin-mist {width:120%;}}
    @media(prefers-reduced-motion:reduce) {.admin-scenery * {animation:none!important;}[data-testid="stMain"] [data-testid="stMetric"] {transition:none;}}
    </style><div class="admin-scenery" aria-hidden="true"><div class="admin-mist one"></div><div class="admin-mist two"></div><svg class="admin-birds" viewBox="0 0 180 70" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M15 40q12-12 24 0 12-12 24 0M85 20q9-9 18 0 9-9 18 0M130 52q7-7 14 0 7-7 14 0"/></svg><div class="admin-ripple"></div><div class="admin-ripple second"></div></div>'''.replace('SCENE', scene), unsafe_allow_html=True)
