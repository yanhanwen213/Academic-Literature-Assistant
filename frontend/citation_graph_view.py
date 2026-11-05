"""Readable citation graph viewport; presentation only."""
import html
import json
import textwrap
from collections import defaultdict, deque

import streamlit.components.v1 as components


def render_citation_graph(snapshot):
    nodes = snapshot.get('nodes') or []
    if not nodes:
        return
    center = next((n['id'] for n in nodes if n.get('is_center')), nodes[0]['id'])
    by_id = {n['id']: n for n in nodes}
    links = defaultdict(list)
    for edge in snapshot.get('edges', []):
        a, b = edge.get('source'), edge.get('target')
        if a in by_id and b in by_id:
            links[a].append((b, 1))
            links[b].append((a, -1))
    ranks = {center: 0}
    queue = deque([center])
    while queue:
        a = queue.popleft()
        for b, direction in links[a]:
            if b not in ranks:
                ranks[b] = ranks[a] + direction
                if ranks[b] == 0:
                    ranks[b] = direction
                queue.append(b)
    groups = defaultdict(list)
    for n in nodes:
        groups[ranks.get(n['id'], 2)].append(n)
    low, high = min(groups), max(groups)
    height = max(500, max(map(len, groups.values())) * 110 + 80)
    width = max(960, (high - low + 1) * 330 + 100)
    positions = {}
    for rank, group in sorted(groups.items()):
        for i, n in enumerate(group):
            positions[n['id']] = (60 + (rank-low)*330, height/2 + (i-(len(group)-1)/2)*110 - 38)
    parts = ['<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#aa98b9"/></marker></defs>']
    for edge in snapshot.get('edges', []):
        a, b = edge.get('source'), edge.get('target')
        if a not in positions or b not in positions:
            continue
        x1,y1=positions[a];x2,y2=positions[b]
        if x2>x1:
            x1+=250
        elif x2<x1:
            x2+=250
        else:
            x1+=250;x2+=250
        y1+=38;y2+=38
        mid=(x1+x2)/2 if x1!=x2 else x1+65
        parts.append(f'<path d="M{x1} {y1}C{mid} {y1},{mid} {y2},{x2} {y2}" fill="none" stroke="#b6a8c3" stroke-width="1.5" marker-end="url(#arrow)"/>')
    for n in nodes:
        x,y=positions[n['id']]
        fill,stroke=('#e9dcf7','#9370ae') if n.get('is_center') else (('#e6f1ed','#88ac9e') if n.get('in_library') else (('#f0eef4','#b5adbf') if n.get('local_ref') else ('#fff','#d3c5df')))
        title=str(n.get('title') or n['id'])
        label=textwrap.wrap(title,width=28)[:2] or [title]
        if len(title)>56:
            label[-1]=label[-1][:26]+'…'
        parts.append(f'<g class="paper" transform="translate({x},{y})"><title>{html.escape(title)}</title><rect width="250" height="76" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        for i,line in enumerate(label):
            parts.append(f'<text x="125" y="{24+i*19}" text-anchor="middle" fill="#4f3e60" font-size="14">{html.escape(line)}</text>')
        parts.append(f'<text x="125" y="64" text-anchor="middle" fill="#8b7898" font-size="12">{html.escape(str(n.get("year") or ""))}</text></g>')
    cx,cy=positions[center]
    config=json.dumps(dict(width=width,height=height,cx=cx+125,cy=cy+38))
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>
    *{box-sizing:border-box}body{margin:0;font:13px "Microsoft YaHei",sans-serif;color:#6a527c;background:#fcfaff} .bar{display:flex;align-items:center;gap:8px;padding:10px 14px;border-bottom:1px solid #e7dcef;background:#faf5ff;flex-wrap:wrap}button{border:1px solid #d8c5e7;background:white;color:#71528a;padding:7px 13px;border-radius:8px;cursor:pointer}button:hover{background:#eee2f8}button:focus-visible{outline:2px solid #a17cba}#hint{margin-left:auto;font-size:12px;color:#8e7a9e}#viewport{width:100%;height:530px;overflow:hidden;touch-action:none;cursor:grab}#viewport:active{cursor:grabbing}svg{width:100%;height:100%;display:block}.paper:hover rect{stroke:#9470b0;stroke-width:2.5}
    </style><div class="bar"><button id="minus" aria-label="缩小">−</button><output id="scale">100%</output><button id="plus" aria-label="放大">＋</button><button id="readable">清晰阅读</button><button id="fit">查看全图</button><span id="hint">滚轮缩放 · 拖动平移 · 悬停查看完整标题</span></div><div id="viewport"><svg role="img" aria-label="可缩放引用关系图"><g id="scene">PARTS</g></svg></div><script>
    const cfg=CONFIG, viewport=document.getElementById('viewport'), scene=document.getElementById('scene');let zoom=1,x=0,y=0,drag=null;
    function draw(){scene.setAttribute('transform',`translate(${x},${y}) scale(${zoom})`);document.getElementById('scale').textContent=Math.round(zoom*100)+'%';}
    function readable(){zoom=1;x=viewport.clientWidth/2-cfg.cx;y=viewport.clientHeight/2-cfg.cy;draw();}
    function change(next,px=viewport.clientWidth/2,py=viewport.clientHeight/2){next=Math.max(.08,Math.min(3,next));x=px-(px-x)*next/zoom;y=py-(py-y)*next/zoom;zoom=next;draw();}
    document.getElementById('plus').onclick=()=>change(zoom*1.25);document.getElementById('minus').onclick=()=>change(zoom/1.25);document.getElementById('readable').onclick=readable;
    document.getElementById('fit').onclick=()=>{zoom=Math.min((viewport.clientWidth-40)/cfg.width,(viewport.clientHeight-40)/cfg.height,1);x=(viewport.clientWidth-cfg.width*zoom)/2;y=(viewport.clientHeight-cfg.height*zoom)/2;draw();};
    viewport.addEventListener('wheel',e=>{e.preventDefault();const r=viewport.getBoundingClientRect();change(zoom*Math.exp(-e.deltaY*.0015),e.clientX-r.left,e.clientY-r.top);},{passive:false});
    viewport.addEventListener('pointerdown',e=>{drag={px:e.clientX,py:e.clientY,x,y};viewport.setPointerCapture(e.pointerId);});viewport.addEventListener('pointermove',e=>{if(drag){x=drag.x+e.clientX-drag.px;y=drag.y+e.clientY-drag.py;draw();}});viewport.addEventListener('pointerup',()=>drag=null);viewport.addEventListener('pointercancel',()=>drag=null);window.addEventListener('resize',readable);readable();
    </script></html>'''.replace('PARTS',''.join(parts)).replace('CONFIG',config)
    components.html(document,height=605,scrolling=False)
