"""Shared, decorative galaxy orbit for the application title."""
import math
import random
from functools import lru_cache


@lru_cache(maxsize=1)
def _orbit_artwork() -> str:
    """Return local SVG/CSS artwork; no scripts, network or application state."""
    rng = random.Random(731)
    stars = []
    for index in range(780):
        radius = 104 + rng.random() ** .7 * 81
        arm = index % 3
        angle = arm * math.tau / 3 + radius * .035 + rng.gauss(0, .16)
        x = 200 + math.cos(angle) * radius
        y = 200 + math.sin(angle) * radius
        size = rng.uniform(.5, 1.4)
        color = ('#9770cb', '#ba8fd9', '#d6a4d8', '#7f66b8')[index % 4]
        stars.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" '
                     f'fill="{color}" opacity="{rng.uniform(.3, .85):.2f}"/>')
    return '''<style>
    .literature-orbit-title {position:relative;display:inline-block;isolation:isolate;white-space:nowrap;margin-left:0;padding:38px 72px;}
    .literature-orbit-title .orbit-label {position:relative;z-index:2;text-shadow:0 0 8px #fff,0 0 16px #fff9;}
    .literature-orbit-title .orbit-plane {position:absolute;left:50%;top:52%;width:calc(100% + 8px);height:280px;transform:translate(-50%,-50%) rotate(-7deg) scaleY(.46);pointer-events:none;z-index:1;}
    .literature-orbit-title svg {display:block;width:100%;height:100%;overflow:visible;filter:drop-shadow(0 0 2px #b69ad955);}
    .literature-orbit-title .orbit-galaxy {transform-origin:200px 200px;animation:literature-binary-orbits 28s linear infinite;}
    .literature-orbit-title .orbit-dust {fill:none;stroke:#b99bd2;stroke-width:7;opacity:.12;filter:blur(3px);}
    .literature-orbit-title .orbit-glints {transform-origin:200px 200px;animation:literature-binary-orbits 20s linear infinite reverse;}
    .literature-orbit-title .orbit-glints circle {fill:#b48ad2;animation:literature-star-twinkle 3.4s ease-in-out infinite alternate;}
    @keyframes literature-binary-orbits {to {transform:rotate(360deg);}}
    @keyframes literature-star-twinkle {from {opacity:.35;}to {opacity:1;}}
    @media(max-width:640px) {.literature-orbit-title {padding:32px 58px;}.literature-orbit-title .orbit-plane {height:250px;}}
    @media(prefers-reduced-motion:reduce) {.literature-orbit-title .orbit-galaxy,.literature-orbit-title .orbit-glints,.literature-orbit-title .orbit-glints circle {animation:none!important;}}
    </style><span class="literature-orbit-title"><span class="orbit-plane" aria-hidden="true"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" focusable="false"><g class="orbit-galaxy"><g class="orbit-dust"><path d="M110 150C60 90 190 15 298 84S370 270 280 321"/><path d="M290 250C340 310 210 385 102 316S30 130 120 79"/></g>''' + ''.join(stars) + '''</g><g class="orbit-glints"><circle cx="54" cy="180" r="2.4"/><circle cx="315" cy="95" r="2"/><circle cx="280" cy="345" r="1.8"/></g></svg></span><span class="orbit-label">文献助手</span></span>'''


def orbit_style() -> str:
    return _orbit_artwork().split("</style>", 1)[0] + "</style>"


def orbit_title() -> str:
    return _orbit_artwork().split("</style>", 1)[1]
