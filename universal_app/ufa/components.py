"""Design system and reusable components for the Universal Forecasting interface.

One monochrome palette (black, near-black, charcoal, grays, white), one sans-serif type family, one set of radii and
spacings. Colour is reserved for direction on charts: green for a value above its stated comparison baseline, red for
below. Every piece of text that originates from an upload, the engine or the explanation layer goes through
viewmodels.safe()/safe_md() before it is placed in HTML or Markdown.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from . import viewmodels as VM

FONT = "'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif"
C = {"bg": "#000000", "panel": "#0d0d0e", "panel2": "#141415", "border": "#2a2a2c", "text": "#f5f5f5", "muted": "#a8a8ad", "accent": "#ffffff", "line": "#d4d4d8",
     "dim": "#6b6b70", "pos": "#22c55e", "neg": "#ef4444"}
# kept for callers that list the classic four workflow steps (the six-page navigation lives in the app)
STEPS = [("upload", "1", "Upload"), ("configure", "2", "Inspect and configure"), ("processing", "3", "Validate and run"), ("results", "4", "Results and explanation")]

CSS = f"""
<style>
html, body, .stApp {{ font-family: {FONT}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp td, .stApp th, .stApp small,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"], .stApp [data-baseweb="select"] div, .stApp [role="option"], .stApp [role="tab"] {{ font-family: {FONT} !important; }}
body {{ background: {C['bg']}; }}
.stApp {{ background: transparent; color: {C['text']}; }}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"], [data-testid="stBottomBlockContainer"] {{ background: transparent; }}
.block-container {{ padding-top: 1.1rem; max-width: 1240px; position: relative; z-index: 1; }}
h1, h2, h3, h4, p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {{ color: {C['text']}; }}
h2, h3 {{ font-weight: 650; letter-spacing: -.01em; }}
[data-testid="stCaptionContainer"], .ufa-muted {{ color: {C['muted']} !important; }}
/* ---- navigation */
.ufa-nav-note {{ color: {C['dim']}; font-size: .76rem; margin: -.3rem 0 1.1rem 0; }}
.stApp .stButton > button {{ border-radius: 10px; border: 1px solid {C['border']}; background: {C['panel']}; color: {C['text']}; font-weight: 550; transition: border-color .15s ease, background .15s ease; }}
.stApp .stButton > button:hover {{ border-color: #6b6b70; background: {C['panel2']}; color: {C['text']}; }}
.stApp .stButton > button[kind="primary"], .stApp .stDownloadButton > button[kind="primary"] {{ background: {C['text']}; color: #000; border-color: {C['text']}; }}
.stApp .stButton > button[kind="primary"] p, .stApp .stDownloadButton > button[kind="primary"] p {{ color: #000; }}
.stApp .stButton > button[kind="primary"]:hover {{ background: #d4d4d8; border-color: #d4d4d8; color: #000; }}
.stApp .stButton > button:disabled {{ opacity: .38; }}
.stApp .stDownloadButton > button {{ border-radius: 10px; border: 1px solid {C['border']}; background: {C['panel']}; color: {C['text']}; font-weight: 550; }}
/* ---- surfaces */
.ufa-card, .ufa-kpi, .ufx-panel {{ background: rgba(13,13,14,.88); border: 1px solid {C['border']}; border-radius: 14px; }}
.ufa-card {{ padding: 1rem 1.1rem; margin-bottom: .7rem; }}
.ufa-card h4 {{ margin: 0 0 .35rem 0; font-size: .98rem; }} .ufa-card p {{ margin: 0; color: {C['muted']}; font-size: .9rem; }}
.ufa-kpi {{ padding: .9rem 1rem; margin-bottom: .6rem; min-height: 5.4rem; }}
.ufa-kpi .k {{ font-size: .7rem; color: {C['muted']}; text-transform: uppercase; letter-spacing: .08em; }}
.ufa-kpi .v {{ font-size: 1.28rem; font-weight: 650; overflow-wrap: anywhere; letter-spacing: -.01em; }}
.ufa-kpi .s {{ font-size: .76rem; color: {C['muted']}; overflow-wrap: anywhere; }}
.ufa-banner {{ border: 1px solid {C['text']}; background: #111; color: {C['text']}; border-radius: 12px; padding: .6rem .9rem; margin-bottom: 1rem; font-size: .9rem; font-weight: 600; }}
.ufa-badge {{ display: inline-block; padding: .1rem .5rem; border-radius: 6px; border: 1px solid {C['border']}; font-size: .72rem; color: {C['muted']}; margin-right: .35rem; }}
.ufa-badge.inferred {{ border-style: dashed; }} .ufa-badge.confirmed, .ufa-badge.engine, .ufa-badge.words {{ border-color: #6b6b70; color: {C['text']}; }}
.ufa-field-error {{ color: {C['text']}; font-size: .84rem; margin: -.4rem 0 .6rem 0; padding-left: .6rem; border-left: 3px solid {C['text']}; }}
.ufa-empty {{ border: 1px dashed {C['border']}; border-radius: 14px; padding: 1.6rem; text-align: center; color: {C['muted']}; background: rgba(13,13,14,.82); }}
.ufa-section {{ border-top: 1px solid {C['border']}; margin-top: 1.1rem; padding-top: .9rem; }}
.ufa-hint {{ color: {C['muted']}; font-size: .8rem; margin: -.45rem 0 .7rem 0; }} .ufa-hint b {{ color: {C['text']}; font-weight: 600; }}
[data-testid="stFileUploaderDropzone"] {{ background: rgba(13,13,14,.9); border: 1.5px dashed #6b6b70; border-radius: 14px; }}
[data-testid="stExpander"] details {{ background: rgba(13,13,14,.88); border: 1px solid {C['border']}; border-radius: 12px; }}
[data-testid="stAlert"] {{ border-radius: 12px; background: rgba(20,20,21,.94) !important; border: 1px solid {C['border']}; border-left: 3px solid {C['text']}; }}
[data-testid="stAlert"] * {{ color: {C['text']} !important; }}
[data-testid="stVegaLiteChart"], [data-testid="stDataFrame"] {{ border-radius: 12px; }}
button:focus-visible, input:focus-visible, [role="combobox"]:focus-visible, a:focus-visible {{ outline: 2px solid {C['text']} !important; outline-offset: 2px; }}
/* ---- upload page: one centred column */
.ufa-center {{ text-align: center; max-width: 40rem; margin: 2.2rem auto .6rem auto; }}
.ufa-center h1 {{ font-size: 2rem; margin: 0 0 .5rem 0; letter-spacing: -.02em; }} .ufa-center p {{ color: {C['muted']}; margin: 0; }}
.ufa-center-note {{ text-align: center; color: {C['muted']}; font-size: .8rem; }}
/* ---- result panels */
.ufx-panel {{ padding: 1rem 1.1rem; margin-bottom: .8rem; }}
.ufx-panel h4 {{ margin: 0 0 .7rem 0; font-size: .74rem; text-transform: uppercase; letter-spacing: .1em; color: {C['muted']}; font-weight: 600; }}
.ufx-kv {{ display: flex; justify-content: space-between; gap: .8rem; padding: .5rem 0; border-top: 1px solid #1c1c1e; font-size: .88rem; }}
.ufx-kv:first-of-type {{ border-top: 0; }} .ufx-kv span {{ color: {C['muted']}; }} .ufx-kv b {{ color: {C['text']}; font-weight: 600; text-align: right; overflow-wrap: anywhere; }}
.ufx-kv b small {{ color: {C['muted']}; font-weight: 400; }}
.ufx-up {{ color: {C['pos']} !important; }} .ufx-down {{ color: {C['neg']} !important; }}
.ufx-row {{ display: grid; grid-template-columns: 7.2rem 1fr 5.4rem; gap: .6rem; align-items: center; font-size: .84rem; padding: .32rem 0; }}
.ufx-row .nm {{ color: {C['muted']}; }} .ufx-row .val {{ text-align: right; color: {C['text']}; font-variant-numeric: tabular-nums; }}
.ufx-row .tr {{ height: .5rem; border-radius: 999px; background: #1c1c1e; overflow: hidden; }}
.ufx-row .tr i {{ display: block; height: 100%; border-radius: 999px; background: #55555a; }}
.ufx-row.sel .tr i {{ background: {C['text']}; }} .ufx-row.sel .nm {{ color: {C['text']}; font-weight: 600; }}
.ufx-grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: .6rem; }}
.ufx-mini {{ border: 1px solid #1c1c1e; border-radius: 10px; padding: .6rem .7rem; background: #0a0a0b; }}
.ufx-mini .k {{ font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; color: {C['muted']}; }} .ufx-mini .v {{ font-size: 1.02rem; font-weight: 650; color: {C['text']}; overflow-wrap: anywhere; }}
.ufx-li {{ display: flex; gap: .55rem; align-items: flex-start; padding: .42rem 0; border-top: 1px solid #1c1c1e; font-size: .84rem; color: {C['line']}; }}
.ufx-li:first-of-type {{ border-top: 0; }}
.ufx-tag {{ flex: none; font-size: .64rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; padding: .14rem .45rem; border-radius: 6px; border: 1px solid #55555a; color: {C['muted']}; }}
.ufx-tag.error {{ border-color: {C['text']}; color: {C['text']}; }}
.ufx-basis {{ color: {C['dim']}; font-size: .76rem; margin-bottom: .7rem; }}
.ufx-table {{ width: 100%; border-collapse: collapse; font-size: .86rem; }}
.ufx-table th {{ text-align: left; font-weight: 600; color: {C['muted']}; font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; padding: .5rem .6rem; border-bottom: 1px solid {C['border']}; }}
.ufx-table td {{ padding: .55rem .6rem; border-bottom: 1px solid #1c1c1e; color: {C['text']}; vertical-align: top; overflow-wrap: anywhere; }}
.ufx-table td small {{ color: {C['muted']}; }}
/* ---- loading panel (grayscale; white line) */
.ufx-load {{ perspective: 1400px; margin: .4rem 0 0 0; }}
.ufx-load-card {{ position: relative; border-radius: 18px 18px 0 0; overflow: hidden; border: 1px solid {C['border']}; border-bottom: 0; padding: 1.3rem 1.5rem .2rem 1.5rem;
  background: radial-gradient(120% 90% at 85% 0%, #1b1b1d 0%, #0a0a0b 48%, #000 100%); transform: rotateX(2.2deg); transform-origin: 50% 100%; transform-style: preserve-3d; }}
.ufx-load-card::after {{ content: ""; position: absolute; inset: 0; pointer-events: none; background: linear-gradient(105deg, transparent 35%, #ffffff10 50%, transparent 65%); background-size: 250% 100%; }}
.ufx-load-fig {{ font-size: 2.6rem; font-weight: 300; line-height: 1.05; color: {C['text']}; letter-spacing: -.02em; }}
.ufx-load-fig small {{ font-size: .95rem; font-weight: 400; color: {C['muted']}; margin-left: .5rem; letter-spacing: 0; }}
.ufx-load-cap {{ color: {C['muted']}; font-size: .88rem; margin-top: .2rem; overflow-wrap: anywhere; }}
.ufx-spark {{ display: block; width: 100%; height: 190px; margin-top: .3rem; overflow: visible; }}
.ufx-spark .area {{ fill: url(#ufxg); opacity: .5; }} .ufx-spark .base {{ fill: none; stroke: #2a2a2c; stroke-width: 2; }}
.ufx-spark .line {{ fill: none; stroke: #fff; stroke-width: 2.6; stroke-linecap: round; stroke-linejoin: round; filter: drop-shadow(0 0 5px #fff) drop-shadow(0 0 16px #ffffff66); }}
.ufx-load-note {{ color: {C['dim']}; font-size: .74rem; padding: .1rem 0 .7rem 0; }}
.ufx-band {{ display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; padding: .85rem 1.5rem; margin-top: -1rem; border-radius: 0 0 18px 18px; color: #000; background: linear-gradient(90deg, #d4d4d8, #fff); }}
.ufx-band.solo {{ margin-top: .2rem; border-radius: 14px; }}
.ufx-band .pct {{ font-size: 2.5rem; font-weight: 750; line-height: 1; font-variant-numeric: tabular-nums; min-width: 5.2rem; }} .ufx-band .pct span {{ font-size: 1.3rem; margin-left: .1rem; }}
.ufx-band .txt {{ flex: 1 1 14rem; display: flex; flex-direction: column; min-width: 0; }}
.ufx-band .txt b {{ font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; opacity: .7; }}
.ufx-band .txt span {{ font-size: 1.02rem; font-weight: 650; }} .ufx-band .txt em {{ font-style: normal; font-size: .84rem; opacity: .8; min-height: 1.1rem; overflow-wrap: anywhere; }}
.ufx-band .pill {{ padding: .38rem .9rem; border-radius: 8px; background: #000; color: #fff; font-weight: 600; font-size: .86rem; max-width: 14rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.ufx-band.failed {{ background: #1c1c1e; color: {C['text']}; border: 1px solid {C['text']}; }} .ufx-band.failed .pill {{ background: {C['text']}; color: #000; }}
.ufx-bar {{ height: .38rem; border-radius: 999px; background: #1c1c1e; overflow: hidden; margin: .8rem 0 .35rem 0; }}
.ufx-bar i {{ display: block; height: 100%; width: var(--to); border-radius: 999px; background: #fff; position: relative; }} .ufx-bar.failed i {{ background: #6b6b70; }}
.ufx-stages {{ list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: .5rem; }}
.ufx-stages li {{ display: flex; gap: .6rem; align-items: flex-start; padding: .6rem .75rem; border: 1px solid #1c1c1e; border-radius: 10px; background: rgba(10,10,11,.9); color: {C['dim']}; font-size: .84rem; }}
.ufx-stages li .d {{ flex: none; width: 1.25rem; height: 1.25rem; border-radius: 50%; border: 1px solid #3a3a3d; display: grid; place-items: center; font-size: .7rem; }}
.ufx-stages li.done {{ color: {C['line']}; }} .ufx-stages li.done .d {{ background: #fff; border-color: #fff; color: #000; }}
.ufx-stages li.running {{ color: {C['text']}; border-color: #6b6b70; }} .ufx-stages li.running .d {{ border-color: #fff; }}
.ufx-stages li.failed {{ color: {C['text']}; border-color: {C['text']}; }} .ufx-stages li.failed .d {{ border-color: #fff; color: #fff; }}
.ufx-stages li small {{ display: block; color: {C['dim']}; font-size: .74rem; }}
@media (max-width: 760px) {{ .block-container {{ padding-left: .9rem; padding-right: .9rem; }} .ufa-center h1 {{ font-size: 1.6rem; }} .ufx-load-fig {{ font-size: 2rem; }}
  .ufx-band .pct {{ font-size: 2rem; min-width: 4rem; }} .ufx-spark {{ height: 130px; }} .ufx-row {{ grid-template-columns: 5.4rem 1fr 4.6rem; }}
  .st-key-ufa_nav [data-testid="stHorizontalBlock"] {{ flex-flow: row wrap; gap: .4rem; }}
  .st-key-ufa_nav [data-testid="stColumn"] {{ flex: 1 1 calc(50% - .4rem) !important; min-width: calc(50% - .4rem) !important; width: auto !important; }}
  .st-key-ufa_nav button p {{ font-size: .82rem; }} }}
@media (prefers-reduced-motion: no-preference) {{
  .ufa-card, .ufa-kpi {{ transition: border-color .15s ease; }} .ufa-card:hover, .ufa-kpi:hover {{ border-color: #55555a; }}
  .ufx-load.running .ufx-load-card {{ animation: ufx-float 7s ease-in-out infinite; }}
  .ufx-load.running .ufx-load-card::after {{ animation: ufx-sweep 3.2s linear infinite; }}
  .ufx-load.running .ufx-spark .line {{ stroke-dasharray: var(--len); stroke-dashoffset: var(--len); animation: ufx-draw 3.4s cubic-bezier(.6,.05,.3,1) infinite; }}
  .ufx-load.running .ufx-spark .area {{ animation: ufx-breathe 3.4s ease-in-out infinite; }}
  .ufx-bar i {{ animation: ufx-grow .7s cubic-bezier(.2,.8,.2,1) both; }}
  .ufx-bar.running i::after {{ content: ""; position: absolute; inset: 0; background: linear-gradient(90deg, transparent, #00000055, transparent); background-size: 40% 100%; background-repeat: no-repeat; animation: ufx-shimmer 1.3s linear infinite; }}
  .ufx-stages li.running .d {{ animation: ufx-pulse 1.2s ease-in-out infinite; }}
  .ufx-band .pct, .ufx-band .txt span {{ animation: ufx-pop .45s ease-out both; }}
}}
@keyframes ufx-draw {{ 0% {{ stroke-dashoffset: var(--len); opacity: 1; }} 62% {{ stroke-dashoffset: 0; opacity: 1; }} 88% {{ stroke-dashoffset: 0; opacity: 1; }} 100% {{ stroke-dashoffset: 0; opacity: 0; }} }}
@keyframes ufx-breathe {{ 0%, 100% {{ opacity: .15; }} 62%, 88% {{ opacity: .5; }} }}
@keyframes ufx-float {{ 0%, 100% {{ transform: rotateX(2.2deg) rotateY(-1.2deg) translateY(0); }} 50% {{ transform: rotateX(1.2deg) rotateY(1.2deg) translateY(-4px); }} }}
@keyframes ufx-sweep {{ 0% {{ background-position: 130% 0; }} 100% {{ background-position: -130% 0; }} }}
@keyframes ufx-grow {{ from {{ width: var(--from); }} to {{ width: var(--to); }} }}
@keyframes ufx-shimmer {{ 0% {{ background-position: -40% 0; }} 100% {{ background-position: 140% 0; }} }}
@keyframes ufx-pulse {{ 0%, 100% {{ box-shadow: 0 0 0 0 #ffffff66; }} 50% {{ box-shadow: 0 0 0 6px #ffffff00; }} }}
@keyframes ufx-pop {{ from {{ opacity: .35; transform: translateY(3px); }} to {{ opacity: 1; transform: none; }} }}
</style>
"""

# ---------------------------------------------------------------- live Saturn background
# A constant snippet (no user, file or engine text is ever placed in it). It draws one canvas, fixed behind the page, with
# plain WebGL: a lit, slowly rotating sphere with banded, softly textured clouds, a rim light and a faint atmospheric glow;
# a ring system with gaps whose lighting drifts slowly; the ring's shadow on the planet and the planet's shadow on the ring.
# Each page names where the planet sits (CSS variable --ufa-sat, read twice a second) and the planet eases there.
# A bounded parallax follows the pointer (at most ~14 px). No library is loaded and nothing is fetched. pointer-events
# are off, drawing is capped at 30 frames a second and pauses when the tab is hidden, "reduce motion" gets still frames
# (no rotation, no parallax), and without WebGL a CSS planet is shown. The same snippet adds one convenience to the
# page: a class on the upload drop zone while a file is dragged over it (purely visual).
SATURN_HTML = r"""<script>
(function () {
  try {
    var W = window.parent, D = W.document, ID = 'ufa-saturn';
    var reduce = W.matchMedia && W.matchMedia('(prefers-reduced-motion: reduce)').matches;
    // ---- drag-over feedback for the upload zone (visual only; the uploader itself is untouched)
    if (!W.__ufaDragHook) { W.__ufaDragHook = 1; var depth = 0;
      function zone() { return D.querySelector('.st-key-ufa_drop'); }
      D.addEventListener('dragenter', function (e) { var z = zone(); if (z && z.contains(e.target)) { depth++; z.classList.add('ufa-drag'); } }, true);
      D.addEventListener('dragleave', function (e) { var z = zone(); if (z && z.contains(e.target)) { depth = Math.max(0, depth - 1); if (!depth) z.classList.remove('ufa-drag'); } }, true);
      D.addEventListener('drop', function () { depth = 0; var z = zone(); if (z) z.classList.remove('ufa-drag'); }, true); }
    var old = D.getElementById(ID);
    if (old && old.__ufaV === 3) return;                       // already running: keep the same planet across reruns and pages
    if (old) old.remove();
    var c = D.createElement('canvas'); c.id = ID; c.setAttribute('aria-hidden', 'true'); c.__ufaV = 3;
    c.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;z-index:0;pointer-events:none;opacity:.3;transition:opacity 1.4s cubic-bezier(.2,.8,.2,1);';
    D.body.insertBefore(c, D.body.firstChild);
    var gl = c.getContext('webgl', {alpha: true, antialias: true, premultipliedAlpha: true});
    if (!gl) { fallback(); return; }
    var VS = 'attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}';
    var FS = [
      'precision mediump float;uniform vec2 R;uniform float T;uniform vec3 P;uniform vec2 O;',
      'float h(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}',
      'float vn(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(h(i),h(i+vec2(1,0)),f.x),mix(h(i+vec2(0,1)),h(i+vec2(1,1)),f.x),f.y);}',
      'float band(float lat,float lon){float b=.60+.15*sin(lat*9.)+.10*sin(lat*23.+1.3)+.06*sin(lat*47.+lon*2.+.6);',
      ' b+=.05*(vn(vec2(lon*3.,lat*40.))-.5)+.03*(vn(vec2(lon*9.,lat*90.))-.5);',
      ' b+=.07*exp(-pow((lat-.42)*9.,2.))*sin(lon*2.+T*.03);return b;}',
      'float ring(float r){if(r<1.24||r>2.30)return 0.;float d=.55+.25*sin(r*38.)+.12*sin(r*91.+1.)+.08*sin(r*173.);',
      ' d*=smoothstep(1.24,1.30,r)*(1.-smoothstep(2.20,2.30,r));d*=1.-.85*(smoothstep(1.93,1.95,r)-smoothstep(1.99,2.01,r));',
      ' d*=.55+.45*smoothstep(1.24,1.55,r);return clamp(d,0.,1.);}',
      'void main(){float s=min(R.x,R.y)*P.z;vec2 ce=vec2(R.x*P.x,R.y*P.y)+O;vec2 q=(gl_FragCoord.xy-ce)/s;',
      ' float tilt=.47+.03*sin(T*.05),roll=-.36;vec3 n=normalize(vec3(sin(roll)*cos(tilt),cos(roll)*cos(tilt),sin(tilt)));',
      ' vec3 L=normalize(vec3(-.62+.08*sin(T*.021),.38,.69));vec3 u=normalize(cross(vec3(0.,0.,1.),n));vec3 v=cross(n,u);',
      ' vec4 col=vec4(0.);float r2=dot(q,q);float rr0=sqrt(r2);float zs=-9.;',
      ' float glow=exp(-max(rr0-1.,0.)*5.5)*.22*smoothstep(.95,1.02,rr0)*(.55+.45*max(dot(normalize(vec3(q,.35)),L),0.));col=vec4(vec3(glow),glow);',
      ' if(r2<1.){zs=sqrt(1.-r2);vec3 N=vec3(q,zs);float lat=asin(clamp(dot(N,n),-1.,1.));float lon=atan(dot(N,v),dot(N,u))+T*.05;',
      '  float dif=max(dot(N,L),0.);float lit=.03+.97*pow(dif,.8);float g=band(lat,lon)*lit;',
      '  float dn=dot(L,n);if(abs(dn)>.001){float t=-dot(N,n)/dn;if(t>0.){vec3 Q=N+L*t;g*=1.-.62*ring(length(Q));}}',
      '  g+=pow(1.-zs,3.)*.22*(.4+.6*max(dot(vec3(q,0.),L.xyz),0.));g+=pow(max(dot(reflect(-L,N),vec3(0.,0.,1.)),0.),24.)*.10;',
      '  g*=.55+.45*smoothstep(0.,.35,zs);float a=smoothstep(1.,.985,r2);col=vec4(vec3(g)*a+col.rgb*(1.-a),max(a,col.a));}',
      ' float zr=-(n.x*q.x+n.y*q.y)/n.z;vec3 Pp=vec3(q,zr);float rr=length(Pp);float d=ring(rr);',
      ' if(d>0.&&(r2>=1.||zr>zs)){float b=dot(Pp,L);float cc=dot(Pp,Pp)-1.;float disc=b*b-cc;float shd=1.;',
      '  if(disc>0.&&(-b-sqrt(disc))>0.)shd=.12;float ang=atan(dot(Pp,v),dot(Pp,u));',
      '  float g=(.40+.5*d)*shd*(.9+.1*sin(ang*2.-T*.04))*(.92+.08*sin(rr*260.+T*.02));float a=d*.9;col.rgb=col.rgb*(1.-a)+vec3(g)*a;col.a=col.a*(1.-a)+a;}',
      ' gl_FragColor=col;}'].join('\n');
    function sh(t, s) { var o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o); if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error('shader'); return o; }
    var pr = gl.createProgram(); gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(pr);
    if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error('link');
    gl.useProgram(pr);
    var buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    var loc = gl.getAttribLocation(pr, 'p'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    var uR = gl.getUniformLocation(pr, 'R'), uT = gl.getUniformLocation(pr, 'T'), uP = gl.getUniformLocation(pr, 'P'), uO = gl.getUniformLocation(pr, 'O');
    var place = [.93, .86, .13], goal = [.93, .86, .13], off = [0, 0], offGoal = [0, 0], k = 1, t0 = Date.now(), frozenT = 40;
    function readPlace() { try { var v = W.getComputedStyle(D.documentElement).getPropertyValue('--ufa-sat').trim().split(/\s+/).map(parseFloat);
      if (v.length >= 4 && v.every(isFinite)) { var narrow = W.innerWidth < 760; goal = narrow ? [.86, .90, Math.min(v[2], .16)] : v.slice(0, 3);
        c.style.opacity = String(Math.max(0, Math.min(.6, narrow ? Math.min(v[3], .24) : v[3]))); } } catch (e) {} }
    function size() { k = Math.min(W.devicePixelRatio || 1, 1.25); var w = Math.round(W.innerWidth * k), h = Math.round(W.innerHeight * k);
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; gl.viewport(0, 0, w, h); } }
    function draw() { size(); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.uniform2f(uR, c.width, c.height);
      gl.uniform1f(uT, reduce ? frozenT : (Date.now() - t0) / 1000 + 40); gl.uniform3f(uP, place[0], place[1], place[2]); gl.uniform2f(uO, off[0] * k, -off[1] * k); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); }
    function ease() { var moving = false; for (var i = 0; i < 3; i++) { var d = goal[i] - place[i]; if (Math.abs(d) > 1e-4) { place[i] += d * (reduce ? 1 : .045); moving = true; } else place[i] = goal[i]; }
      for (var j = 0; j < 2; j++) { var e = offGoal[j] - off[j]; if (Math.abs(e) > .05) { off[j] += e * .06; moving = true; } } return moving; }
    c.setAttribute('data-mode', reduce ? 'webgl-still' : 'webgl-live');
    c.__ufaDraw = function () { ease(); draw(); }; c.__ufaGl = gl;
    readPlace(); place = goal.slice(); draw();
    var seen = -1;
    W.setInterval(function () { if (c !== D.getElementById(ID)) return; readPlace(); var f = +(c.getAttribute('data-frames') || 0);
      if (reduce || f === seen) { place = goal.slice(); off = offGoal.slice(); draw(); } seen = f; }, 500);   // no animation frames (reduced motion, or a host that pauses them): jump to the place
    if (reduce) { W.addEventListener('resize', draw); return; }
    W.addEventListener('pointermove', function (e) { offGoal = [((e.clientX / W.innerWidth) - .5) * -28, ((e.clientY / W.innerHeight) - .5) * -18]; }, {passive: true});
    var last = 0;
    function loop(ts) { if (c !== D.getElementById(ID)) return; if (!D.hidden && ts - last > 33) { last = ts; ease(); draw(); c.setAttribute('data-frames', (+(c.getAttribute('data-frames') || 0) + 1)); } W.requestAnimationFrame(loop); }
    W.requestAnimationFrame(loop);
    function fallback() { c.remove(); var f = D.createElement('div'); f.id = ID; f.__ufaV = 3; f.setAttribute('aria-hidden', 'true'); f.setAttribute('data-mode', 'css-fallback');
      f.style.cssText = 'position:fixed;right:9vw;top:14vh;width:30vmin;height:30vmin;z-index:0;pointer-events:none;opacity:.4;border-radius:50%;background:radial-gradient(circle at 34% 30%,#d4d4d8 0%,#6b6b70 38%,#141415 72%,#000 100%);box-shadow:0 0 60px rgba(255,255,255,.12);';
      var r = D.createElement('div'); r.style.cssText = 'position:absolute;left:-55%;top:38%;width:210%;height:24%;border-radius:50%;border:5px solid #8a8a8f;border-left-color:#55555a;border-right-color:#55555a;transform:rotate(-20deg);box-sizing:border-box;';
      f.appendChild(r); D.body.insertBefore(f, D.body.firstChild); }
  } catch (e) { try { var x = window.parent.document.getElementById('ufa-saturn'); if (x && x.tagName === 'CANVAS') x.remove(); } catch (e2) {} }
})();
</script>"""


def saturn():
    """Places the Saturn background. The snippet is a constant; see SATURN_HTML."""
    st.iframe(SATURN_HTML, height=1)


PAGE_KEYS = ("upload", "preview", "configure", "processing", "results", "insights", "reports")

# ---------------------------------------------------------------- motion system and premium layer
# One small set of motion tokens used everywhere: --m-fast for hover / press, --m-med for entrances and indicators,
# --m-slow for page-level reveals. Only opacity, transform, clip and colour animate (no layout shift). Everything is
# switched off under prefers-reduced-motion, and nothing waits for an animation before it can be used.
MOTION_CSS = f"""
<style>
:root {{ --m-fast: .16s; --m-med: .34s; --m-slow: .7s; --m-ease: cubic-bezier(.2,.8,.2,1); --m-io: cubic-bezier(.65,0,.35,1);
  --glass: rgba(10,10,11,.72); --hair: rgba(255,255,255,.08); --hair2: rgba(255,255,255,.14); --glow: 0 0 24px rgba(255,255,255,.10); }}
.stApp {{ background: radial-gradient(1200px 700px at 78% 8%, rgba(255,255,255,.045), transparent 60%), radial-gradient(900px 600px at 0% 100%, rgba(255,255,255,.03), transparent 60%); }}
.block-container {{ padding-top: .7rem; max-width: 1320px; padding-left: 2.4rem; padding-right: 2.4rem; }}
@media (min-width: 761px) and (max-width: 1180px) {{ .st-key-ufa_nav .stButton > button span:has(> [data-testid="stIconMaterial"]) {{ display: none; }}
  .st-key-ufa_nav .stButton > button p {{ font-size: .8rem !important; }} }}
.st-key-ufa_nav .stButton > button [data-testid="stMarkdownContainer"] {{ overflow: visible; }}
[data-testid="stAlertContainer"] {{ background: transparent !important; }}   /* alerts stay grayscale: no red/yellow/blue fills */
[data-testid="stAlert"] {{ animation: ufx-in .35s var(--m-ease) both; }}
/* style-only elements take no space */
[data-testid="stElementContainer"]:has(.ufx-css), [data-testid="stElementContainer"]:has(iframe[srcdoc*="ufa-saturn"]),
[data-testid="stElementContainer"]:has(iframe[srcdoc*="scrollTo"]) {{ display: none; }}
/* no grey-out flash while the page reruns */
[data-stale="true"], .stale-element {{ opacity: 1 !important; filter: none !important; transition: none !important; }}
[data-testid="stStatusWidget"] {{ opacity: .55; }}
/* ---- type scale */
.stApp h1 {{ font-weight: 700; letter-spacing: -.035em; }}
.ufx-eyebrow {{ font-size: .72rem; letter-spacing: .22em; text-transform: uppercase; color: {C['muted']}; font-weight: 600; }}
.ufx-h {{ display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin: .2rem 0 1rem 0; }}
.ufx-h h2 {{ margin: 0; font-size: 1.65rem; letter-spacing: -.03em; font-weight: 680; }}
.ufx-h p {{ margin: 0; color: {C['muted']}; font-size: .9rem; max-width: 46rem; }}
/* ---- page transition: each page has its own animation name, so moving between pages always replays it */
{''.join(f'.st-key-ufa_page_{p} {{ animation: ufx-pg-{p} var(--m-slow) var(--m-ease) both; }} @keyframes ufx-pg-{p} {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: none; }} }} ' for p in PAGE_KEYS)}
.ufx-enter {{ animation: ufx-in var(--m-slow) var(--m-ease) both; }}
@keyframes ufx-in {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes ufx-fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes ufx-blur {{ from {{ opacity: 0; filter: blur(6px); transform: translateY(4px); }} to {{ opacity: 1; filter: none; transform: none; }} }}
/* ---- brand */
.ufx-brand {{ display: flex; align-items: center; gap: .55rem; font-size: .78rem; font-weight: 700; letter-spacing: .34em; color: #e5e5e5; margin: .1rem 0 .55rem .2rem;
  animation: ufx-fade .9s var(--m-ease) both; }}
.ufx-brand i {{ position: relative; width: 14px; height: 14px; border-radius: 50%; background: radial-gradient(circle at 35% 30%, #fff, #8a8a8f 55%, #2a2a2c); box-shadow: 0 0 10px rgba(255,255,255,.35); }}
.ufx-brand i::after {{ content: ""; position: absolute; left: -6px; right: -6px; top: 5px; height: 4px; border-radius: 50%; border: 1px solid rgba(255,255,255,.75); transform: rotate(-18deg); }}
/* ---- navigation: glass bar, sliding indicator, icon motion */
.st-key-ufa_nav {{ position: sticky; top: .35rem; z-index: 20; padding: 5px; border: 1px solid var(--hair2); border-radius: 16px; background: var(--glass);
  backdrop-filter: blur(16px) saturate(130%); -webkit-backdrop-filter: blur(16px); box-shadow: 0 14px 44px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.06); margin-bottom: 1.3rem; }}
.st-key-ufa_nav [data-testid="stHorizontalBlock"] {{ gap: 6px !important; position: relative; }}
.st-key-ufa_nav [data-testid="stHorizontalBlock"]::before {{ content: ""; position: absolute; z-index: 0; top: 0; bottom: 0; left: 0; width: calc((100% - 30px) / 6);
  transform: translateX(calc(var(--ufa-nav-i, 0) * (100% + 6px))); border-radius: 11px; border: 1px solid rgba(255,255,255,.34);
  background: linear-gradient(180deg, rgba(255,255,255,.17), rgba(255,255,255,.05)); box-shadow: 0 0 26px rgba(255,255,255,.13), inset 0 1px 0 rgba(255,255,255,.25);
  transition: transform .5s var(--m-ease); }}
.st-key-ufa_nav [data-testid="stHorizontalBlock"]::after {{ content: ""; position: absolute; z-index: 0; bottom: -6px; left: 0; width: calc((100% - 30px) / 6); height: 2px;
  transform: translateX(calc(var(--ufa-nav-i, 0) * (100% + 6px))) scaleX(.36); background: #fff; border-radius: 2px; box-shadow: 0 0 10px #fff; transition: transform .5s var(--m-ease); }}
.st-key-ufa_nav [data-testid="stColumn"] {{ position: relative; z-index: 1; }}
.st-key-ufa_nav .stButton > button {{ height: 44px; border-radius: 11px; border: 1px solid transparent !important; background: transparent !important; color: {C['muted']} !important;
  font-weight: 600; letter-spacing: .01em; box-shadow: none !important; transition: color var(--m-fast) var(--m-ease), background var(--m-fast) var(--m-ease), transform var(--m-fast) var(--m-ease); }}
.st-key-ufa_nav .stButton > button {{ padding: 0 .45rem; }}
.st-key-ufa_nav .stButton > button p {{ color: inherit !important; font-size: .88rem; white-space: nowrap; overflow: visible; text-overflow: clip; }}
.st-key-ufa_nav .stButton > button:hover:not(:disabled) {{ background: rgba(255,255,255,.055) !important; color: {C['text']} !important; }}
.st-key-ufa_nav .stButton > button:active:not(:disabled) {{ transform: scale(.975); }}
.st-key-ufa_nav .stButton > button[kind="primary"] {{ color: #fff !important; background: transparent !important; }}
.st-key-ufa_nav .stButton > button[kind="primary"] p {{ font-weight: 700; }}
.st-key-ufa_nav .stButton > button:disabled {{ opacity: .3; cursor: not-allowed; }}
.st-key-ufa_nav .stButton > button [data-testid="stIconMaterial"] {{ transition: transform var(--m-med) var(--m-ease), opacity var(--m-fast); opacity: .75; }}
.st-key-ufa_nav .stButton > button:hover:not(:disabled) [data-testid="stIconMaterial"] {{ transform: translateY(-2px) scale(1.08); opacity: 1; }}
.st-key-ufa_nav .stButton > button[kind="primary"] [data-testid="stIconMaterial"] {{ opacity: 1; animation: ufx-icon .6s var(--m-ease) both; }}
@keyframes ufx-icon {{ 0% {{ transform: scale(.7); opacity: 0; }} 60% {{ transform: scale(1.12); }} 100% {{ transform: none; opacity: 1; }} }}
/* ---- buttons everywhere */
.stApp .stButton > button, .stApp .stDownloadButton > button {{ transition: border-color var(--m-fast) var(--m-ease), background var(--m-fast) var(--m-ease), transform var(--m-fast) var(--m-ease), box-shadow var(--m-med) var(--m-ease); }}
.stApp .stButton > button:hover:not(:disabled), .stApp .stDownloadButton > button:hover:not(:disabled) {{ box-shadow: var(--glow); }}
.stApp .stButton > button:active:not(:disabled), .stApp .stDownloadButton > button:active:not(:disabled) {{ transform: scale(.98); }}
.stApp .stDownloadButton > button[kind="primary"]:hover {{ background: #d4d4d8; color: #000; }}
/* ---- upload: hero, drop zone, file card */
.ufx-hero {{ position: relative; text-align: center; margin: 1.8rem auto 1.4rem auto; max-width: 52rem; }}
.ufx-hero h1 {{ font-size: clamp(2.1rem, 4.6vw, 3.6rem); line-height: 1.04; margin: .55rem 0 .8rem 0; padding: 0; letter-spacing: -.045em; font-weight: 720; color: #fff;
  text-shadow: 0 0 40px rgba(255,255,255,.18); }}
.ufx-hero h1 [data-heading-text] > span {{ display: block; color: #8a8a8f; text-shadow: none; }}
.ufx-hero [data-testid="stHeaderActionElements"] {{ display: none; }}
.ufx-hero p {{ color: {C['muted']}; font-size: 1.02rem; max-width: 38rem; margin: 0 auto; line-height: 1.55; }}
.ufx-hero > * {{ animation: ufx-in .8s var(--m-ease) both; }} .ufx-hero > *:nth-child(2) {{ animation-delay: .08s; }} .ufx-hero > *:nth-child(3) {{ animation-delay: .16s; }} .ufx-hero > *:nth-child(4) {{ animation-delay: .24s; }}
.ufx-hero-viz {{ display: block; width: 100%; height: 110px; margin: 1.3rem auto 0 auto; max-width: 40rem; }}
.ufx-hero-viz .gr line {{ stroke: #1c1c1e; stroke-dasharray: 2 6; }} .ufx-hero-viz .split {{ stroke: #3a3a3d; stroke-dasharray: 3 5; }}
.ufx-hero-viz .obs {{ fill: none; stroke: #9a9aa0; stroke-width: 2; stroke-dasharray: 900; stroke-dashoffset: 0; }}
.ufx-hero-viz .fc {{ fill: none; stroke: #fff; stroke-width: 2.4; stroke-dasharray: 6 6; filter: drop-shadow(0 0 6px #ffffff88); }}
.ufx-hero-viz .band {{ fill: rgba(255,255,255,.09); }} .ufx-hero-viz .dot {{ fill: #000; stroke: #fff; stroke-width: 2; }}
.st-key-ufa_drop [data-testid="stFileUploaderDropzone"] {{ position: relative; isolation: isolate; min-height: 9.5rem; border-radius: 18px; border: 1px solid var(--hair2);
  background: linear-gradient(180deg, rgba(20,20,21,.86), rgba(8,8,9,.9)); box-shadow: 0 24px 60px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.05);
  transition: border-color var(--m-med) var(--m-ease), transform var(--m-med) var(--m-ease), box-shadow var(--m-med) var(--m-ease); overflow: hidden; }}
.st-key-ufa_drop [data-testid="stFileUploaderDropzone"]::before {{ content: ""; position: absolute; inset: -1px; z-index: -1; border-radius: inherit; padding: 1px; opacity: 0;
  background: conic-gradient(from var(--ufx-a, 0deg), transparent 0 70%, rgba(255,255,255,.9) 82%, transparent 92%);
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0); -webkit-mask-composite: xor; mask-composite: exclude; transition: opacity var(--m-med); }}
.st-key-ufa_drop [data-testid="stFileUploaderDropzone"]::after {{ content: ""; position: absolute; inset: 0; z-index: -1; pointer-events: none;
  background: linear-gradient(100deg, transparent 30%, rgba(255,255,255,.07) 50%, transparent 70%); background-size: 260% 100%; background-position: 150% 0; }}
.st-key-ufa_drop [data-testid="stFileUploaderDropzone"]:hover, .st-key-ufa_drop.ufa-drag [data-testid="stFileUploaderDropzone"] {{ border-color: rgba(255,255,255,.4); box-shadow: 0 24px 70px rgba(0,0,0,.5), 0 0 40px rgba(255,255,255,.07); }}
.st-key-ufa_drop.ufa-drag [data-testid="stFileUploaderDropzone"] {{ transform: scale(1.012); }}
.st-key-ufa_drop [data-testid="stFileUploaderDropzone"] {{ display: flex; flex-direction: column-reverse; align-items: center; justify-content: center; gap: .9rem; padding: 1.6rem 1rem; text-align: center; }}
.st-key-ufa_drop [data-testid="stFileUploaderDropzoneInstructions"] {{ margin: 0; text-align: center; display: flex; flex-direction: column; align-items: center; }}
.st-key-ufa_drop [data-testid="stFileUploaderDropzoneInstructions"] span {{ color: {C['muted']}; }}
.st-key-ufa_drop [data-testid="stFileUploaderDropzoneInstructions"]::before {{ content: "Drag and drop a CSV file here"; display: block; font-size: 1.12rem; font-weight: 650; color: #fff; letter-spacing: -.01em; margin-bottom: .2rem; }}
.st-key-ufa_drop [data-testid="stFileUploaderDropzone"] button {{ border-radius: 10px; padding: .45rem 1.1rem; }}
.st-key-ufa_drop [data-testid="stFileUploaderFile"] {{ animation: ufx-in .4s var(--m-ease) both; }}
.st-key-ufa_drop [data-testid="stFileUploaderDropzoneInstructions"] svg, .st-key-ufa_drop [data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"] {{ transition: transform var(--m-med) var(--m-ease); }}
.st-key-ufa_drop [data-testid="stFileUploaderDropzone"]:hover [data-testid="stIconMaterial"], .st-key-ufa_drop.ufa-drag [data-testid="stIconMaterial"] {{ transform: translateY(-3px); }}
@property --ufx-a {{ syntax: '<angle>'; initial-value: 0deg; inherits: false; }}
.ufx-chips {{ display: flex; flex-wrap: wrap; justify-content: center; gap: .4rem; margin: .8rem 0 .2rem 0; }}
.ufx-chips span {{ font-size: .74rem; color: {C['muted']}; border: 1px solid var(--hair); border-radius: 999px; padding: .2rem .65rem; background: rgba(10,10,11,.6); }}
.ufx-file {{ display: grid; grid-template-columns: auto 1fr auto; gap: 1rem; align-items: center; margin: 1rem 0 .4rem 0; padding: 1rem 1.1rem; border-radius: 16px;
  border: 1px solid var(--hair2); background: linear-gradient(180deg, rgba(22,22,23,.9), rgba(10,10,11,.9)); animation: ufx-in .5s var(--m-ease) both; }}
.ufx-file .ic {{ width: 2.6rem; height: 2.6rem; border-radius: 12px; display: grid; place-items: center; border: 1px solid rgba(255,255,255,.3); background: rgba(255,255,255,.06); }}
.ufx-file .ic svg {{ width: 1.3rem; height: 1.3rem; }} .ufx-file .ic path {{ fill: none; stroke: #fff; stroke-width: 2.2; stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: 30; stroke-dashoffset: 0; }}
.ufx-file .nm {{ font-weight: 650; overflow-wrap: anywhere; }} .ufx-file .mt {{ color: {C['muted']}; font-size: .82rem; }}
.ufx-file .st {{ display: flex; flex-direction: column; gap: .25rem; align-items: flex-end; font-size: .76rem; }}
.ufx-ok {{ color: {C['text']}; }} .ufx-ok::before {{ content: "● "; color: #fff; }} .ufx-need {{ color: {C['muted']}; }} .ufx-need::before {{ content: "○ "; }}
.st-key-ufa_upload_err {{ animation: ufx-shake .42s var(--m-ease) both; }}
.ufx-err {{ border: 1px solid rgba(255,255,255,.5); border-radius: 16px; padding: 1rem 1.1rem; margin: 1rem 0 .4rem 0; background: rgba(28,28,30,.9); animation: ufx-shake .42s var(--m-ease) both; }}
.ufx-err b {{ display: block; letter-spacing: .08em; font-size: .74rem; text-transform: uppercase; color: {C['muted']}; }} .ufx-err p {{ margin: .3rem 0 0 0; }} .ufx-err small {{ color: {C['muted']}; }}
@keyframes ufx-shake {{ 0% {{ transform: translateX(0); opacity: 0; }} 25% {{ transform: translateX(-5px); opacity: 1; }} 50% {{ transform: translateX(4px); }} 75% {{ transform: translateX(-2px); }} 100% {{ transform: none; }} }}
/* ---- dataset preview */
.ufx-accept {{ display: flex; align-items: center; gap: .7rem; padding: .55rem .9rem; border-radius: 12px; border: 1px solid var(--hair2); background: rgba(14,14,15,.85); margin-bottom: 1rem; font-size: .88rem; }}
.ufx-accept svg {{ width: 1.2rem; height: 1.2rem; flex: none; }} .ufx-accept path {{ fill: none; stroke: #fff; stroke-width: 2.4; stroke-linecap: round; stroke-dasharray: 30; }}
.ufx-accept span {{ color: {C['muted']}; }} .ufx-accept b {{ color: {C['text']}; font-weight: 650; }}
.ufx-ptable .sc {{ overflow: auto; max-height: 30rem; border: 1px solid var(--hair); border-radius: 14px; background: rgba(8,8,9,.82); }}
.ufx-ptable table {{ border-collapse: separate; border-spacing: 0; width: 100%; font-size: .84rem; font-variant-numeric: tabular-nums; }}
.ufx-ptable th {{ position: sticky; top: 0; z-index: 1; text-align: left; padding: .6rem .75rem; background: rgba(16,16,17,.97); border-bottom: 1px solid var(--hair2); white-space: nowrap; }}
.ufx-ptable th span {{ display: block; font-weight: 650; color: {C['text']}; }}
.ufx-ptable th em, .ufx-prof em {{ display: inline-block; margin-top: .2rem; font-style: normal; font-size: .64rem; letter-spacing: .08em; text-transform: uppercase; padding: .05rem .4rem;
  border-radius: 5px; border: 1px solid #3a3a3d; color: {C['muted']}; }}
em.r-timestamp {{ border-color: #fff !important; color: #fff !important; }} em.r-target {{ background: #fff; color: #000 !important; border-color: #fff !important; }}
em.r-group {{ border-style: dashed !important; color: #d4d4d8 !important; }} em.r-id, em.r-empty {{ opacity: .55; }}
.ufx-ptable td {{ padding: .42rem .75rem; border-bottom: 1px solid #161618; white-space: nowrap; color: #d4d4d8; transition: background var(--m-fast); }}
.ufx-ptable td.n {{ text-align: right; }} .ufx-ptable td.ix, .ufx-ptable th.ix {{ color: {C['dim']}; text-align: right; width: 2.5rem; }}
.ufx-ptable td.miss {{ color: {C['dim']}; font-style: italic; background: repeating-linear-gradient(135deg, rgba(255,255,255,.05) 0 4px, transparent 4px 8px); }}
.ufx-ptable tbody tr {{ animation: ufx-in .45s var(--m-ease) both; animation-delay: calc(var(--i) * 28ms); }}
.ufx-ptable tbody tr:hover td {{ background: rgba(255,255,255,.07); color: #fff; }}
.ufx-profiles {{ display: grid; gap: 0; border: 1px solid var(--hair); border-radius: 14px; overflow: hidden; background: rgba(8,8,9,.82); }}
.ufx-prof {{ display: grid; grid-template-columns: minmax(10rem, 1.3fr) minmax(8rem, 1.6fr) minmax(9rem, 1.2fr) 6.5rem; gap: 1rem; align-items: center; padding: .6rem .9rem;
  border-top: 1px solid #161618; font-size: .84rem; animation: ufx-in .5s var(--m-ease) both; animation-delay: calc(var(--i) * 40ms); transition: background var(--m-fast); }}
.ufx-prof:first-child {{ border-top: 0; }} .ufx-prof:hover {{ background: rgba(255,255,255,.04); }}
.ufx-prof .nm b {{ display: block; overflow-wrap: anywhere; }} .ufx-prof .ds, .ufx-prof .ms span:last-child, .ufx-prof .na {{ color: {C['muted']}; font-size: .78rem; }}
.ufx-prof .ms {{ display: flex; flex-direction: column; gap: .25rem; }}
.ufx-prof .bar {{ height: .34rem; border-radius: 99px; background: #1c1c1e; overflow: hidden; }} .ufx-prof .bar i {{ display: block; height: 100%; width: var(--w); background: #fff; border-radius: 99px; }}
.ufx-cspark {{ width: 100%; height: 34px; display: block; }} .ufx-cspark path {{ fill: none; stroke: #d4d4d8; stroke-width: 1.4; stroke-linejoin: round; }}
.ufx-mmap {{ border: 1px solid var(--hair); border-radius: 14px; padding: .9rem; background: rgba(8,8,9,.82); }}
.ufx-mmap .mr {{ display: grid; grid-template-columns: 10rem 1fr; gap: .8rem; align-items: center; padding: .14rem 0; font-size: .78rem; color: {C['muted']}; }}
.ufx-mmap .mr span {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.ufx-mmap .cells {{ display: grid; grid-template-columns: repeat(var(--k), 1fr); gap: 2px; }}
.ufx-mmap .cells i {{ display: block; height: .8rem; border-radius: 2px; background: rgba(255,255,255, calc(.05 + var(--a) * .95)); }}
/* ---- configure */
.st-key-ufa_cfg [data-testid="stElementContainer"] {{ animation: ufx-in .55s var(--m-ease) both; }}
.st-key-ufa_cfg [data-testid="stColumn"]:nth-child(2) [data-testid="stElementContainer"] {{ animation-delay: .12s; }}
.ufa-hint .ufa-badge {{ animation: ufx-blur .6s var(--m-ease) both; animation-delay: .25s; }}
.ufa-badge.changed {{ background: #fff; color: #000; border-color: #fff; }}
.ufx-req {{ border: 1px solid var(--hair2); border-radius: 16px; padding: 1rem 1.1rem; background: linear-gradient(180deg, rgba(20,20,21,.9), rgba(9,9,10,.9)); }}
.ufx-req h4 {{ margin: 0 0 .7rem 0; font-size: .72rem; text-transform: uppercase; letter-spacing: .14em; color: {C['muted']}; }}
.ufx-req .sent {{ font-size: 1.05rem; line-height: 1.6; color: #d4d4d8; }} .ufx-req .sent b {{ color: #fff; font-weight: 650; border-bottom: 1px solid rgba(255,255,255,.3); }}
.ufx-req .sent b.chg {{ background: rgba(255,255,255,.14); border-radius: 4px; padding: 0 .2rem; }}
.ufx-req .chk {{ margin-top: .8rem; display: grid; gap: .3rem; font-size: .8rem; }}
.ufx-pulse-a {{ animation: ufx-flash-a .7s var(--m-ease) both; }} .ufx-pulse-b {{ animation: ufx-flash-b .7s var(--m-ease) both; }}
@keyframes ufx-flash-a {{ 0% {{ box-shadow: 0 0 0 0 rgba(255,255,255,.35); }} 100% {{ box-shadow: 0 0 0 14px rgba(255,255,255,0); }} }}
@keyframes ufx-flash-b {{ 0% {{ box-shadow: 0 0 0 0 rgba(255,255,255,.35); }} 100% {{ box-shadow: 0 0 0 14px rgba(255,255,255,0); }} }}
.ufa-field-error {{ animation: ufx-in .3s var(--m-ease) both; margin: -.35rem 0 .7rem 0; }}
/* ---- results */
.st-key-ufa_page_results .ufa-kpi, .st-key-ufa_page_insights .ufa-kpi, .st-key-ufa_page_preview .ufa-kpi {{ animation: ufx-blur .7s var(--m-ease) both; }}
[data-testid="stColumn"]:nth-child(2) .ufa-kpi {{ animation-delay: .06s; }} [data-testid="stColumn"]:nth-child(3) .ufa-kpi {{ animation-delay: .12s; }}
[data-testid="stColumn"]:nth-child(4) .ufa-kpi {{ animation-delay: .18s; }} [data-testid="stColumn"]:nth-child(5) .ufa-kpi {{ animation-delay: .24s; }}
.ufa-kpi {{ position: relative; overflow: hidden; background: linear-gradient(180deg, rgba(22,22,23,.88), rgba(10,10,11,.88)); border-color: var(--hair2); }}
.ufa-kpi::after {{ content: ""; position: absolute; left: 0; right: 0; top: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,.35), transparent); }}
.ufx-panel {{ animation: ufx-in .6s var(--m-ease) both; background: linear-gradient(180deg, rgba(18,18,19,.88), rgba(9,9,10,.88)); border-color: var(--hair2); }}
.ufx-row .tr i {{ transform-origin: left; animation: ufx-bar .9s var(--m-ease) both; }}
@keyframes ufx-bar {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
iframe[title="st.iframe"] {{ border: 0; }}
.st-key-ufa_chart iframe {{ border-radius: 16px; }}
.st-key-ufa_chart {{ border: 1px solid var(--hair2); border-radius: 18px; padding: .9rem 1rem .6rem 1rem; background: linear-gradient(180deg, rgba(14,14,15,.9), rgba(6,6,7,.9));
  box-shadow: 0 30px 80px rgba(0,0,0,.45); animation: ufx-in .7s var(--m-ease) both; }}
/* ---- processing */
.st-key-ufa_page_processing .ufx-load-card {{ box-shadow: 0 40px 120px rgba(0,0,0,.6), 0 0 80px rgba(255,255,255,.05); }}
.ufx-stages li {{ transition: border-color var(--m-med) var(--m-ease), color var(--m-med) var(--m-ease), background var(--m-med) var(--m-ease); }}
.ufx-stages li.done {{ animation: ufx-blur .45s var(--m-ease) both; }}
.ufx-stages li.running {{ background: linear-gradient(90deg, rgba(255,255,255,.08), rgba(10,10,11,.9)); }}
.ufx-band {{ background: linear-gradient(90deg, #cfcfd3, #ffffff); box-shadow: 0 0 40px rgba(255,255,255,.12); }}
/* ---- insights / reports */
.ufx-facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: .6rem; margin: .4rem 0 1rem 0; }}
.ufx-fact {{ border: 1px solid var(--hair); border-radius: 14px; padding: .8rem .95rem; background: rgba(12,12,13,.85); font-size: .86rem; color: #d4d4d8; animation: ufx-in .55s var(--m-ease) both;
  animation-delay: calc(var(--i, 0) * 60ms); transition: border-color var(--m-fast), transform var(--m-fast) var(--m-ease); }}
.ufx-fact:hover {{ border-color: rgba(255,255,255,.3); transform: translateY(-2px); }}
.ufx-fact b {{ display: block; font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; color: {C['muted']}; margin-bottom: .3rem; font-weight: 600; }}
.ufx-fact .big {{ font-size: 1.25rem; font-weight: 680; color: #fff; }}
.st-key-ufa_ins_charts [data-testid="stVegaLiteChart"], .st-key-ufa_rep_list [data-testid="stDataFrame"], .st-key-ufa_ins_table [data-testid="stDataFrame"] {{ animation: ufx-in .7s var(--m-ease) both; }}
.ufx-repcard {{ border: 1px solid var(--hair2); border-radius: 18px; padding: 1.1rem 1.2rem; background: linear-gradient(180deg, rgba(20,20,21,.9), rgba(8,8,9,.9)); animation: ufx-in .5s var(--m-ease) both; }}
.ufx-repcard h3 {{ margin: .1rem 0 .2rem 0; font-size: 1.25rem; letter-spacing: -.02em; overflow-wrap: anywhere; }}
.ufx-repcard .meta {{ display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .55rem; }}
.ufx-repcard .meta span {{ font-size: .74rem; border: 1px solid var(--hair2); border-radius: 999px; padding: .15rem .6rem; color: #d4d4d8; }}
[data-testid="stTabs"] [role="tab"] {{ transition: color var(--m-fast); }} [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: #fff; transition: all .4s var(--m-ease) !important; }}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {{ animation: ufx-fade .4s var(--m-ease) both; }}
[data-testid="stToast"] {{ background: rgba(16,16,17,.96) !important; border: 1px solid var(--hair2); }}
[data-testid="stExpander"] details {{ transition: border-color var(--m-fast); }} [data-testid="stExpander"] details:hover {{ border-color: #3a3a3d; }}
[data-baseweb="select"] > div, [data-testid="stNumberInputContainer"], .stTextInput input {{ transition: border-color var(--m-fast), box-shadow var(--m-med); }}
[data-baseweb="select"] > div:hover, [data-testid="stNumberInputContainer"]:hover {{ border-color: #55555a !important; }}
@media (prefers-reduced-motion: no-preference) {{
  .ufx-hero-viz .obs {{ animation: ufx-draw-hero 2.4s var(--m-io) .2s both; }}
  .ufx-hero-viz .fc {{ animation: ufx-dash 1.6s linear infinite, ufx-fade 1s ease 2.2s both; }}
  .ufx-hero-viz .band {{ animation: ufx-fade 1.2s ease 2.4s both, ufx-breathe2 6s ease-in-out 3.6s infinite; }}
  .ufx-hero-viz .dot {{ animation: ufx-fade .4s ease 2.2s both; }}
  .st-key-ufa_drop [data-testid="stFileUploaderDropzone"]:hover::before, .st-key-ufa_drop.ufa-drag [data-testid="stFileUploaderDropzone"]::before {{ opacity: 1; animation: ufx-spin 2.8s linear infinite; }}
  .st-key-ufa_drop [data-testid="stFileUploaderDropzone"]:hover::after, .st-key-ufa_drop.ufa-drag [data-testid="stFileUploaderDropzone"]::after {{ animation: ufx-sheen 1.6s var(--m-ease) both; }}
  .ufx-file .ic path, .ufx-accept path {{ animation: ufx-check .7s var(--m-ease) .15s both; }}
  .ufx-cspark path {{ stroke-dasharray: var(--len); animation: ufx-draw-once 1.4s var(--m-io) both; }}
  .ufx-prof .bar i {{ animation: ufx-bar 1s var(--m-ease) .2s both; transform-origin: left; }}
}}
@keyframes ufx-draw-hero {{ from {{ stroke-dashoffset: 900; }} to {{ stroke-dashoffset: 0; }} }}
@keyframes ufx-draw-once {{ from {{ stroke-dashoffset: var(--len); }} to {{ stroke-dashoffset: 0; }} }}
@keyframes ufx-dash {{ to {{ stroke-dashoffset: -24; }} }}
@keyframes ufx-breathe2 {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: .55; }} }}
@keyframes ufx-spin {{ to {{ --ufx-a: 360deg; }} }}
@keyframes ufx-sheen {{ from {{ background-position: 150% 0; }} to {{ background-position: -50% 0; }} }}
@keyframes ufx-check {{ from {{ stroke-dashoffset: 30; }} to {{ stroke-dashoffset: 0; }} }}
@media (max-width: 760px) {{
  .st-key-ufa_nav {{ position: relative; top: 0; }}
  .st-key-ufa_nav [data-testid="stHorizontalBlock"]::before, .st-key-ufa_nav [data-testid="stHorizontalBlock"]::after {{ display: none; }}
  .st-key-ufa_nav [data-testid="stHorizontalBlock"] {{ flex-flow: row wrap; }}
  .st-key-ufa_nav [data-testid="stColumn"] {{ flex: 1 1 calc(33.3% - 6px) !important; min-width: calc(33.3% - 6px) !important; width: auto !important; }}
  .st-key-ufa_nav .stButton > button {{ height: 52px; padding: 0 .3rem; }} .st-key-ufa_nav .stButton > button p {{ font-size: .72rem; white-space: normal; line-height: 1.15; }}
  .st-key-ufa_nav .stButton > button[kind="primary"] {{ background: rgba(255,255,255,.12) !important; border-color: rgba(255,255,255,.34) !important; }}
  .ufx-prof {{ grid-template-columns: 1fr 1fr; }} .ufx-prof .ds {{ display: none; }} .ufx-mmap .mr {{ grid-template-columns: 6rem 1fr; }}
  .ufx-file {{ grid-template-columns: auto 1fr; }} .ufx-file .st {{ grid-column: 1 / -1; align-items: flex-start; }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition: none !important; }}
}}
</style>
"""


def theme():
    st.markdown("<div class='ufx-css'>" + CSS + MOTION_CSS + "</div>", unsafe_allow_html=True)


def nav_state(index: int):
    """Moves the navigation indicator: a constant style rule holding only the page index (no user text)."""
    st.markdown(f"<div class='ufx-css'><style>:root{{--ufa-nav-i:{int(index)}}}</style></div>", unsafe_allow_html=True)


SATURN_PLACES = {"upload": (0.88, 0.60, 0.15, 0.34), "processing": (0.84, 0.70, 0.17, 0.42), "preview": (0.93, 0.86, 0.13, 0.22),
                 "configure": (0.93, 0.86, 0.13, 0.22), "results": (0.94, 0.88, 0.12, 0.16), "insights": (0.93, 0.86, 0.13, 0.18), "reports": (0.93, 0.86, 0.13, 0.18)}


def saturn_place(page: str):
    """Where the planet sits on this page (x, y from the bottom-left, size, opacity). The shader eases to it slowly."""
    x, y, s, o = SATURN_PLACES.get(page, SATURN_PLACES["preview"])
    st.markdown(f"<div class='ufx-css'><style>:root{{--ufa-sat:{x} {y} {s} {o}}}</style></div>", unsafe_allow_html=True)


def banner(text: str | None):
    if text:
        st.markdown(f"<div class='ufa-banner' role='status'>{VM.safe(text)}</div>", unsafe_allow_html=True)


def card(title: str, body: str):
    st.markdown(f"<div class='ufa-card'><h4>{VM.safe(title)}</h4><p>{VM.safe(body)}</p></div>", unsafe_allow_html=True)


def kpi(col, label: str, value, sub: str = ""):
    col.markdown(f"<div class='ufa-kpi'><div class='k'>{VM.safe(label)}</div><div class='v'>{VM.safe(value)}</div><div class='s'>{VM.safe(sub)}</div></div>", unsafe_allow_html=True)


def badge(text: str, kind: str = "") -> str:
    return f"<span class='ufa-badge {kind}'>{VM.safe(text)}</span>"


def field_error(msg: str | None):
    if msg:
        st.markdown(f"<div class='ufa-field-error' role='alert'>{VM.safe(msg)}</div>", unsafe_allow_html=True)


def empty_state(title: str, body: str):
    st.markdown(f"<div class='ufa-empty'><h4>{VM.safe(title)}</h4><div>{VM.safe(body)}</div></div>", unsafe_allow_html=True)


def issues_list(issues: list[dict]):
    """Engine / service issues, never hidden: errors, then warnings, then notes."""
    for sev, fn in (("error", st.error), ("warning", st.warning), ("info", st.info)):
        for i in [x for x in issues if str(x.get("severity")) == sev]:
            hint = f" What to do: {i.get('hint')}" if i.get("hint") else ""
            fn(f"{i.get('code', 'ISSUE')}: {i.get('message', '')}{hint}")


def forecast_chart(hist: pd.DataFrame, fc: pd.DataFrame, levels: list[str], target_label: str, direction: str = "flat") -> alt.Chart:
    """Observed history in gray, the forecast in white, or green / red when `direction` says the forecast ends above / below
    its stated baseline (the caller states the baseline next to the chart). Interval bands are gray. Values are plotted as given."""
    fcol = {"up": C["pos"], "down": C["neg"]}.get(direction, C["text"])
    ax = dict(labelColor=C["muted"], titleColor=C["muted"], gridColor="#1c1c1e", domainColor=C["border"], tickColor=C["border"], labelFontSize=12, titleFontSize=12,
              labelFont="Segoe UI", titleFont="Segoe UI")
    x = alt.X("timestamp:T", title=None, axis=alt.Axis(**ax))
    y = alt.Y("value:Q", title=str(target_label)[:60], scale=alt.Scale(zero=False), axis=alt.Axis(**ax))
    layers = []
    for j, lv in enumerate(sorted(levels, key=float, reverse=True)):
        k = VM.level_key(lv)
        layers.append(alt.Chart(fc).mark_area(color="#ffffff", opacity={0: 0.10, 1: 0.18}.get(j, 0.14)).encode(
            x=x, y=alt.Y(f"lo_{k}:Q", title=str(target_label)[:60], scale=alt.Scale(zero=False)), y2=f"hi_{k}:Q",
            tooltip=[alt.Tooltip("timestamp:T", title="Time"), alt.Tooltip(f"lo_{k}:Q", title=f"Lower ({float(lv):.0%})"), alt.Tooltip(f"hi_{k}:Q", title=f"Upper ({float(lv):.0%})")]))
    if len(hist):
        layers.append(alt.Chart(hist).mark_line(color="#9a9aa0", strokeWidth=1.5).encode(x=x, y=y, tooltip=[alt.Tooltip("timestamp:T", title="Time"), alt.Tooltip("value:Q", title="Observed")]))
    if len(fc):
        tip = [alt.Tooltip("timestamp:T", title="Time"), alt.Tooltip("value:Q", title="Forecast"), alt.Tooltip("step:Q", title="Step")]
        layers.append(alt.Chart(fc).mark_line(color=fcol, strokeWidth=2.4, point=alt.OverlayMarkDef(color=fcol, size=30)).encode(x=x, y=y, tooltip=tip))
    return alt.layer(*layers).properties(height=390, background="transparent").configure_view(stroke=C["border"]).interactive(bind_y=False)
