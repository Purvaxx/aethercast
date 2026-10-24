"""Interactive, animated charts drawn in the browser with plain SVG and a few lines of JavaScript (no library, nothing
fetched).

Integrity rules:
  * Every plotted value is passed through unchanged: observed values from the file, forecast points and interval bounds
    from the engine response. The browser only scales them to pixels.
  * Animation never draws an intermediate or interpolated value. Lines are fully computed first and then REVEALED by
    a growing clip rectangle; bands fade in; zooming changes the visible time window, not the data.
  * Tooltips show the stored numbers for the nearest real timestamp (observed or predicted, labelled as such).

Isolation: the page is delivered as a data: URL, so the browser gives it an opaque origin. Column names and other text
from an upload travel as JSON (with <, > and & escaped) and are only ever written with textContent, never as markup.
"""
from __future__ import annotations

import base64
import json
import math

import pandas as pd

from . import viewmodels as VM


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) else None


def _iso(t) -> str:
    try:
        return pd.Timestamp(t).isoformat()
    except Exception:                                             # noqa: BLE001
        return str(t)


def safe_json(obj) -> str:
    """JSON that cannot close a <script> element or start markup."""
    return (json.dumps(obj, allow_nan=False, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))


def data_url(html: str) -> str:
    return "data:text/html;charset=utf-8;base64," + base64.b64encode(html.encode("utf-8")).decode("ascii")


def forecast_payload(hist: pd.DataFrame, series: dict, target_label: str, direction: str, recent_points: int) -> dict:
    """Everything the chart needs, copied from the inputs without modification."""
    obs = [[_iso(t), _num(v)] for t, v in zip(hist["timestamp"], hist["value"])] if len(hist) else []
    levels = sorted(VM.interval_levels(series), key=float)
    fc = []
    for p in series.get("forecast") or []:
        bands = {lv: [_num(p["intervals"][lv][0]), _num(p["intervals"][lv][1])] for lv in levels if lv in (p.get("intervals") or {})}
        fc.append({"t": _iso(p["timestamp"]), "y": _num(p["point"]), "step": p.get("step"), "b": bands})
    return {"obs": obs, "fc": fc, "levels": levels, "label": str(target_label)[:80], "dir": direction if direction in ("up", "down", "flat") else "none",
            "recent": int(max(recent_points, 10))}


CHART_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#000;--txt:#f5f5f5;--mut:#a8a8ad;--dim:#6b6b70;--grid:#1c1c1e;--obs:#9a9aa0;--ease:cubic-bezier(.2,.8,.2,1)}
*{box-sizing:border-box}html,body{margin:0;background:transparent;color:var(--txt);font:13px 'Segoe UI',system-ui,-apple-system,Arial,sans-serif;overflow:hidden}
.top{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;margin:0 0 6px 0}
.leg{display:flex;gap:14px;flex-wrap:wrap;color:var(--mut);font-size:12px}.leg i{display:inline-block;width:18px;height:0;border-top:2px solid var(--obs);vertical-align:middle;margin-right:6px}
.leg .f i{border-top-width:3px}.leg .b i{height:9px;border:0;background:rgba(255,255,255,.18);border-radius:2px}
.seg{display:inline-flex;padding:3px;border:1px solid #2a2a2c;border-radius:10px;background:rgba(13,13,14,.85);position:relative}
.seg button{position:relative;z-index:1;border:0;background:transparent;color:var(--mut);font:600 12px inherit;font-family:inherit;padding:5px 11px;border-radius:7px;cursor:pointer;transition:color .2s var(--ease)}
.seg button:hover{color:var(--txt)}.seg button[aria-pressed=true]{color:#000}.seg button:focus-visible{outline:2px solid #fff;outline-offset:1px}
.seg .ind{position:absolute;z-index:0;top:3px;bottom:3px;border-radius:7px;background:linear-gradient(180deg,#fafafa,#cfcfd3);box-shadow:0 0 14px rgba(255,255,255,.25);transition:left .35s var(--ease),width .35s var(--ease)}
.wrap{position:relative}svg{display:block;width:100%;overflow:visible}svg:focus{outline:none}svg:focus-visible{outline:1px solid #6b6b70;border-radius:8px}
.ax text{fill:var(--mut);font-size:11px}.ax line{stroke:var(--grid)}.split{stroke:#55555a;stroke-dasharray:4 4}.splitlab{fill:var(--dim);font-size:11px;letter-spacing:.06em}
.tip{position:absolute;pointer-events:none;min-width:150px;max-width:260px;padding:9px 11px;border-radius:10px;background:rgba(10,10,11,.94);border:1px solid #3a3a3d;
 box-shadow:0 8px 30px rgba(0,0,0,.6);font-size:12px;line-height:1.45;opacity:0;transform:translateY(4px);transition:opacity .14s var(--ease),transform .14s var(--ease)}
.tip.on{opacity:1;transform:none}.tip .k{color:var(--mut);font-size:11px;letter-spacing:.05em;text-transform:uppercase}.tip .v{font-size:15px;font-weight:650;font-variant-numeric:tabular-nums}
.tip .r{color:var(--mut);font-variant-numeric:tabular-nums}
.hint{color:var(--dim);font-size:11px;margin-top:4px}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
</style></head><body>
<div class="top"><div class="leg"><span><i></i>Observed</span><span class="f"><i id="fi"></i>Forecast</span><span class="b" id="bl"><i></i>Prediction interval</span></div>
<div class="seg" role="group" aria-label="Visible time range"><span class="ind" id="ind"></span>
<button type="button" data-r="recent" aria-pressed="true">Recent</button><button type="button" data-r="focus" aria-pressed="false">Forecast focus</button><button type="button" data-r="all" aria-pressed="false">All history</button></div></div>
<div class="wrap" id="wrap"><svg id="svg" tabindex="0" role="img"></svg><div class="tip" id="tip" role="status" aria-live="polite"></div></div>
<div class="hint">Hover or use the arrow keys to read exact values. Ranges change the window only; no value is altered.</div>
<script>
(function(){
var D=__DATA__;
var NS='http://www.w3.org/2000/svg',svg=document.getElementById('svg'),tip=document.getElementById('tip'),wrap=document.getElementById('wrap');
var reduce=window.matchMedia&&matchMedia('(prefers-reduced-motion: reduce)').matches;
var FC={up:'#22c55e',down:'#ef4444'}[D.dir]||'#ffffff';document.getElementById('fi').style.borderTopColor=FC;
if(!D.levels.length)document.getElementById('bl').style.display='none';
svg.setAttribute('aria-label','Chart of '+D.label+': '+D.obs.length+' observed values and '+D.fc.length+' forecast steps from the engine');
function pt(s){return Date.parse(/([zZ]|[+-]\d\d:?\d\d)$/.test(s)?s:s+'Z')}
 var obs=D.obs.filter(function(p){return p[1]!==null}).map(function(p){return {t:pt(p[0]),y:p[1]}}).filter(function(p){return isFinite(p.t)});
var fc=D.fc.filter(function(p){return p.y!==null}).map(function(p){return {t:pt(p.t),y:p.y,s:p.step,b:p.b,iso:p.t}}).filter(function(p){return isFinite(p.t)});
var H=400,M={l:64,r:16,t:14,b:30},W=600;
function el(n,a,p){var e=document.createElementNS(NS,n);for(var k in a)e.setAttribute(k,a[k]);(p||svg).appendChild(e);return e}
function fmt(v){if(v===null||v===undefined)return 'unavailable';var a=Math.abs(v);if(a!==0&&(a>=1e9||a<1e-4))return v.toExponential(4);var s=v.toLocaleString('en-US',{maximumFractionDigits:6});return s}
function fdate(t){var d=new Date(t);var iso=d.toISOString();return iso.slice(0,16).replace('T',' ')}
function windowFor(r){var last=fc.length?fc[fc.length-1].t:(obs.length?obs[obs.length-1].t:0);var i0=0;
 if(r==='recent')i0=Math.max(0,obs.length-D.recent);else if(r==='focus')i0=Math.max(0,obs.length-Math.max(3*fc.length,24));
 var first=obs.length?obs[i0].t:(fc.length?fc[0].t:0);if(first===last){first-=3600e3;last+=3600e3}return [first,last]}
var dom=windowFor('recent'),cur=dom.slice();
function yExtent(a,b){var lo=Infinity,hi=-Infinity;obs.forEach(function(p){if(p.t>=a&&p.t<=b){lo=Math.min(lo,p.y);hi=Math.max(hi,p.y)}});
 fc.forEach(function(p){lo=Math.min(lo,p.y);hi=Math.max(hi,p.y);for(var k in p.b){if(p.b[k][0]!==null)lo=Math.min(lo,p.b[k][0]);if(p.b[k][1]!==null)hi=Math.max(hi,p.b[k][1])}});
 if(!isFinite(lo)){lo=0;hi=1}if(lo===hi){lo-=1;hi+=1}var pad=(hi-lo)*.06;return [lo-pad,hi+pad]}
function nice(lo,hi,n){var span=hi-lo,step=Math.pow(10,Math.floor(Math.log10(span/n))),e=span/n/step;step*=e>=5?10:e>=2?5:e>=1?2:1;var out=[];for(var v=Math.ceil(lo/step)*step;v<=hi+1e-9*step;v+=step)out.push(v);return out}
var gAx,gBand,gObs,gFc,gPts,gHov,clipO,clipF,splitL,splitT,yd=[0,1],X,Y,revealed=reduce;
function build(){svg.innerHTML='';W=Math.max(wrap.clientWidth,280);svg.setAttribute('viewBox','0 0 '+W+' '+H);svg.setAttribute('height',H);
 var defs=el('defs',{});var c1=el('clipPath',{id:'co'},defs);clipO=el('rect',{x:M.l,y:0,width:reduce?W:0,height:H},c1);
 var c2=el('clipPath',{id:'cf'},defs);clipF=el('rect',{x:M.l,y:0,width:reduce?W:0,height:H},c2);var c3=el('clipPath',{id:'cp'},defs);el('rect',{x:M.l,y:0,width:W-M.l-M.r,height:H},c3);
 gAx=el('g',{'class':'ax'});var plot=el('g',{'clip-path':'url(#cp)'});gBand=el('g',{'clip-path':'url(#cf)'},plot);gObs=el('g',{'clip-path':'url(#co)'},plot);gFc=el('g',{'clip-path':'url(#cf)'},plot);
 splitL=el('line',{'class':'split'},plot);splitT=el('text',{'class':'splitlab'});gPts=el('g',{'clip-path':'url(#cf)'},plot);gHov=el('g',{},plot);render()}
function render(){var a=cur[0],b=cur[1];yd=yExtent(a,b);X=function(t){return M.l+(W-M.l-M.r)*(t-a)/(b-a)};Y=function(v){return M.t+(H-M.t-M.b)*(1-(v-yd[0])/(yd[1]-yd[0]))};
 gAx.innerHTML='';nice(yd[0],yd[1],5).forEach(function(v){var y=Y(v);el('line',{x1:M.l,x2:W-M.r,y1:y,y2:y},gAx);var t=el('text',{x:M.l-8,y:y+4,'text-anchor':'end'},gAx);t.textContent=fmt(+v.toPrecision(12))});
 var n=Math.max(2,Math.min(7,Math.floor((W-M.l-M.r)/110)));for(var i=0;i<=n;i++){var t=a+(b-a)*i/n;var tx=el('text',{x:X(t),y:H-8,'text-anchor':i===0?'start':i===n?'end':'middle'},gAx);var d=new Date(t);
  tx.textContent=(b-a)>3*864e5?d.toISOString().slice(0,10):d.toISOString().slice(5,16).replace('T',' ')}
 gBand.innerHTML='';D.levels.slice().reverse().forEach(function(lv,j){var up=[],dn=[];fc.forEach(function(p){if(p.b[lv]&&p.b[lv][0]!==null){up.push(X(p.t)+','+Y(p.b[lv][1]));dn.unshift(X(p.t)+','+Y(p.b[lv][0]))}});
  if(up.length)el('polygon',{points:up.concat(dn).join(' '),fill:'#ffffff',opacity:j===0?.10:.18},gBand)});
 gObs.innerHTML='';var vis=obs.filter(function(p){return p.t>=a-(b-a)*.02});var step=Math.max(1,Math.floor(vis.length/(W*1.5)));var pts=[];for(var k=0;k<vis.length;k+=step)pts.push(X(vis[k].t).toFixed(1)+','+Y(vis[k].y).toFixed(1));
  if(vis.length)pts.push(X(vis[vis.length-1].t).toFixed(1)+','+Y(vis[vis.length-1].y).toFixed(1));
 if(pts.length)el('polyline',{points:pts.join(' '),fill:'none',stroke:'#9a9aa0','stroke-width':1.5,'stroke-linejoin':'round'},gObs);
 gFc.innerHTML='';gPts.innerHTML='';var lo=obs.length?obs[obs.length-1]:null;var fp=(lo?[X(lo.t)+','+Y(lo.y)]:[]).concat(fc.map(function(p){return X(p.t)+','+Y(p.y)}));
 if(fc.length){el('polyline',{points:fp.join(' '),fill:'none',stroke:FC,'stroke-width':2.4,'stroke-linejoin':'round','stroke-linecap':'round',style:'filter:drop-shadow(0 0 6px '+FC+'66)'},gFc);
  fc.forEach(function(p){el('circle',{cx:X(p.t),cy:Y(p.y),r:fc.length>60?1.6:3,fill:FC},gPts)})}
 if(lo&&fc.length){var sx=(X(lo.t)+X(fc[0].t))/2;splitL.setAttribute('x1',sx);splitL.setAttribute('x2',sx);splitL.setAttribute('y1',M.t);splitL.setAttribute('y2',H-M.b);
  splitT.setAttribute('x',Math.min(sx+6,W-M.r-60));splitT.setAttribute('y',M.t+11);splitT.textContent='FORECAST'}
 if(revealed){clipO.setAttribute('width',W);clipF.setAttribute('width',W)}}
function reveal(){if(reduce){revealed=true;return}var t0=null,split=obs.length&&fc.length?(X(obs[obs.length-1].t)):W;gBand.style.opacity=0;
 function f(ts){if(t0===null)t0=ts;var e=ts-t0;var p1=Math.min(e/900,1),q1=1-Math.pow(1-p1,3);clipO.setAttribute('width',M.l+(split-M.l)*q1-M.l+2);
  if(e>700){var p2=Math.min((e-700)/650,1),q2=1-Math.pow(1-p2,3);clipF.setAttribute('width',(split-M.l)+(W-split)*q2+4);gBand.style.opacity=Math.min((e-800)/500,1)}
  if(e<1450)requestAnimationFrame(f);else{revealed=true;clipO.setAttribute('width',W);clipF.setAttribute('width',W);gBand.style.opacity=1}}
 requestAnimationFrame(f);setTimeout(function(){if(!revealed){revealed=true;clipO.setAttribute('width',W);clipF.setAttribute('width',W);gBand.style.opacity=1}},2200)}
var tw=null;function tweenTo(nd){if(reduce){cur=nd.slice();render();return}var from=cur.slice(),t0=null;if(tw)cancelAnimationFrame(tw);
 function f(ts){if(t0===null)t0=ts;var p=Math.min((ts-t0)/420,1),q=p<.5?4*p*p*p:1-Math.pow(-2*p+2,3)/2;cur=[from[0]+(nd[0]-from[0])*q,from[1]+(nd[1]-from[1])*q];render();if(p<1)tw=requestAnimationFrame(f)}
 tw=requestAnimationFrame(f);setTimeout(function(){cur=nd.slice();render()},700)}
var btns=[].slice.call(document.querySelectorAll('.seg button')),ind=document.getElementById('ind');
function placeInd(b){ind.style.left=b.offsetLeft+'px';ind.style.width=b.offsetWidth+'px'}
btns.forEach(function(b){b.addEventListener('click',function(){btns.forEach(function(x){x.setAttribute('aria-pressed',x===b)});placeInd(b);tweenTo(windowFor(b.dataset.r))})});
var all=obs.map(function(p){return {t:p.t,y:p.y,k:'Observed'}}).concat(fc.map(function(p){return {t:p.t,y:p.y,k:'Forecast',s:p.s,b:p.b}}));var hi=-1;
function show(i){hi=i;gHov.innerHTML='';if(i<0){tip.classList.remove('on');return}var p=all[i];if(p.t<cur[0]||p.t>cur[1]){tip.classList.remove('on');return}var x=X(p.t),y=Y(p.y);
 el('line',{x1:x,x2:x,y1:M.t,y2:H-M.b,stroke:'#6b6b70','stroke-width':1},gHov);el('line',{x1:M.l,x2:W-M.r,y1:y,y2:y,stroke:'#2a2a2c','stroke-width':1},gHov);
 el('circle',{cx:x,cy:y,r:5,fill:'#000',stroke:p.k==='Forecast'?FC:'#d4d4d8','stroke-width':2},gHov);
 var h='<div class="k"></div><div class="v"></div><div class="r d"></div>';tip.innerHTML=h;tip.querySelector('.k').textContent=p.k==='Forecast'?('Predicted · step '+p.s):'Observed';
 tip.querySelector('.v').textContent=fmt(p.y);tip.querySelector('.d').textContent=fdate(p.t);
 if(p.b){D.levels.slice().reverse().forEach(function(lv){if(p.b[lv]){var r=document.createElement('div');r.className='r';r.textContent=(Math.round(parseFloat(lv)*100))+'% interval: '+fmt(p.b[lv][0])+' to '+fmt(p.b[lv][1]);tip.appendChild(r)}})}
 var tx=x+14,ww=tip.offsetWidth||170;if(tx+ww>W)tx=x-ww-14;tip.style.left=Math.max(0,tx)+'px';tip.style.top=Math.max(0,Math.min(y-30,H-tip.offsetHeight-6))+'px';tip.classList.add('on')}
function nearest(px){var best=-1,bd=Infinity;for(var i=0;i<all.length;i++){if(all[i].t<cur[0]||all[i].t>cur[1])continue;var d=Math.abs(X(all[i].t)-px);if(d<bd){bd=d;best=i}}return best}
svg.addEventListener('pointermove',function(e){var r=svg.getBoundingClientRect();show(nearest((e.clientX-r.left)*W/r.width))});
svg.addEventListener('pointerleave',function(){show(-1)});
svg.addEventListener('keydown',function(e){if(e.key!=='ArrowLeft'&&e.key!=='ArrowRight'&&e.key!=='Escape')return;e.preventDefault();if(e.key==='Escape'){show(-1);return}
 var i=hi<0?(obs.length?obs.length-1:0):hi+(e.key==='ArrowRight'?1:-1);i=Math.max(0,Math.min(all.length-1,i));while(i>0&&all[i].t<cur[0])i++;show(i)});
build();placeInd(btns[0]);reveal();
var rt=null;window.addEventListener('resize',function(){clearTimeout(rt);rt=setTimeout(function(){build();placeInd(btns.filter(function(b){return b.getAttribute('aria-pressed')==='true'})[0])},120)});
})();
</script></body></html>"""


def forecast_chart_src(hist: pd.DataFrame, series: dict, target_label: str, direction: str, recent_points: int) -> str:
    return data_url(CHART_TEMPLATE.replace("__DATA__", safe_json(forecast_payload(hist, series, target_label, direction, recent_points))))
