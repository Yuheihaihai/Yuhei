#!/usr/bin/env python3
"""X Virality Optimization Engine — Mobile Web UI"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import yaml
from flask import Flask, request, jsonify, render_template_string

from src.core.virality_engine import PostInput, ViralityEngine

app = Flask(__name__)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        config = {}
        if os.path.exists("config.yaml"):
            with open("config.yaml") as f:
                config = yaml.safe_load(f) or {}
        _engine = ViralityEngine(config)
    return _engine


HTML = r"""
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#000000">
<title>Virality Engine</title>
<style>
/* ===== Reset & Foundation ===== */
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
html{font-size:16px;-webkit-text-size-adjust:100%}
body{
  font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display','Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  background:#000;color:#f5f5f7;min-height:100vh;min-height:100dvh;
  overscroll-behavior:none;
}

/* ===== Layout Shell ===== */
.app{max-width:480px;margin:0 auto;padding:0 20px 100px}

/* ===== Header ===== */
.header{padding:52px 0 8px;text-align:center}
.header h1{font-size:28px;font-weight:700;letter-spacing:-.5px;color:#fff}
.header p{font-size:14px;color:#86868b;margin-top:6px;line-height:1.4}

/* ===== Input Area ===== */
.input-area{margin-top:24px}
.textarea-wrap{position:relative}
.textarea-wrap textarea{
  width:100%;min-height:130px;background:#1c1c1e;border:none;border-radius:16px;
  color:#f5f5f7;padding:16px 16px 40px;font-size:17px;line-height:1.5;resize:none;
  font-family:inherit;transition:box-shadow .2s;
}
.textarea-wrap textarea::placeholder{color:#636366}
.textarea-wrap textarea:focus{outline:none;box-shadow:0 0 0 2px #0a84ff}
.char-count{position:absolute;bottom:12px;right:16px;font-size:12px;color:#636366;pointer-events:none}
.char-count.warn{color:#ff9f0a}
.char-count.over{color:#ff453a}

/* ===== Settings Row ===== */
.settings-toggle{
  display:flex;align-items:center;justify-content:center;gap:6px;
  margin-top:12px;font-size:13px;color:#86868b;cursor:pointer;
  -webkit-user-select:none;user-select:none;padding:4px 0;
}
.settings-toggle svg{width:14px;height:14px;fill:#86868b;transition:transform .2s}
.settings-toggle.open svg{transform:rotate(180deg)}
.settings-panel{
  display:grid;grid-template-columns:1fr 1fr;gap:8px;
  max-height:0;overflow:hidden;transition:max-height .25s ease,margin .25s ease;
}
.settings-panel.open{max-height:80px;margin-top:10px}
.settings-panel input{
  background:#1c1c1e;border:none;border-radius:12px;color:#f5f5f7;
  padding:12px 14px;font-size:15px;font-family:inherit;width:100%;
}
.settings-panel input:focus{outline:none;box-shadow:0 0 0 2px #0a84ff}
.settings-panel input::placeholder{color:#636366}

/* ===== Analyze Button ===== */
.analyze-btn{
  width:100%;padding:16px;margin-top:16px;border:none;border-radius:14px;
  font-size:17px;font-weight:600;font-family:inherit;cursor:pointer;
  background:#0a84ff;color:#fff;transition:all .15s;
  position:relative;overflow:hidden;
}
.analyze-btn:hover{background:#0071e3}
.analyze-btn:active{transform:scale(.98);background:#0068d6}
.analyze-btn:disabled{background:#1c1c1e;color:#48484a;cursor:default;transform:none}

/* ===== Loading ===== */
.loading{display:none;text-align:center;padding:40px 0}
.loading-ring{
  display:inline-block;width:32px;height:32px;
  border:3px solid #333;border-top-color:#0a84ff;border-radius:50%;
  animation:spin .8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-text{font-size:14px;color:#86868b;margin-top:12px}

/* ===== Results ===== */
.results{display:none;margin-top:28px;animation:fadeUp .4s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}

/* ===== Score Hero ===== */
.score-hero{text-align:center;padding:28px 0}
.ring-container{position:relative;width:140px;height:140px;margin:0 auto}
.ring-container svg{width:140px;height:140px;transform:rotate(-90deg)}
.ring-bg{fill:none;stroke:#2c2c2e;stroke-width:10}
.ring-fill{fill:none;stroke-width:10;stroke-linecap:round;transition:stroke-dashoffset 1s ease}
.ring-score{
  position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  font-size:36px;font-weight:700;letter-spacing:-1px;
}
.ring-label{font-size:12px;color:#86868b;font-weight:400}
.score-category{
  display:inline-flex;align-items:center;gap:6px;margin-top:14px;
  padding:6px 16px;border-radius:20px;font-size:14px;font-weight:500;
}
.impressions-row{
  display:flex;justify-content:center;gap:20px;margin-top:16px;
}
.imp-item{text-align:center}
.imp-item .val{font-size:18px;font-weight:600}
.imp-item .label{font-size:11px;color:#86868b;margin-top:2px}

/* ===== Feature Cards ===== */
.section-label{
  font-size:13px;font-weight:600;color:#86868b;text-transform:uppercase;
  letter-spacing:.8px;margin:28px 0 12px;
}
.features-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.feat-card{
  background:#1c1c1e;border-radius:14px;padding:14px;
  transition:transform .15s;
}
.feat-card:active{transform:scale(.97)}
.feat-card .feat-name{font-size:11px;color:#86868b;text-transform:capitalize;margin-bottom:8px}
.feat-card .feat-val{font-size:22px;font-weight:600;margin-bottom:8px}
.feat-bar{height:4px;background:#2c2c2e;border-radius:2px;overflow:hidden}
.feat-bar-fill{height:100%;border-radius:2px;transition:width .8s ease}

/* ===== Weakness Pills ===== */
.weakness-list{display:flex;flex-direction:column;gap:6px}
.weakness-pill{
  display:flex;align-items:flex-start;gap:8px;
  background:#1c1c1e;border-radius:12px;padding:12px 14px;
  font-size:13px;line-height:1.4;color:#ff9f0a;
}
.weakness-pill .icon{flex-shrink:0;font-size:14px;margin-top:1px}

/* ===== Recommendation Cards ===== */
.rec-list{display:flex;flex-direction:column;gap:8px}
.rec-card{
  background:#1c1c1e;border-radius:14px;padding:14px 16px;
  border-left:3px solid #30d158;
}
.rec-card .rec-header{display:flex;align-items:center;justify-content:space-between}
.rec-card .rec-tag{
  font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.5px;color:#30d158;
}
.rec-card .rec-delta{font-size:12px;color:#86868b}
.rec-card .rec-body{font-size:14px;color:#f5f5f7;margin-top:8px;line-height:1.5}
.rec-card .rec-body .original{color:#86868b;text-decoration:line-through;font-size:13px}
.rec-card .rec-body .rewritten{color:#30d158;font-weight:500}
.rec-card.timing{border-left-color:#0a84ff}
.rec-card.timing .rec-tag{color:#0a84ff}
.rec-card.format{border-left-color:#bf5af2}
.rec-card.format .rec-tag{color:#bf5af2}
.rec-card.trigger{border-left-color:#ff9f0a}
.rec-card.trigger .rec-tag{color:#ff9f0a}

/* ===== Hook Variants ===== */
.hooks-list{display:flex;flex-direction:column;gap:6px}
.hook-card{
  background:#1c1c1e;border-radius:14px;padding:14px 16px;
  font-size:14px;line-height:1.5;cursor:pointer;
  display:flex;align-items:center;gap:10px;
  transition:background .15s,transform .15s;
  -webkit-user-select:none;user-select:none;
}
.hook-card:active{background:#2c2c2e;transform:scale(.98)}
.hook-num{
  width:24px;height:24px;border-radius:50%;background:#0a84ff;
  display:flex;align-items:center;justify-content:center;
  font-size:12px;font-weight:600;flex-shrink:0;
}
.hook-text{flex:1}
.hook-copy{
  font-size:12px;color:#0a84ff;white-space:nowrap;flex-shrink:0;
}

/* ===== Meta Details ===== */
.meta-grid{
  display:grid;grid-template-columns:1fr 1fr;gap:8px;
}
.meta-item{
  background:#1c1c1e;border-radius:14px;padding:14px;
}
.meta-item .meta-label{font-size:11px;color:#86868b;margin-bottom:4px}
.meta-item .meta-value{font-size:14px;font-weight:500}

/* ===== Toast ===== */
.toast{
  position:fixed;bottom:max(24px,env(safe-area-inset-bottom,24px));
  left:50%;transform:translateX(-50%) translateY(80px);
  background:#30d158;color:#000;font-weight:600;font-size:14px;
  padding:10px 24px;border-radius:100px;
  transition:transform .3s cubic-bezier(.4,0,.2,1);z-index:100;
  pointer-events:none;white-space:nowrap;
}
.toast.show{transform:translateX(-50%) translateY(0)}

/* ===== History ===== */
.history{margin-top:32px}
.history-item{
  display:flex;align-items:center;gap:12px;
  background:#1c1c1e;border-radius:14px;padding:14px 16px;
  margin-bottom:6px;cursor:pointer;transition:background .15s;
}
.history-item:active{background:#2c2c2e}
.history-score{
  width:40px;height:40px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-size:13px;font-weight:700;flex-shrink:0;
}
.history-text{flex:1;font-size:13px;color:#f5f5f7;line-height:1.3;overflow:hidden;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
}
.history-clear{
  font-size:13px;color:#ff453a;text-align:center;padding:8px;cursor:pointer;
  -webkit-user-select:none;user-select:none;margin-top:4px;
}
</style>
</head>
<body>
<div class="app">
  <!-- Header -->
  <div class="header">
    <h1>Virality Engine</h1>
    <p>Analyze your X post. Get a virality score and optimization strategy.</p>
  </div>

  <!-- Input -->
  <div class="input-area">
    <div class="textarea-wrap">
      <textarea id="text" placeholder="Write your post..." oninput="updateCharCount()"></textarea>
      <div class="char-count" id="charCount">0 / 280</div>
    </div>
    <div class="settings-toggle" id="settingsToggle" onclick="toggleSettings()">
      Settings
      <svg viewBox="0 0 20 20"><path d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z"/></svg>
    </div>
    <div class="settings-panel" id="settingsPanel">
      <input id="followers" type="number" placeholder="Followers" value="10000">
      <input id="engrate" type="number" step="0.001" placeholder="Eng rate" value="0.02">
    </div>
    <button class="analyze-btn" id="analyzeBtn" onclick="analyze()">Analyze Post</button>
  </div>

  <!-- Loading -->
  <div class="loading" id="loading">
    <div class="loading-ring"></div>
    <div class="loading-text">Analyzing your post...</div>
  </div>

  <!-- Results -->
  <div class="results" id="results"></div>

  <!-- History -->
  <div class="history" id="historySection" style="display:none">
    <div class="section-label">Recent</div>
    <div id="historyList"></div>
    <div class="history-clear" onclick="clearHistory()">Clear History</div>
  </div>
</div>

<!-- Toast -->
<div class="toast" id="toast">Copied!</div>

<script>
/* ===== State ===== */
let history=JSON.parse(localStorage.getItem('ve_history')||'[]');
renderHistory();

/* ===== Character Counter ===== */
function updateCharCount(){
  const ta=document.getElementById('text');
  const cc=document.getElementById('charCount');
  const len=ta.value.length;
  cc.textContent=len+' / 280';
  cc.className='char-count'+(len>280?' over':len>240?' warn':'');
}

/* ===== Settings ===== */
function toggleSettings(){
  const p=document.getElementById('settingsPanel');
  const t=document.getElementById('settingsToggle');
  p.classList.toggle('open');
  t.classList.toggle('open');
}

/* ===== Analyze ===== */
async function analyze(){
  const text=document.getElementById('text').value.trim();
  if(!text)return;
  const btn=document.getElementById('analyzeBtn');
  const loading=document.getElementById('loading');
  const results=document.getElementById('results');
  btn.disabled=true;btn.textContent='Analyzing...';
  loading.style.display='block';results.style.display='none';
  try{
    const resp=await fetch('/api/analyze',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        text,
        followers:parseInt(document.getElementById('followers').value)||10000,
        engagement_rate:parseFloat(document.getElementById('engrate').value)||0.02
      })
    });
    const d=await resp.json();
    results.innerHTML=renderResult(d);
    results.style.display='block';
    saveHistory(text,d.score.final_score,d.score.category);
    // Trigger ring animation after render
    requestAnimationFrame(()=>{
      const ring=document.querySelector('.ring-fill');
      if(ring)ring.style.strokeDashoffset=ring.dataset.target;
    });
  }catch(e){
    results.innerHTML='<div style="text-align:center;padding:20px;color:#ff453a">'+e.message+'</div>';
    results.style.display='block';
  }finally{
    btn.disabled=false;btn.textContent='Analyze Post';loading.style.display='none';
  }
}

/* ===== Colors ===== */
function featColor(v){
  if(v>=.7)return'#30d158';if(v>=.5)return'#ffd60a';
  if(v>=.3)return'#ff9f0a';return'#ff453a';
}
function scoreGradient(v){
  if(v>=.85)return['#30d158','#1c3829'];
  if(v>=.7)return['#30d158','#1c3829'];
  if(v>=.5)return['#ffd60a','#332d1a'];
  if(v>=.3)return['#ff9f0a','#33261a'];
  return['#ff453a','#331a1a'];
}
function catLabel(c){
  return{high_potential:'High Potential',medium_potential:'Medium Potential',moderate:'Moderate',low_potential:'Low Potential'}[c]||c;
}

/* ===== Render ===== */
function renderResult(d){
  const s=d.score;
  const pct=s.final_score;
  const[ringCol,catBg]=scoreGradient(pct);
  const circ=2*Math.PI*60;
  const offset=circ*(1-pct);

  let h='';
  // -- Score Hero --
  h+='<div class="score-hero">';
  h+='<div class="ring-container">';
  h+='<svg viewBox="0 0 140 140">';
  h+='<circle class="ring-bg" cx="70" cy="70" r="60"/>';
  h+='<circle class="ring-fill" cx="70" cy="70" r="60" stroke="'+ringCol+'" stroke-dasharray="'+circ+'" stroke-dashoffset="'+circ+'" data-target="'+offset+'"/>';
  h+='</svg>';
  h+='<div class="ring-score" style="color:'+ringCol+'">'+(pct*100).toFixed(0)+'<br><span class="ring-label">score</span></div>';
  h+='</div>';
  h+='<div class="score-category" style="background:'+catBg+';color:'+ringCol+'">'+catLabel(s.category)+'</div>';
  h+='<div class="impressions-row">';
  h+='<div class="imp-item"><div class="val">'+fmtNum(s.predicted_impressions_p50)+'</div><div class="label">p50 impressions</div></div>';
  h+='<div class="imp-item"><div class="val">'+fmtNum(s.predicted_impressions_p75)+'</div><div class="label">p75</div></div>';
  h+='<div class="imp-item"><div class="val">'+fmtNum(s.predicted_impressions_p90)+'</div><div class="label">p90</div></div>';
  h+='</div></div>';

  // -- Features --
  h+='<div class="section-label">Feature Breakdown</div>';
  h+='<div class="features-grid">';
  const feats=s.feature_breakdown;
  const fnames={hook_strength:'Hook',emotional_intensity:'Emotion',curiosity_gap:'Curiosity',
    controversy_index:'Controversy',readability_score:'Readability',structure_score:'Structure',
    timing_score:'Timing',trend_alignment:'Trend'};
  for(const[k,v]of Object.entries(feats)){
    const c=featColor(v);
    h+='<div class="feat-card"><div class="feat-name">'+(fnames[k]||k)+'</div>';
    h+='<div class="feat-val" style="color:'+c+'">'+(v*100).toFixed(0)+'</div>';
    h+='<div class="feat-bar"><div class="feat-bar-fill" style="width:'+(v*100)+'%;background:'+c+'"></div></div>';
    h+='</div>';
  }
  h+='</div>';

  // -- Weaknesses --
  if(s.weaknesses&&s.weaknesses.length){
    h+='<div class="section-label">Weaknesses</div><div class="weakness-list">';
    for(const w of s.weaknesses){
      const short=w.replace(/ \(below.*$/,'');
      h+='<div class="weakness-pill"><span class="icon">!</span><span>'+short+'</span></div>';
    }
    h+='</div>';
  }

  // -- Recommendations --
  if(d.recommendations&&d.recommendations.length){
    h+='<div class="section-label">Recommendations</div><div class="rec-list">';
    for(const r of d.recommendations){
      const cls=r.type==='timing'?'timing':r.type==='format'?'format':r.type==='trigger'?'trigger':'';
      h+='<div class="rec-card '+cls+'">';
      h+='<div class="rec-header"><span class="rec-tag">'+r.type+'</span>';
      h+='<span class="rec-delta">'+(r.score_delta>=0?'+':'')+r.score_delta.toFixed(3)+'</span></div>';
      h+='<div class="rec-body">';
      if(r.type==='rewrite'&&r.detail.rewritten){
        h+='<div class="original">'+escHtml(r.detail.original||'')+'</div>';
        h+='<div class="rewritten">'+escHtml(r.detail.rewritten)+'</div>';
      }else if(r.type==='timing'&&r.detail.optimal_windows){
        for(const w of r.detail.optimal_windows.slice(0,3))
          h+='<div>'+w[0]+' '+w[1]+':00 UTC &middot; '+w[2]+'</div>';
      }else if(r.type==='format'&&r.detail.suggestions){
        for(const sg of r.detail.suggestions)h+='<div>'+escHtml(sg.action)+'</div>';
      }else if(r.type==='trigger'&&r.detail.triggers){
        for(const t of r.detail.triggers)h+='<div>'+escHtml(t)+'</div>';
      }
      h+='</div></div>';
    }
    h+='</div>';
  }

  // -- Hook Variants --
  if(d.hook_variants&&d.hook_variants.length){
    h+='<div class="section-label">Hook Variants</div><div class="hooks-list">';
    d.hook_variants.forEach((hk,i)=>{
      h+='<div class="hook-card" onclick="copyText(\''+escJs(hk)+'\')">';
      h+='<div class="hook-num">'+(i+1)+'</div>';
      h+='<div class="hook-text">'+escHtml(hk)+'</div>';
      h+='<div class="hook-copy">Copy</div>';
      h+='</div>';
    });
    h+='</div>';
  }

  // -- Meta --
  h+='<div class="section-label">Details</div><div class="meta-grid">';
  h+='<div class="meta-item"><div class="meta-label">Media</div><div class="meta-value">'+escHtml(d.media_recommendation).replace(/_/g,' ')+'</div></div>';
  h+='<div class="meta-item"><div class="meta-label">CTA Type</div><div class="meta-value">'+escHtml(d.engagement_bait_type).replace(/_/g,' ')+'</div></div>';
  h+='<div class="meta-item"><div class="meta-label">Thread</div><div class="meta-value">'+(d.thread_expansion.recommended?d.thread_expansion.optimal_length+' posts':'Not needed')+'</div></div>';
  h+='<div class="meta-item"><div class="meta-label">Hashtags</div><div class="meta-value">'+(d.hashtag_strategy.tags&&d.hashtag_strategy.tags.length?d.hashtag_strategy.tags.join(' '):'None')+'</div></div>';
  h+='</div>';

  return h;
}

/* ===== Helpers ===== */
function fmtNum(n){
  if(n>=1e6)return(n/1e6).toFixed(1)+'M';
  if(n>=1e3)return(n/1e3).toFixed(1)+'K';
  return String(n);
}
function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function escJs(s){return s.replace(/\\/g,'\\\\').replace(/'/g,"\\'").replace(/"/g,'\\"');}

function copyText(text){
  navigator.clipboard.writeText(text).then(()=>{showToast('Copied!')}).catch(()=>{});
}
function showToast(msg){
  const t=document.getElementById('toast');
  t.textContent=msg;t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),1500);
}

/* ===== History ===== */
function saveHistory(text,score,category){
  history.unshift({text:text.substring(0,100),score,category,ts:Date.now()});
  if(history.length>20)history=history.slice(0,20);
  localStorage.setItem('ve_history',JSON.stringify(history));
  renderHistory();
}
function renderHistory(){
  const sec=document.getElementById('historySection');
  const list=document.getElementById('historyList');
  if(!history.length){sec.style.display='none';return;}
  sec.style.display='block';
  let h='';
  for(const item of history.slice(0,5)){
    const[c]=scoreGradient(item.score);
    h+='<div class="history-item" onclick="loadHistory(\''+escJs(item.text)+'\')">';
    h+='<div class="history-score" style="background:'+c+'22;color:'+c+'">'+(item.score*100).toFixed(0)+'</div>';
    h+='<div class="history-text">'+escHtml(item.text)+'</div>';
    h+='</div>';
  }
  list.innerHTML=h;
}
function loadHistory(text){
  document.getElementById('text').value=text;
  updateCharCount();
  window.scrollTo({top:0,behavior:'smooth'});
}
function clearHistory(){
  history=[];localStorage.removeItem('ve_history');renderHistory();
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "text is required"}), 400

    engine = get_engine()
    post = PostInput(
        text=text,
        author_follower_count=data.get("followers", 10000),
        author_engagement_rate=data.get("engagement_rate", 0.02),
        posted_at=datetime.now(timezone.utc).isoformat(),
    )
    result = engine.analyze(post)

    return jsonify({
        "score": {
            "final_score": result.score.final_score,
            "raw_score": result.score.raw_score,
            "predicted_impressions_p50": result.score.predicted_impressions_p50,
            "predicted_impressions_p75": result.score.predicted_impressions_p75,
            "predicted_impressions_p90": result.score.predicted_impressions_p90,
            "feature_breakdown": result.score.feature_breakdown,
            "weaknesses": result.score.weaknesses,
            "category": result.score.category,
        },
        "recommendations": [
            {
                "type": r.rec_type,
                "detail": r.detail,
                "score_delta": r.score_delta,
            }
            for r in result.recommendations
        ],
        "hook_variants": result.hook_variants,
        "thread_expansion": result.thread_expansion,
        "media_recommendation": result.media_recommendation,
        "hashtag_strategy": result.hashtag_strategy,
        "engagement_bait_type": result.engagement_bait_type,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
