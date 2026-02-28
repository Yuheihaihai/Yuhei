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

HTML = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>X Virality Engine</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh}
.container{max-width:600px;margin:0 auto;padding:16px}
h1{font-size:20px;text-align:center;padding:16px 0;color:#58a6ff}
.subtitle{text-align:center;color:#8b949e;font-size:13px;margin-bottom:20px}
textarea{width:100%;height:120px;background:#161b22;border:1px solid #30363d;border-radius:12px;color:#e6edf3;padding:14px;font-size:16px;resize:vertical;font-family:inherit}
textarea:focus{outline:none;border-color:#58a6ff}
.row{display:flex;gap:8px;margin-top:10px}
.row input{flex:1;background:#161b22;border:1px solid #30363d;border-radius:10px;color:#e6edf3;padding:10px 12px;font-size:14px}
.row input:focus{outline:none;border-color:#58a6ff}
.btn{width:100%;padding:14px;background:#238636;color:#fff;border:none;border-radius:12px;font-size:16px;font-weight:600;margin-top:14px;cursor:pointer;transition:background .2s}
.btn:hover{background:#2ea043}
.btn:active{background:#1a7f37}
.btn:disabled{background:#21262d;color:#484f58;cursor:not-allowed}
.spinner{display:none;text-align:center;padding:30px;color:#8b949e;font-size:14px}
.result{display:none;margin-top:20px}
.score-card{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:18px;margin-bottom:14px}
.score-main{display:flex;align-items:center;gap:14px;margin-bottom:14px}
.score-circle{width:72px;height:72px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:700;flex-shrink:0}
.score-info{flex:1}
.score-info .cat{font-size:15px;font-weight:600;margin-bottom:2px}
.score-info .imp{font-size:12px;color:#8b949e}
.bar-row{display:flex;align-items:center;margin:6px 0}
.bar-label{width:110px;font-size:11px;color:#8b949e;flex-shrink:0}
.bar-bg{flex:1;height:8px;background:#21262d;border-radius:4px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px;transition:width .6s ease}
.bar-val{width:36px;text-align:right;font-size:11px;color:#8b949e;margin-left:6px}
.section-title{font-size:13px;font-weight:600;color:#58a6ff;margin:14px 0 8px;padding-top:10px;border-top:1px solid #21262d}
.weakness{background:#1c1208;border:1px solid #9e6a03;border-radius:8px;padding:8px 12px;margin:4px 0;font-size:12px;color:#f0c674}
.rec{background:#0d1f0d;border:1px solid #238636;border-radius:8px;padding:10px 12px;margin:6px 0}
.rec-type{font-size:11px;font-weight:600;color:#3fb950;text-transform:uppercase}
.rec-detail{font-size:13px;color:#e6edf3;margin-top:4px}
.rec-delta{font-size:11px;color:#8b949e;margin-top:2px}
.hook{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 12px;margin:4px 0;font-size:13px;cursor:pointer;transition:border-color .2s}
.hook:active{border-color:#58a6ff}
.meta-row{display:flex;justify-content:space-between;font-size:12px;color:#8b949e;padding:4px 0}
.copied{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#238636;color:#fff;padding:8px 20px;border-radius:8px;font-size:14px;display:none;z-index:99}
</style>
</head>
<body>
<div class="container">
  <h1>X Virality Engine</h1>
  <p class="subtitle">Post your draft. Get a virality score + strategy.</p>

  <textarea id="text" placeholder="Enter your X post here..."></textarea>
  <div class="row">
    <input id="followers" type="number" placeholder="Followers (10000)" value="10000">
    <input id="engrate" type="number" step="0.001" placeholder="Eng rate (0.02)" value="0.02">
  </div>
  <button class="btn" id="analyzeBtn" onclick="analyze()">Analyze</button>

  <div class="spinner" id="spinner">Analyzing...</div>
  <div class="result" id="result"></div>
</div>

<div class="copied" id="copied">Copied!</div>

<script>
async function analyze(){
  const text=document.getElementById('text').value.trim();
  if(!text)return;
  const btn=document.getElementById('analyzeBtn');
  const spinner=document.getElementById('spinner');
  const result=document.getElementById('result');
  btn.disabled=true;
  spinner.style.display='block';
  result.style.display='none';
  try{
    const resp=await fetch('/api/analyze',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        text:text,
        followers:parseInt(document.getElementById('followers').value)||10000,
        engagement_rate:parseFloat(document.getElementById('engrate').value)||0.02
      })
    });
    const d=await resp.json();
    result.innerHTML=renderResult(d);
    result.style.display='block';
  }catch(e){result.innerHTML='<p style="color:#f85149">Error: '+e.message+'</p>';result.style.display='block';}
  finally{btn.disabled=false;spinner.style.display='none';}
}

function barColor(v){
  if(v>=0.7)return'#3fb950';
  if(v>=0.5)return'#d29922';
  if(v>=0.3)return'#db6d28';
  return'#f85149';
}

function scoreColor(v){
  if(v>=0.85)return'#238636';
  if(v>=0.7)return'#2ea043';
  if(v>=0.5)return'#9e6a03';
  if(v>=0.3)return'#db6d28';
  return'#da3633';
}

function catIcon(c){
  return{high_potential:'🟢',medium_potential:'🟡',moderate:'🟠',low_potential:'🔴'}[c]||'⚪';
}

function renderResult(d){
  const s=d.score;
  let h='<div class="score-card">';
  h+='<div class="score-main">';
  h+='<div class="score-circle" style="background:'+scoreColor(s.final_score)+'">'+(s.final_score*100).toFixed(0)+'</div>';
  h+='<div class="score-info"><div class="cat">'+catIcon(s.category)+' '+s.category.replace('_',' ')+'</div>';
  h+='<div class="imp">p50: '+Number(s.predicted_impressions_p50).toLocaleString()+' imp</div>';
  h+='<div class="imp">p90: '+Number(s.predicted_impressions_p90).toLocaleString()+' imp</div>';
  h+='</div></div>';

  const feats=s.feature_breakdown;
  for(const[k,v]of Object.entries(feats)){
    h+='<div class="bar-row"><span class="bar-label">'+k.replace(/_/g,' ')+'</span>';
    h+='<div class="bar-bg"><div class="bar-fill" style="width:'+(v*100)+'%;background:'+barColor(v)+'"></div></div>';
    h+='<span class="bar-val">'+(v*100).toFixed(0)+'</span></div>';
  }
  h+='</div>';

  if(s.weaknesses&&s.weaknesses.length){
    h+='<div class="section-title">Weaknesses</div>';
    for(const w of s.weaknesses)h+='<div class="weakness">⚠ '+w+'</div>';
  }

  if(d.recommendations&&d.recommendations.length){
    h+='<div class="section-title">Recommendations</div>';
    for(const r of d.recommendations){
      h+='<div class="rec"><div class="rec-type">'+r.type+'</div>';
      if(r.type==='rewrite'&&r.detail.rewritten){
        h+='<div class="rec-detail">→ '+r.detail.rewritten+'</div>';
      }else if(r.type==='timing'&&r.detail.optimal_windows){
        for(const w of r.detail.optimal_windows.slice(0,3)){
          h+='<div class="rec-detail">'+w[0]+' '+w[1]+':00 UTC ('+w[2]+')</div>';
        }
      }else if(r.type==='format'&&r.detail.suggestions){
        for(const sg of r.detail.suggestions)h+='<div class="rec-detail">→ '+sg.action+'</div>';
      }else if(r.type==='trigger'&&r.detail.triggers){
        for(const t of r.detail.triggers)h+='<div class="rec-detail">→ '+t+'</div>';
      }
      h+='<div class="rec-delta">score delta: '+(r.score_delta>=0?'+':'')+r.score_delta.toFixed(3)+'</div>';
      h+='</div>';
    }
  }

  if(d.hook_variants&&d.hook_variants.length){
    h+='<div class="section-title">Hook Variants (tap to copy)</div>';
    for(const hk of d.hook_variants){
      h+='<div class="hook" onclick="copyHook(this)">'+hk+'</div>';
    }
  }

  h+='<div class="section-title">Details</div>';
  h+='<div class="meta-row"><span>Media</span><span>'+d.media_recommendation+'</span></div>';
  h+='<div class="meta-row"><span>CTA type</span><span>'+d.engagement_bait_type+'</span></div>';
  h+='<div class="meta-row"><span>Thread</span><span>'+(d.thread_expansion.recommended?'Yes ('+d.thread_expansion.optimal_length+' posts)':'Not recommended')+'</span></div>';

  return h;
}

function copyHook(el){
  navigator.clipboard.writeText(el.textContent).then(()=>{
    const c=document.getElementById('copied');
    c.style.display='block';
    setTimeout(()=>c.style.display='none',1200);
  });
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
