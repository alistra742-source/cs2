"""Lightweight live-camera modal injected into the dashboard."""

LIVE_INJECTION = r'''
<style>
#liveOverlay{display:none;position:fixed;inset:0;z-index:200;padding:18px;background:rgba(0,0,0,.72);backdrop-filter:blur(4px);align-items:center;justify-content:center}
#liveOverlay.on{display:flex}
.live-modal{width:min(1100px,100%);max-height:min(820px,calc(100vh - 36px));display:flex;flex-direction:column;gap:10px;padding:16px;background:#131316;border:1px solid #34343a;border-radius:16px;box-shadow:0 26px 80px rgba(0,0,0,.62)}
.live-head{display:flex;align-items:center;gap:10px;min-width:0}
.live-title{font-family:'JetBrains Mono','Courier New',monospace;font-size:13px;font-weight:700;letter-spacing:1.5px;color:#e7e7ea}
.live-title b{color:#34d399}
.live-state{margin-left:auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#8a8a92;font-family:'JetBrains Mono','Courier New',monospace;font-size:11px}
.live-close{padding:7px 11px;border-radius:9px;border:1px solid #5a2323;background:#2a1212;color:#fca5a5;cursor:pointer;font-weight:700}
.live-tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.live-tools button{padding:7px 12px;border-radius:9px;border:1px solid #323547;background:#1a1a22;color:#e7e7ea;cursor:pointer;font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;letter-spacing:.4px}
.live-tools button.on{background:#059669;border-color:#34d399;color:#fff}
.live-type-bar{display:none}
.live-type-bar.on{display:block}
#liveTypeInput{width:100%;padding:10px 12px;border-radius:10px;border:1px solid #323547;background:#08080a;color:#e7e7ea;font-size:16px}
.live-frame{position:relative;min-height:240px;max-height:calc(100vh - 300px);overflow:auto;display:flex;align-items:flex-start;justify-content:center;background:#050506;border:1px solid #34343a;border-radius:12px}
.live-frame img#liveImage{display:none;width:100%;height:auto;object-fit:contain;user-select:none;-webkit-user-drag:none;touch-action:none}
.live-placeholder{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;padding:24px;text-align:center;color:#8a8a92;font-family:'JetBrains Mono','Courier New',monospace;font-size:12px;line-height:1.6}
.live-hud{position:absolute;left:10px;top:10px;z-index:3;padding:4px 8px;border-radius:7px;background:rgba(6,6,8,.78);border:1px solid #2a2a32;color:#34d399;font-family:'JetBrains Mono','Courier New',monospace;font-size:11px;letter-spacing:.2px;pointer-events:none}
.live-last{position:absolute;right:10px;top:10px;z-index:3;max-width:62%;padding:4px 8px;border-radius:7px;background:rgba(6,6,8,.78);border:1px solid #2a2a32;color:#e7e7ea;font-family:'JetBrains Mono','Courier New',monospace;font-size:10px;pointer-events:none;text-align:right}
.live-trail{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2}
.live-mark{position:absolute;z-index:4;width:14px;height:14px;margin:-7px 0 0 -7px;border:2px solid #34d399;border-radius:50%;box-shadow:0 0 0 2px rgba(52,211,153,.25);pointer-events:none;display:none}
.live-mark.on{display:block}
.live-hit{display:flex;gap:8px;align-items:stretch}
#liveHitField{flex:1;min-width:0;min-height:72px;max-height:120px;resize:vertical;padding:8px 10px;border-radius:10px;border:1px solid #26262b;background:#08080a;color:#d4d4d8;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.45}
.live-challenge-wrap{display:none;gap:10px;align-items:stretch}
.live-challenge-wrap.on{display:flex}
.live-challenge-shot{flex:0 0 auto;max-width:220px;max-height:160px;border-radius:10px;border:1px solid #33374b;background:#050506;object-fit:contain}
.live-pointer-log{flex:1;min-width:0;max-height:160px;overflow:auto;padding:8px 10px;background:#08080a;border:1px solid #26262b;border-radius:10px;color:#a1a1aa;font-family:'JetBrains Mono','Courier New',monospace;font-size:11px;line-height:1.55;white-space:pre-wrap}
.live-foot{color:#5c5c64;font-family:'JetBrains Mono','Courier New',monospace;font-size:10px;letter-spacing:.3px}
@media(max-width:640px){#liveOverlay{padding:10px}.live-modal{padding:12px;border-radius:13px}.live-state{display:none}.live-frame{min-height:200px;max-height:calc(100vh - 340px)}.live-challenge-wrap.on{flex-direction:column}.live-challenge-shot{max-width:100%}.live-hit{flex-direction:column}}
</style>
<div id="liveOverlay" role="dialog" aria-modal="true" aria-labelledby="liveTitle">
  <div class="live-modal">
    <div class="live-head">
      <div id="liveTitle" class="live-title"><b>LIVE</b> CAMERA</div>
      <div id="liveState" class="live-state">Waiting for browser</div>
      <button class="live-close" type="button" onclick="closeLive()" aria-label="Close live camera">CLOSE</button>
    </div>
    <div class="live-tools">
      <button id="liveKeyBtn" type="button" aria-pressed="false">KEY</button>
      <button id="liveDragBtn" type="button" aria-pressed="false">DRAG</button>
      <button id="liveRegBtn" type="button">REGISTER</button>
      <button id="liveCopyAll" type="button">COPY ALL</button>
    </div>
    <div id="liveTypeBar" class="live-type-bar">
      <input id="liveTypeInput" type="text" inputmode="text" autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false" placeholder="Type into Chrome">
    </div>
    <div class="live-frame" id="liveFrame">
      <div id="livePlaceholder" class="live-placeholder">Tap REGISTER to open Discord signup, or start the runner.</div>
      <img id="liveImage" alt="Latest real Chrome camera frame" draggable="false">
      <svg id="liveTrail" class="live-trail" aria-hidden="true"></svg>
      <div id="liveMark" class="live-mark" aria-hidden="true"></div>
      <div id="liveHud" class="live-hud">—</div>
      <div id="liveLast" class="live-last"></div>
    </div>
    <div class="live-hit">
      <textarea id="liveHitField" readonly placeholder="Selector and JS of the last click land here."></textarea>
    </div>
    <div id="liveChallengeWrap" class="live-challenge-wrap">
      <img id="liveChallengeImg" class="live-challenge-shot" alt="Latest hCaptcha challenge screenshot">
      <div id="livePointerLog" class="live-pointer-log"></div>
    </div>
    <div class="live-foot">Real Chrome camera — refreshes every 3 seconds. REGISTER opens discord.com/register. KEY opens your phone keyboard. DRAG on = drag, off = click only. Challenge shots are saved and kept.</div>
  </div>
</div>
<script>
var LC={worker:'B1',timer:null,interactive:false,connected:false,dsf:1,drag:null,dragOn:false,keyOn:false,typeLast:'',lastSrc:'',lastChallenge:'',pointerLog:[]};
function lcImgSrc(src){
  src=String(src||'');
  if(!src)return '';
  if(src.indexOf('data:image/')===0||src.indexOf('/challenges/')===0||src.indexOf('http')===0)return src;
  return 'data:image/png;base64,'+src;
}

function lcSetStatus(message){
  var el=document.getElementById('liveState');
  if(el)el.textContent=message;
}
function lcShowPlaceholder(message){
  var img=document.getElementById('liveImage');
  var ph=document.getElementById('livePlaceholder');
  if(img)img.style.display='none';
  if(ph){ph.textContent=message;ph.style.display='flex';}
}
function lcSetImage(src){
  var img=document.getElementById('liveImage');
  var ph=document.getElementById('livePlaceholder');
  if(!img||!src)return;
  if(src===LC.lastSrc && img.style.display==='block')return;
  img.onload=function(){
    img.style.display='block';
    if(ph)ph.style.display='none';
  };
  img.onerror=function(){lcShowPlaceholder('Waiting for the first camera frame.');};
  var next=lcImgSrc(src);
  LC.lastSrc=src;
  img.src=next;
}
function lcSetHud(text){
  var el=document.getElementById('liveHud');
  if(el)el.textContent=text||'—';
}
function lcSetLast(text){
  var el=document.getElementById('liveLast');
  if(el)el.textContent=text||'';
}
function lcFmt(n){return Math.round(Number(n)||0);}
function lcDescribe(p){
  if(!p||!p.kind)return '';
  if(p.kind==='click'){
    var line='clicked '+lcFmt(p.x)+', '+lcFmt(p.y);
    if(p.selector)line+='  '+p.selector;
    return line;
  }
  if(p.kind==='drag')return 'drag '+lcFmt(p.x1)+','+lcFmt(p.y1)+' → '+lcFmt(p.x2)+','+lcFmt(p.y2);
  if(p.kind==='mousedown'||p.kind==='mouseup'||p.kind==='mousemove')
    return p.kind+' '+lcFmt(p.x)+', '+lcFmt(p.y);
  return p.kind;
}
function lcHitText(p){
  if(!p)return '';
  var lines=[lcDescribe(p)];
  if(p.selector)lines.push('selector: '+p.selector);
  if(p.js)lines.push('js: '+p.js);
  if(p.click_js)lines.push('click: '+p.click_js);
  if(p.tag)lines.push('tag: '+p.tag+(p.type?' '+p.type:''));
  if(p.text)lines.push('text: '+p.text);
  return lines.join('\n');
}
function lcSetHitField(p){
  var field=document.getElementById('liveHitField');
  if(!field)return;
  var text=lcHitText(p);
  if(text)field.value=text;
}
function lcRenderPointerLog(items){
  LC.pointerLog=items||[];
  var box=document.getElementById('livePointerLog');
  var wrap=document.getElementById('liveChallengeWrap');
  var shot=document.getElementById('liveChallengeImg');
  var hasShot=!!(shot&&shot.getAttribute('src'));
  var rows=(items||[]).slice(-12).map(function(p){
    var line=(p.t?p.t+' ':'')+lcDescribe(p);
    if(p.js && p.kind==='click')line+='\n  '+p.js;
    return line;
  }).filter(Boolean);
  if(box)box.textContent=rows.length?rows.join('\n'):'Clicks and drags will appear here with page coordinates.';
  if(wrap)wrap.classList.toggle('on', hasShot || rows.length>0);
}
function lcCopyAll(){
  var items=LC.pointerLog||[];
  var field=document.getElementById('liveHitField');
  var text=items.length?items.map(function(p){
    return (p.t?p.t+' ':'')+lcHitText(p);
  }).join('\n\n'):((field&&field.value)||'');
  if(!text){lcSetStatus('Nothing to copy');return;}
  function done(){lcSetStatus('Copied all clicks');}
  function fallback(){
    var ta=document.createElement('textarea');
    ta.value=text;ta.style.position='fixed';ta.style.opacity='0';
    document.body.appendChild(ta);ta.select();
    try{document.execCommand('copy');done();}catch(e){lcSetStatus('Copy failed');}
    document.body.removeChild(ta);
  }
  if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(text).then(done).catch(fallback);
  }else fallback();
}
function lcSetChallenge(src){
  var img=document.getElementById('liveChallengeImg');
  var wrap=document.getElementById('liveChallengeWrap');
  if(!img||!src)return;
  if(src===LC.lastChallenge) return;
  LC.lastChallenge=src;
  img.src=lcImgSrc(src);
  if(wrap)wrap.classList.add('on');
}
function lcPageXY(event){
  var img=document.getElementById('liveImage');
  if(!img||!img.naturalWidth||!img.naturalHeight)return null;
  var rect=img.getBoundingClientRect();
  if(!rect.width||!rect.height)return null;
  var dsf=Number(LC.dsf)||1;
  if(dsf<=0)dsf=1;
  return {
    x:(event.clientX-rect.left)*(img.naturalWidth/rect.width)/dsf,
    y:(event.clientY-rect.top)*(img.naturalHeight/rect.height)/dsf
  };
}
function lcMarkAt(x,y){
  var img=document.getElementById('liveImage');
  var mark=document.getElementById('liveMark');
  if(!img||!mark||!img.naturalWidth)return;
  var dsf=Number(LC.dsf)||1;
  if(dsf<=0)dsf=1;
  mark.style.left=(img.offsetLeft+(Number(x)*dsf)*(img.clientWidth/img.naturalWidth))+'px';
  mark.style.top=(img.offsetTop+(Number(y)*dsf)*(img.clientHeight/img.naturalHeight))+'px';
  mark.classList.add('on');
}
function lcClearTrail(){
  var svg=document.getElementById('liveTrail');
  if(svg){while(svg.firstChild)svg.removeChild(svg.firstChild);}
}
function lcDrawTrail(x1,y1,x2,y2){
  var img=document.getElementById('liveImage');
  var svg=document.getElementById('liveTrail');
  if(!img||!svg||!img.naturalWidth)return;
  var dsf=Number(LC.dsf)||1;
  if(dsf<=0)dsf=1;
  var sx=img.offsetLeft+(x1*dsf)*(img.clientWidth/img.naturalWidth);
  var sy=img.offsetTop+(y1*dsf)*(img.clientHeight/img.naturalHeight);
  var ex=img.offsetLeft+(x2*dsf)*(img.clientWidth/img.naturalWidth);
  var ey=img.offsetTop+(y2*dsf)*(img.clientHeight/img.naturalHeight);
  svg.setAttribute('viewBox','0 0 '+img.offsetWidth+' '+img.offsetHeight);
  svg.setAttribute('width',img.offsetWidth);
  svg.setAttribute('height',img.offsetHeight);
  lcClearTrail();
  var line=document.createElementNS('http://www.w3.org/2000/svg','line');
  line.setAttribute('x1',sx);line.setAttribute('y1',sy);
  line.setAttribute('x2',ex);line.setAttribute('y2',ey);
  line.setAttribute('stroke','#34d399');
  line.setAttribute('stroke-width','2');
  line.setAttribute('stroke-linecap','round');
  svg.appendChild(line);
}
function lcApplyState(st){
  if(!st)return;
  if(st.connected)LC.connected=true;
  if(st.connected)LC.interactive=true;
  if(st.device_scale_factor)LC.dsf=Number(st.device_scale_factor)||1;
  if(st.screenshot)lcSetImage(st.screenshot);
  if(st.challenge_screenshot)lcSetChallenge(st.challenge_screenshot);
  if(st.last_pointer){
    lcSetLast(lcDescribe(st.last_pointer));
    lcSetHitField(st.last_pointer);
    if(st.last_pointer.kind==='click')lcMarkAt(st.last_pointer.x,st.last_pointer.y);
    if(st.last_pointer.kind==='drag')lcMarkAt(st.last_pointer.x2,st.last_pointer.y2);
    if(st.last_pointer.is_input && !LC.keyOn)lcOpenKeyboard();
  }
  if(st.pointer_log)lcRenderPointerLog(st.pointer_log);
}
function lcLoadFrame(){
  fetch('/browser/state?worker='+encodeURIComponent(LC.worker)+'&t='+Date.now())
    .then(function(r){if(!r.ok)throw new Error('browser state unavailable');return r.json();})
    .then(function(st){
      if(st&&st.screenshot)lcApplyState(st);
      else if(!st||!st.connected)lcShowPlaceholder('Waiting for the first camera frame.');
      else lcApplyState(st);
    })
    .catch(function(){lcShowPlaceholder('Waiting for the first camera frame.');});
}
function lcSend(body){
  return fetch('/browser/action?worker='+encodeURIComponent(LC.worker),{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)
  }).then(function(r){return r.json();}).then(function(st){
    lcApplyState(st);
    return st;
  });
}
function lcSetDrag(on){
  LC.dragOn=!!on;
  var btn=document.getElementById('liveDragBtn');
  if(btn){btn.classList.toggle('on',LC.dragOn);btn.setAttribute('aria-pressed',LC.dragOn?'true':'false');}
}
function lcOpenKeyboard(){
  LC.keyOn=true;
  var bar=document.getElementById('liveTypeBar');
  var btn=document.getElementById('liveKeyBtn');
  var input=document.getElementById('liveTypeInput');
  if(bar)bar.classList.add('on');
  if(btn){btn.classList.add('on');btn.setAttribute('aria-pressed','true');}
  if(input){
    try{input.focus({preventScroll:false});}catch(e){try{input.focus();}catch(e2){}}
  }
}
function lcCloseKeyboard(){
  LC.keyOn=false;
  var bar=document.getElementById('liveTypeBar');
  var btn=document.getElementById('liveKeyBtn');
  var input=document.getElementById('liveTypeInput');
  if(bar)bar.classList.remove('on');
  if(btn){btn.classList.remove('on');btn.setAttribute('aria-pressed','false');}
  if(input)try{input.blur();}catch(e){}
}
function lcToggleKeyboard(){
  if(LC.keyOn)lcCloseKeyboard();
  else lcOpenKeyboard();
}
function lcToggleDrag(){lcSetDrag(!LC.dragOn);}
function lcOnTypeInput(){
  var el=document.getElementById('liveTypeInput');
  if(!el)return;
  var v=el.value||'';
  var last=LC.typeLast||'';
  if(v===last)return;
  if(v.length>last.length && v.indexOf(last)===0){
    lcSend({action:'type',text:v.slice(last.length)}).catch(function(){});
  }else if(v.length<last.length && last.indexOf(v)===0){
    var n=last.length-v.length;
    var chain=Promise.resolve();
    for(var i=0;i<n;i++){
      chain=chain.then(function(){return lcSend({action:'key',key:'Backspace'});});
    }
    chain.catch(function(){});
  }else{
    var wipe=Promise.resolve();
    for(var j=0;j<last.length;j++){
      wipe=wipe.then(function(){return lcSend({action:'key',key:'Backspace'});});
    }
    wipe.then(function(){if(v)return lcSend({action:'type',text:v});}).catch(function(){});
  }
  LC.typeLast=v;
}
function lcOnTypeKey(event){
  if(event.key==='Enter'){
    event.preventDefault();
    lcSend({action:'key',key:'Enter'}).catch(function(){});
  }else if(event.key==='Tab'){
    event.preventDefault();
    lcSend({action:'key',key:'Tab'}).catch(function(){});
  }
}
function lcOnPointerDown(event){
  if(!LC.interactive)return;
  if(event.button!==undefined && event.button!==0)return;
  var p=lcPageXY(event);
  if(!p)return;
  event.preventDefault();
  LC.drag={x1:p.x,y1:p.y,x2:p.x,y2:p.y,moved:false};
  try{event.currentTarget.setPointerCapture(event.pointerId);}catch(e){}
  lcSetHud(lcFmt(p.x)+', '+lcFmt(p.y));
  lcMarkAt(p.x,p.y);
}
function lcOnPointerMove(event){
  var p=lcPageXY(event);
  if(p)lcSetHud(lcFmt(p.x)+', '+lcFmt(p.y));
  if(!LC.drag||!p)return;
  event.preventDefault();
  LC.drag.x2=p.x;LC.drag.y2=p.y;
  var dx=p.x-LC.drag.x1,dy=p.y-LC.drag.y1;
  if((dx*dx+dy*dy)>36)LC.drag.moved=true;
  if(LC.drag.moved && LC.dragOn){
    lcDrawTrail(LC.drag.x1,LC.drag.y1,p.x,p.y);
    lcSetLast('dragging '+lcFmt(LC.drag.x1)+','+lcFmt(LC.drag.y1)+' → '+lcFmt(p.x)+','+lcFmt(p.y));
  }
}
function lcOnPointerUp(event){
  if(!LC.drag)return;
  event.preventDefault();
  var p=lcPageXY(event)||{x:LC.drag.x2,y:LC.drag.y2};
  var start=LC.drag;
  LC.drag=null;
  lcClearTrail();
  var send;
  if(start.moved && LC.dragOn){
    lcSetLast('drag '+lcFmt(start.x1)+','+lcFmt(start.y1)+' → '+lcFmt(p.x)+','+lcFmt(p.y));
    lcMarkAt(p.x,p.y);
    send=lcSend({action:'drag',x1:start.x1,y1:start.y1,x2:p.x,y2:p.y});
  }else{
    lcSetLast('clicked '+lcFmt(start.x1)+', '+lcFmt(start.y1));
    lcMarkAt(start.x1,start.y1);
    send=lcSend({action:'click',x:start.x1,y:start.y1});
  }
  send.catch(function(){lcSetStatus('Manual pointer action failed');});
}
function lcRefresh(){
  var overlay=document.getElementById('liveOverlay');
  if(!overlay||!overlay.classList.contains('on'))return;
  fetch('/status?t='+Date.now())
    .then(function(r){if(!r.ok)throw new Error('status unavailable');return r.json();})
    .then(function(data){
      var workers=data.workers||[];
      var worker=workers.find(function(w){return w.id===LC.worker;})||{};
      var state=worker.status||'idle';
      LC.interactive=(state==='demo'||state==='running'||state==='done');
      var liveImg=document.getElementById('liveImage');
      if(liveImg)liveImg.style.cursor=LC.interactive?'crosshair':'default';
      lcSetStatus(state==='running'||state==='starting'?'Live · '+state:
        (state==='demo'?'Demo · click the frame':'Camera · '+state));
      if(state==='idle'||state==='stopped'){
        lcShowPlaceholder('Start the browser runner to view its camera.');
      }else{
        lcLoadFrame();
      }
    })
    .catch(function(){lcSetStatus('Camera status unavailable');});
  LC.timer=setTimeout(lcRefresh,3000);
}
function lcGoRegister(){
  lcSetStatus('Opening Discord register…');
  fetch('/browser/start?worker='+encodeURIComponent(LC.worker),{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({url:'https://discord.com/register',force:true})
  }).then(function(r){return r.json();}).then(function(st){
    if(st&&st.connected){LC.connected=true;LC.interactive=true;}
    lcApplyState(st||{});
    if(st&&st.error)lcSetStatus(st.error);
    else lcSetStatus('Camera · register');
  }).catch(function(){lcSetStatus('Could not open Discord register');});
}
function openLive(){
  var overlay=document.getElementById('liveOverlay');
  if(!overlay)return;
  overlay.classList.add('on');
  if(LC.timer)clearTimeout(LC.timer);
  fetch('/browser/start?worker='+encodeURIComponent(LC.worker),{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({})
  }).then(function(r){return r.json();}).then(function(st){
    if(st&&st.connected){LC.connected=true;LC.interactive=true;}
    lcApplyState(st||{});
  }).catch(function(){}).then(function(){lcRefresh();});
}
function closeLive(){
  var overlay=document.getElementById('liveOverlay');
  if(overlay)overlay.classList.remove('on');
  if(LC.timer){clearTimeout(LC.timer);LC.timer=null;}
  LC.drag=null;
  lcClearTrail();
  lcCloseKeyboard();
}
window.openLive=openLive;
window.closeLive=closeLive;
(function(){
  var overlay=document.getElementById('liveOverlay');
  if(overlay)overlay.addEventListener('click',function(e){if(e.target===overlay)closeLive();});
  var img=document.getElementById('liveImage');
  if(img){
    img.addEventListener('pointerdown',lcOnPointerDown);
    img.addEventListener('pointermove',lcOnPointerMove);
    img.addEventListener('pointerup',lcOnPointerUp);
    img.addEventListener('pointercancel',lcOnPointerUp);
    img.addEventListener('dragstart',function(e){e.preventDefault();});
  }
  var keyBtn=document.getElementById('liveKeyBtn');
  if(keyBtn)keyBtn.addEventListener('click',function(e){e.preventDefault();lcToggleKeyboard();});
  var dragBtn=document.getElementById('liveDragBtn');
  if(dragBtn)dragBtn.addEventListener('click',function(e){e.preventDefault();lcToggleDrag();});
  var regBtn=document.getElementById('liveRegBtn');
  if(regBtn)regBtn.addEventListener('click',function(e){e.preventDefault();lcGoRegister();});
  var copyBtn=document.getElementById('liveCopyAll');
  if(copyBtn)copyBtn.addEventListener('click',function(e){e.preventDefault();lcCopyAll();});
  var typeInput=document.getElementById('liveTypeInput');
  if(typeInput){
    typeInput.addEventListener('input',lcOnTypeInput);
    typeInput.addEventListener('keydown',lcOnTypeKey);
  }
  document.addEventListener('keydown',function(e){
    if(e.key==='Escape'){
      if(LC.keyOn){lcCloseKeyboard();return;}
      closeLive();
    }
  });
})();
</script>
'''
