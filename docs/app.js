// ===== Tone.js graph =====
const ctx = {
  master: new Tone.Gain(0.9).toDestination(),
  a: { synth: new Tone.Synth({oscillator:{type:"sawtooth"}, envelope:{attack:0.01, release:0.1}}),
       fx: new Tone.Filter(400, "lowpass") },
  b: { synth: new Tone.Synth({oscillator:{type:"square"}, envelope:{attack:0.02, release:0.2}}),
       fx: new Tone.Chorus(4, 2.5, 0.2).start() },
  drums: { kick: new Tone.MembraneSynth({octaves:10, pitchDecay:0.02}),
           hat: new Tone.NoiseSynth({volume:-20, envelope:{attack:0.005, decay:0.05, sustain:0}}) },
  mixAB: new Tone.CrossFade(0.5),
  meter: new Tone.Meter()
};
ctx.a.synth.chain(ctx.a.fx, ctx.mixAB.a);
ctx.b.synth.chain(ctx.b.fx, ctx.mixAB.b);
ctx.drums.kick.connect(ctx.master);
ctx.drums.hat.connect(ctx.master);
ctx.mixAB.connect(ctx.master);
ctx.master.connect(ctx.meter);

// ===== State =====
let state = { t:0, alpha:0.5, features:{d_head:0, d_brake:0, d_thr:0}, sim:{} };
let mapping = defaultMapping;  // 可以被编辑器覆盖
let lastKickAt = 0;

// ===== Utils =====
function clamp(x, a, b){ return Math.max(a, Math.min(b, x)); }
function scale(x, inLo, inHi, outLo, outHi){
  const r = (x - inLo) / (inHi - inLo);
  return outLo + Math.max(0, Math.min(1, r)) * (outHi - outLo);
}

// ===== 默认映射：把单帧特征打进声音（可被覆盖）=====
function defaultMapping(frame){
  const { alpha, features:{d_head, d_brake, d_thr} } = frame;

  // 混合推子：NOR/VER等类 vs Hybrid
  ctx.mixAB.fade.value = clamp(alpha, 0, 1);

  // 旋律随转向角速度（d_head）微动
  const midi = 60 + Math.round(scale(d_head, -1.5, 1.5, -5, 7));
  ctx.a.synth.triggerAttackRelease(Tone.Midi(midi).toFrequency(), "8n");

  // 油门 -> 滤波器开度 + 帽子触发概率
  ctx.a.fx.frequency.value = 200 + 3800 * Math.abs(d_thr);
  if (Math.random() < Math.abs(d_thr) * 0.7){
    ctx.drums.hat.triggerAttackRelease("16n");
  }

  // 刹车 -> 踢鼓（做个限频，最多每 2/16 触一次）
  const now = Tone.now();
  if (Math.abs(d_brake) > 0.6 && now - lastKickAt > 0.125){
    ctx.drums.kick.triggerAttackRelease("C2", "8n");
    lastKickAt = now;
  }
}

// ===== Live coding 编辑器 =====
const codeEl = document.getElementById('code');
if (codeEl) codeEl.value = `function mapping(frame){ defaultMapping(frame); }`;

const applyBtn = document.getElementById('apply');
if (applyBtn) {
  applyBtn.onclick = ()=>{
    try{
      const fn = new Function('ctx','state','defaultMapping', codeEl.value + '; return mapping;');
      mapping = fn(ctx, state, defaultMapping);
      alert('Applied!');
    }catch(e){ alert('Error:\n' + e.message); }
  };
  document.addEventListener('keydown', (e)=>{
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter'){ applyBtn.click(); }
  });
}

// ===== 开始音频（浏览器策略：必须用户交互触发）=====
const startBtn = document.getElementById('start');
if (startBtn){
  startBtn.onclick = async ()=>{
    await Tone.start();
    startBtn.innerText = 'Running';
  };
}

// ====== 合成参数（来自服务端“hybrid_params”消息）======
function applySynthParams(params={}, alpha=0.5){
  // 1) 全局速度
  if (typeof params.bpm === 'number'){
    Tone.Transport.bpm.value = params.bpm;
  }

  // 2) A/B 声部权重（同时受 alpha 控制）
  const mix = clamp(alpha, 0, 1);
  ctx.mixAB.fade.value = mix;

  // 3) 轨道音量/音色（0..1）
  const vol = (x)=> Tone.gainToDb(clamp(x,0,1)); // 映射到 dB
  if (typeof params.lead === 'number')  ctx.a.synth.volume.value = vol(params.lead);
  if (typeof params.bass === 'number')  ctx.b.synth.volume.value = vol(params.bass);

  // 打击乐“密度”用 internal state 保存
  state._kickDensity  = clamp(params.kick ?? 0.5, 0, 1);
  state._hatDensity   = clamp(params.hat  ?? 0.5, 0, 1);
  state._snareDensity = clamp(params.snare?? 0.5, 0, 1);
}

// ====== WebSocket 连接（支持 query 覆盖）======
const wsSpan  = document.getElementById('ws');
const framePre= document.getElementById('frame');

function pickWSUrl(){
  const url = new URL(window.location.href);
  // 允许 ?ws=wss://your-bridge/ws 覆盖
  const q = url.searchParams.get('ws');
  if (q) return q;
  // 如果页面里通过 <script> 预置 window.WS_URL，也可用
  if (window.WS_URL) return window.WS_URL;
  // 默认占位：提醒配置
  return 'wss://YOUR-BRIDGE.onrender.com/ws';
}

function connect(){
  const ws = new WebSocket(pickWSUrl());
  ws.onopen = ()=>{
    if (wsSpan){ wsSpan.textContent='connected'; wsSpan.style.color='#10b981'; }
  };
  ws.onclose = ()=>{
    if (wsSpan){ wsSpan.textContent='disconnected'; wsSpan.style.color='#ef4444'; }
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev)=>{
    let msg = {};
    try{ msg = JSON.parse(ev.data); }catch(e){ return; }

    // A) 参数消息：来自 Streamlit 每帧推送的 z→音频参数
    if (msg.type === 'hybrid_params' && msg.params){
      applySynthParams(msg.params, msg.alpha ?? state.alpha);
      if (framePre) framePre.textContent = JSON.stringify({hybrid_params: msg}, null, 2);
      return;
    }

    // B) 帧消息：老格式 {t, alpha, features, sim}
    if (typeof msg.t === 'number' && msg.features){
      state = msg;
      if (framePre) framePre.textContent = JSON.stringify(state, null, 2);
      if (Tone.getContext().state === 'running'){
        // 带上“密度”来影响触发概率
        const f = structuredClone(state);
        const d = (x)=> (typeof x === 'number'? clamp(x,0,1):0.5);
        f.features._kickDensity  = d(state._kickDensity);
        f.features._hatDensity   = d(state._hatDensity);
        f.features._snareDensity = d(state._snareDensity);
        mapping(f);
      }
    }
  };
}
connect();
