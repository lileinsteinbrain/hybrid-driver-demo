// ----- Audio graph (可被 live code 覆盖) -----
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

let state = { t:0, alpha:0.5, features:{d_head:0, d_brake:0, d_thr:0}, sim:{} };

function clamp(x, a, b){ return Math.max(a, Math.min(b, x)); }
function scale(x, inLo, inHi, outLo, outHi){
  const r = (x - inLo) / (inHi - inLo);
  return outLo + Math.max(0, Math.min(1, r)) * (outHi - outLo);
}

// 默认映射，可被编辑器覆盖
function defaultMapping(frame){
  const { alpha, features:{d_head, d_brake, d_thr} } = frame;
  ctx.mixAB.fade.value = clamp(alpha, 0, 1);
  const midi = 60 + Math.round(scale(d_head, -1.5, 1.5, -5, 7));
  ctx.a.synth.triggerAttackRelease(Tone.Midi(midi).toFrequency(), "8n");
  ctx.a.fx.frequency.value = 200 + 3800*Math.abs(d_thr);
  if (Math.random() < Math.abs(d_thr)*0.7){ ctx.drums.hat.triggerAttackRelease("16n"); }
  if (Math.abs(d_brake) > 0.6){ ctx.drums.kick.triggerAttackRelease("C2", "8n"); }
}

const codeEl = document.getElementById('code');
codeEl.value = `function mapping(frame){ defaultMapping(frame); }`;
let mapping = defaultMapping;
document.getElementById('apply').onclick = ()=>{
  try{
    const fn = new Function('ctx','state','defaultMapping', codeEl.value + '; return mapping;');
    mapping = fn(ctx, state, defaultMapping);
    alert('Applied!');
  }catch(e){ alert('Error:\\n' + e.message); }
};
document.addEventListener('keydown', (e)=>{
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter'){ document.getElementById('apply').click(); }
});

document.getElementById('start').onclick = async ()=>{
  await Tone.start();
  document.getElementById('start').innerText = 'Running';
};

// WebSocket
const wsSpan = document.getElementById('ws');
const framePre = document.getElementById('frame');
function connect(){
  const ws = new WebSocket(window.WS_URL);
  ws.onopen = ()=>{ wsSpan.textContent='connected'; wsSpan.style.color='#6ee7b7'; };
  ws.onclose = ()=>{ wsSpan.textContent='disconnected'; wsSpan.style.color='#fca5a5'; setTimeout(connect, 1000); };
  ws.onmessage = (ev)=>{
    state = JSON.parse(ev.data);
    framePre.textContent = JSON.stringify(state, null, 2);
    if (Tone.getContext().state === 'running'){ mapping(state); }
  }
}
connect();
