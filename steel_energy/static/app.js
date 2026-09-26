"use strict";
const $ = id => document.getElementById(id);
const number = (v, digits = 2) => Number(v).toFixed(digits);
async function json(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}
async function health() {
  try { const h = await json('/health'); $('health').textContent = `API ready / model v${h.model_version} / cached loads: ${h.load_count}`; }
  catch (_) { $('health').textContent = 'API model unavailable / complete the pipeline to enable replay'; }
}
function bars(id, rows, maximum) {
  const container = $(id); container.replaceChildren();
  rows.forEach(r => {
    const row = document.createElement('div'); row.className = 'bar-row' + (r.selected ? ' selected' : '');
    const label = document.createElement('span'); label.textContent = r.label;
    const track = document.createElement('div'); track.className = 'bar-track';
    const fill = document.createElement('div'); fill.className = 'bar-fill'; fill.style.width = `${Math.max(0, r.value) / maximum * 100}%`; track.append(fill);
    const value = document.createElement('span'); value.className = 'bar-value'; value.textContent = number(r.value);
    row.append(label, track, value); container.append(row);
  });
}
function chart(data, day) {
  const rows = data.series.filter(r => r.interval_start.slice(0,10) === day);
  const threshold = data.model.threshold_kwh;
  const max = Math.ceil(Math.max(threshold, ...rows.flatMap(r => [r.energy,r.prediction,r.baseline])) / 20) * 20;
  const x = i => 55 + i / Math.max(1, rows.length - 1) * 920, y = v => 285 - v / max * 255;
  const svg = $('chart'); svg.replaceChildren();
  const ns = 'http://www.w3.org/2000/svg';
  function node(tag, attributes, text) { const e=document.createElementNS(ns,tag); Object.entries(attributes).forEach(([k,v])=>e.setAttribute(k,v)); if(text!==undefined)e.textContent=text; svg.append(e); return e; }
  for(let v=0;v<=max;v+=20){node('line',{x1:55,x2:975,y1:y(v),y2:y(v),stroke:'#e3e6de'});node('text',{x:43,y:y(v)+4,'text-anchor':'end','font-size':12,fill:'#597078'},String(v));}
  node('text',{x:55,y:16,'font-size':12,fill:'#597078'},'kWh / 15 min');
  node('line',{x1:55,x2:975,y1:y(threshold),y2:y(threshold),stroke:'#a24431','stroke-dasharray':'4 5'});
  [['baseline','#89938d','5 4'],['energy','#183c49',''],['prediction','#a96720','']].forEach(([key,color,dash])=>node('path',{d:rows.map((r,i)=>`${i?'L':'M'}${x(i)},${y(r[key])}`).join(' '),fill:'none',stroke:color,'stroke-width':key==='baseline'?1.4:2.1,'stroke-dasharray':dash}));
  rows.forEach((r,i)=>{if(i%12===0)node('text',{x:x(i),y:312,'text-anchor':'middle','font-size':12,fill:'#597078'},r.interval_start.slice(11,16)); if(r.missed){const c=node('circle',{cx:x(i),cy:y(r.energy),r:3.5,fill:'#a24431'});const t=document.createElementNS(ns,'title');t.textContent=`Missed peak at ${r.interval_start}: ${number(r.energy)} kWh`;c.append(t);}});
  $('day-summary').textContent = `${rows.length} intervals · daily MAE ${number(rows.reduce((s,r)=>s+Math.abs(r.energy-r.prediction),0)/rows.length)} kWh · ${rows.filter(r=>r.missed).length} missed high-load intervals (red dots)`;
}
async function init() {
  health();
  try {
    const d = await json('/evidence'), f = d.final, a=d.diagnostics;
    $('mae').textContent=number(f.metrics.mae,3); $('baseline').textContent=`Baseline MAE: ${number(f.baseline_metrics.mae,3)} kWh / interval`;
    $('improvement').textContent=`${number(f.relative_mae_improvement_pct,1)}%`; $('ci').textContent=`MAE reduction 95% block-bootstrap CI: ${f.moving_block_95_ci_kwh.map(v=>number(v)).join('–')} kWh`;
    $('recall').textContent=`${number(f.advisory.recall*100,1)}%`; $('misses').textContent=`${a.missed_intervals} of ${f.advisory.high_load_count} high-load intervals missed`;
    const days=[...new Set(d.series.map(r=>r.interval_start.slice(0,10)))];
    days.forEach(day=>{const o=document.createElement('option');o.value=day;o.textContent=day;$('day').append(o);});
    $('day').value=a.largest_misses[0]?.interval_start.slice(0,10)||days[0];
    $('day').addEventListener('change',()=>chart(d,$('day').value));chart(d,$('day').value);
    const ex=d.experiments.slice().sort((a,b)=>a.mae-b.mae);bars('experiments',ex.map(r=>({label:`${r.id} / ${r.kind}`,value:r.mae,selected:r.id===f.selected_id})),Math.max(...ex.map(r=>r.mae)));
    $('peak-description').textContent=`${a.missed_intervals} missed intervals across ${a.missed_episodes} episodes. ${a.rising_misses} misses occurred above the last available energy reading. Median rise: ${number(a.median_ramp_from_last_available_kwh)} kWh.`;
    const max=Math.max(1,...a.hourly.map(r=>r.missed)); a.hourly.forEach(r=>{const b=document.createElement('div');b.className='hour';b.style.height=`${Math.max(1,r.missed/max*110)}px`;b.title=`${r.hour}:00 — ${r.missed} misses / ${r.high_load} high-load intervals`;b.setAttribute('aria-label',b.title);if(r.hour%3===0){const l=document.createElement('small');l.textContent=String(r.hour).padStart(2,'0');b.append(l);} $('hours').append(b);});
    bars('importance',d.importance.rows.map(r=>({label:r.group,value:r.mae_increase_kwh})),Math.max(.01,...d.importance.rows.map(r=>r.mae_increase_kwh)));
    $('candidate').textContent=`${d.model.spec.id} / Random Forest / registry v${d.model.registered_version}`;$('checksum').textContent=d.model.model_sha256;
    $('content').hidden=false;
  } catch(e) { $('error').textContent=`Evidence unavailable. ${e.message}. See the README for Docker startup instructions.`;$('error').hidden=false; }
  finally { $('loading').hidden=true; }
}
$('replay').addEventListener('click',async()=>{
  $('replay').disabled=true;$('replay-result').textContent='Scoring historical readings…';
  try {const payload=await json('/replay-request');const start=performance.now();const r=await json('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});$('replay-result').textContent=`Forecast: ${number(r.forecast_kwh,3)} kWh / average ${number(r.average_kw,2)} kW. Browser round trip: ${number(performance.now()-start,0)} ms. Model v${r.model_version}. Advisory disabled; production approval: ${r.production_approved}.`;}
  catch(e){$('replay-result').textContent=`Replay unavailable: ${e.message}`;}
  finally{$('replay').disabled=false;health();}
});
init();
