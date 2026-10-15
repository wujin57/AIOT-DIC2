'use strict';
const $ = id => document.getElementById(id);
const money = n => data?.problem === 'boston' ? `${number(n)} kUSD` : new Intl.NumberFormat('en-US', {style:'currency', currency:'USD', maximumFractionDigits:2}).format(n);
const number = n => n.toLocaleString('en-US', {maximumFractionDigits:4});
const meta = {
  pearson: {name:'Pearson Correlation', type:'Filter method', icon:'↗', description:'Ranks absolute Pearson correlations with the target on training data. Captures linear association; constant columns receive a score of zero.'},
  mi: {name:'Mutual Information', type:'Filter method', icon:'◈', description:'Ranks training-data mutual information. Categorical indicators are treated as discrete. Randomness is fixed at 42.'},
  rfe: {name:'RFE', type:'Wrapper method', icon:'✂', description:'Recursively eliminates predictors using standardized OLS coefficient magnitudes. Scaling is refitted inside each validation fold.'},
  rfecv: {name:'RFECV', type:'Wrapper · automatic k', icon:'◎', description:'Automatically selects k with 5-fold training-only cross-validation. The internal selection curve is shown below; outer CV independently evaluates the full selection procedure.'},
  lasso: {name:'LASSO', type:'Embedded · automatic k', icon:'λ', description:'LassoCV tunes alpha on standardized predictors and selects nonzero coefficients. Native LASSO chooses its own count. The selected columns are then refit with OLS.'}
};
let data, method='pearson', k=2, comparing=false, active, charts={};
const fixed = name => ['pearson','mi','rfe'].includes(name);
const modelFor = name => data.models.find(m=>m.method===name && (fixed(name)?m.k===k:true));
const bestFor = name => data.models.filter(m=>m.method===name).sort((a,b)=>a.cv.rmse-b.cv.rmse || a.k-b.k)[0];
const chartColors=['#7660d9','#25a285','#e8a044','#5494ce','#c47ca5'];
function draw(id, type, datasets, xTitle, yTitle, extra={}) {
  if(charts[id]) charts[id].destroy();
  charts[id]=new Chart($(id),{type,data:{datasets}, options:{responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'nearest',intersect:false},plugins:{legend:{display:datasets.length>1,position:'bottom',labels:{usePointStyle:true,boxWidth:6,font:{size:10},padding:18}},tooltip:{callbacks:{label:ctx=>`${ctx.dataset.label}: ${number(ctx.parsed.y)}`}}},scales:{x:{type:'linear',title:{display:true,text:xTitle,color:'#9299a9',font:{size:10}},grid:{display:false},ticks:{color:'#9299a9',font:{size:10}},...extra.x},y:{title:{display:true,text:yTitle,color:'#9299a9',font:{size:10}},grid:{color:'#f0f1f6'},border:{display:false},ticks:{color:'#9299a9',font:{size:10}},...extra.y}}}});
}
function series(label, points, color, more={}){return {label,data:points,borderColor:color,backgroundColor:color,pointRadius:4,pointHoverRadius:6,borderWidth:2,tension:.2,...more};}
function renderCharts(){
  const rows=data.models.filter(m=>m.method===method);
  let sets;
  if(comparing){
    sets=Object.keys(meta).map((name,i)=>series(meta[name].name,data.models.filter(m=>m.method===name).map(m=>({x:m.k,y:m.cv.rmse})),chartColors[i],{showLine:fixed(name)}));
    $('curve-caption').textContent='Outer CV RMSE · automatic methods appear at their final selected k';
  }else if(method==='rfecv'){
    sets=[series('RFECV internal tuning',active.selection.internal_curve.map((v,i)=>({x:i+1,y:v})),chartColors[0])];
    $('curve-caption').textContent='Internal 5-fold tuning RMSE · distinct from outer CV metrics';
  }else{
    sets=[series('Outer CV RMSE',rows.map(m=>({x:m.k,y:m.cv.rmse})),chartColors[0],{showLine:fixed(method)})];
    $('curve-caption').textContent=method==='lasso'?'Native LASSO · one automatically selected subset':'5-fold outer CV RMSE · selection repeated inside each fold';
  }
  draw('rmse-chart','line',sets,'Encoded features (k)',`RMSE (${data.unit || 'USD'})`,{x:{min:0,max:data.encoded_features.length,ticks:{stepSize:1,color:'#9299a9'}}});
  const r2sets=comparing?Object.keys(meta).map((name,i)=>series(meta[name].name,data.models.filter(m=>m.method===name).map(m=>({x:m.k,y:m.test.r2})),chartColors[i],{showLine:fixed(name)})):[series('Training R²',rows.map(m=>({x:m.k,y:m.train.r2})),'#b2a2e8',{borderDash:[5,4],showLine:fixed(method)}),series('Test R²',rows.map(m=>({x:m.k,y:m.test.r2})),chartColors[1],{showLine:fixed(method)})];
  $('r2-caption').textContent=comparing?'Holdout R² across methods · not used to choose the model':'Training and holdout R² · higher is better';
  draw('r2-chart','line',r2sets,'Encoded features (k)','R²',{x:{min:0,max:data.encoded_features.length,ticks:{stepSize:1,color:'#9299a9'}}});
  const all=active.residuals.flatMap(r=>[r.actual,r.predicted]);const lo=Math.min(...all)*.9, hi=Math.max(...all)*1.05;
  draw('prediction-chart','scatter',[series('Held-out records',active.residuals.map(r=>({x:r.actual,y:r.predicted})),chartColors[0]),series('Perfect prediction',[{x:lo,y:lo},{x:hi,y:hi}],'#bcc2d1',{showLine:true,pointRadius:0,borderDash:[5,5]})],`Actual ${data.target} (${data.unit})`,`Predicted ${data.target} (${data.unit})`);
  draw('residual-chart','scatter',[series('Residual',active.residuals.map(r=>({x:r.predicted,y:r.residual})),chartColors[1]),series('Zero error',[{x:lo,y:0},{x:hi,y:0}],'#bcc2d1',{showLine:true,pointRadius:0,borderDash:[5,5]})],`Predicted ${data.target} (${data.unit})`,`Residual (${data.unit})`);
}
function predict(){
  if(!active)return;
  const form=$('prediction-form');
  const numeric = Object.keys(data.ranges);
  const vals=numeric.map((name,i)=>Number(form.elements[`field-${i}`].value));
  if(!form.checkValidity()||!vals.every(Number.isFinite)){$('prediction-value').textContent='Enter valid values';return;}
  const state=form.elements.state?.value;
  const x=data.problem === 'boston' ? vals : [...vals,Number(state==='Florida'),Number(state==='New York')];
  const result=active.intercept+active.indices.reduce((sum,j,i)=>sum+x[j]*active.coefficients[i],0);
  $('prediction-value').textContent=Number.isFinite(result)?money(result):'Input is too large';
  $('prediction-caption').textContent=`${meta[method].name} · ${active.k} selected encoded features · OLS`;
  const outside=vals.some((v,i)=>v<data.ranges[data.original_features[i]][0]||v>data.ranges[data.original_features[i]][1]);
  $('prediction-warning').textContent=outside?'⚠ Outside observed training ranges. This scenario involves extrapolation.':'';
}
function render(){
  active=modelFor(method);
  document.querySelectorAll('[data-method]').forEach(b=>{b.classList.toggle('active',b.dataset.method===method);b.setAttribute('aria-pressed',b.dataset.method===method);});
  $('method-description').textContent=meta[method].description;
  const best=bestFor(method);
  $('feature-count').innerHTML=fixed(method)?Array.from({length:data.encoded_features.length},(_,i)=>i+1).map(n=>`<button data-k="${n}" class="${n===k?'active ':''}${n===best.k?'best':''}" aria-pressed="${n===k}">k = ${n}<small>${n===best.k?'★ Best CV':n===data.encoded_features.length?'All columns':`${n} encoded ${n===1?'column':'columns'}`}</small></button>`).join(''):`<button class="active" aria-pressed="true" disabled>Auto k = ${active.k}<small>${method==='rfecv'?'Selected by 5-fold RFECV':'Nonzero LASSO coefficients'}</small></button>`;
  $('model-label').textContent=`${meta[method].name} · k = ${active.k}`;
  $('metrics').innerHTML=[['Test RMSE',money(active.test.rmse),'Lower prediction error is better'],['Test R²',active.test.r2.toFixed(4),`Explained variance on ${data.test_rows.length} test rows`],['Training adjusted R²',active.train.adjusted_r2.toFixed(4),'Penalizes training model complexity'],['Test MAE',money(active.test.mae),'Mean absolute prediction error'],['Outer CV RMSE',money(active.cv.rmse),`Fold SD ${money(active.cv_rmse_sd)}`]].map(([label,value,note])=>`<div class="metric"><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`).join('');
  $('features').innerHTML=active.features.length?active.features.map(f=>`<span class="chip">${f}</span>`).join(''):'<span class="chip">Intercept only</span>';
  $('model-status').textContent=active.id===data.best_id?'★ Recommended by CV':`${active.k} / ${data.encoded_features.length} encoded predictors`;
  $('equation').textContent=`${data.target} = ${money(active.intercept)} `+active.features.map((f,i)=>`${active.coefficients[i]<0?'−':'+'} ${number(Math.abs(active.coefficients[i]))} × [${f}]`).join(' ');
  $('coefficients').innerHTML=[['Intercept',active.intercept],...active.features.map((f,i)=>[f,active.coefficients[i]])].map(([f,c])=>`<tr><td>${f}</td><td>${number(c)}</td><td class="${c>=0?'positive':'negative'}">${f==='Intercept'?'Baseline':c>=0?'↗ Positive':'↘ Negative'}</td></tr>`).join('');
  const s=active.selection;
  $('selection-detail').innerHTML=(s.alpha?`<p>LassoCV alpha: <b>${number(s.alpha)}</b>. Scores below are standardized LASSO coefficients, not final OLS coefficients.</p>`:'')+(s.scores?data.encoded_features.map((f,i)=>`<div>${f}: <b>${number(s.scores[i])}</b></div>`).join(''):'<p>RFE rank 1 indicates a selected feature. Elimination compares scaled coefficients.</p>'+data.encoded_features.map((f,i)=>`<div>${f}: rank ${s.ranking[i]}</div>`).join(''))+`<p>Feature counts across outer folds: ${active.fold_features.map(f=>f.length).join(', ')}.</p>`;
  $('residuals').innerHTML=[...active.residuals].sort((a,b)=>Math.abs(b.residual)-Math.abs(a.residual)).map(r=>`<tr><td>#${String(r.row).padStart(2,'0')}</td><td>${money(r.actual)}</td><td>${money(r.predicted)}</td><td class="${r.residual>=0?'positive':'negative'}">${money(r.residual)}</td><td>${money(Math.abs(r.residual))}</td></tr>`).join('');
  $('comparison').hidden=!comparing;$('compare').setAttribute('aria-pressed',comparing);
  if(comparing)$('comparison-body').innerHTML=Object.keys(meta).map(modelFor).sort((a,b)=>a.cv.rmse-b.cv.rmse).map(m=>`<tr><td>${meta[m.method].name}</td><td>${m.k}${fixed(m.method)?'':' (auto)'}</td><td>${money(m.cv.rmse)}</td><td>${money(m.test.rmse)}</td><td>${m.test.r2.toFixed(4)}</td><td>${m.features.join(', ')||'Intercept only'}</td></tr>`).join('');
  predict();renderCharts();
}
let modelRequest = 0;
async function loadProblem(problem) {
  const current = ++modelRequest;
  const panel = $('model-results');
  panel.hidden = true;
  active = null;
  $('error').hidden = true;
  if (problem === 'custom') {
    $('recommendation').textContent = '自訂 Dataset 目前僅提供預覽，請選擇預設問題以查看模型。';
    return;
  }
  $('recommendation').textContent = 'Loading trained models…';
  try {
    const filename = problem === 'boston' ? 'boston-models.json' : 'models.json';
    const response = await fetch(`data/${filename}`);
    if(!response.ok) throw new Error(`Model data returned HTTP ${response.status}`);
    const next = await response.json();
    if(current !== modelRequest) return;
    data=next;
    data.target ||= 'Profit'; data.unit ||= 'USD'; data.title ||= '50 Startups'; data.csv ||= '50_Startups.csv';
    const best=data.models.find(m=>m.id===data.best_id);
    method=best.method; k=fixed(method)?best.k:Math.min(2,data.encoded_features.length); comparing=false;
    configureDataset(filename);
    $('recommendation').textContent=`${data.title} · CV recommendation: ${meta[best.method].name} · ${best.k} features · ${money(best.cv.rmse)} validation RMSE`;
    panel.hidden=false;
    render();
  } catch(error) {
    if(current !== modelRequest) return;
    $('error').hidden=false; $('error').textContent=`模型載入失敗：${error.message}`;
    $('recommendation').textContent='Model data unavailable';
  }
}
window.loadProblem = loadProblem;
function configureDataset(filename) {
  document.querySelector('h1').textContent=`${data.title} · Regression Explorer`;
  document.title=`${data.title} · Regression Explorer`;
  document.querySelector('.hero > p').textContent=`Compare five feature selection methods for predicting ${data.target} with ${data.title}.`;
  document.querySelector('.hero-tags span').textContent=data.title;
  document.querySelector('#controls .badge').textContent=`${data.train_rows.length} training / ${data.test_rows.length} test records`;
  document.querySelector('#controls > .hint').textContent=`${data.original_features.length} original → ${data.encoded_features.length} encoded predictors`;
  document.querySelector('.equation-card p').textContent=`Target unit: ${data.unit} · OLS refit after feature selection`;
  document.querySelector('#predictor h3').textContent=`Live ${data.target} predictor`;
  document.querySelector('#predictor .section-top p').textContent=`Explore a ${data.title} scenario`;
  document.querySelector('.prediction > span').textContent=`ESTIMATED ${data.target}`;
  document.querySelector('#prediction-form > .hint:last-child').textContent=`Model fitted to ${data.train_rows.length} training records. A point estimate, not a guarantee.`;
  const inputs=document.querySelector('#prediction-form .input-grid'); inputs.replaceChildren();
  Object.entries(data.ranges).forEach(([name,range],i)=>{
    const label=document.createElement('label'); label.textContent=name;
    const input=document.createElement('input'); input.name=`field-${i}`; input.type='number'; input.step='any'; input.required=true; input.min='0';
    input.value=number((range[0]+range[1])/2).replaceAll(',','');
    label.append(input); inputs.append(label);
  });
  if(data.categories.length) {
    const label=document.createElement('label'); label.textContent='State';
    const select=document.createElement('select'); select.name='state';
    data.categories.forEach(value=>select.add(new Option(value,value))); label.append(select); inputs.append(label);
  }
  const predictionCard=$('prediction-chart').closest('.card');
  predictionCard.querySelector('h3').textContent=`Actual vs. predicted ${data.target}`;
  predictionCard.querySelector('p').textContent=`${data.test_rows.length} held-out records · dashed line is a perfect prediction`;
  $('prediction-chart').setAttribute('aria-label',`Actual versus predicted ${data.target}`);
  $('residual-chart').setAttribute('aria-label',`Residuals against predicted ${data.target}`);
  const residualCard=$('residuals').closest('.card');
  residualCard.querySelector('p').textContent=`All ${data.test_rows.length} unseen records, sorted by absolute error. Row IDs refer to the source CSV.`;
  residualCard.querySelectorAll('th')[1].textContent=`Actual ${data.target}`;
  residualCard.querySelectorAll('th')[2].textContent=`Predicted ${data.target}`;
  document.querySelectorAll('a[download]').forEach(link=>{link.href=link.closest('#methodology')?`data/${filename}`:`data/${data.csv}`;});
  document.querySelector('#dataset h3').textContent=data.title;
  document.querySelector('#dataset summary').textContent=`Explore all ${data.records.length} records`;
  document.querySelector('.dataset-stats').innerHTML=`<div><strong>${data.records.length}</strong><span>Records</span></div><div><strong>${data.original_features.length} → ${data.encoded_features.length}</strong><span>Original → encoded predictors</span></div><div><strong>${data.target}</strong><span>Target (${data.unit})</span></div><div><strong>0</strong><span>Missing values</span></div>`;
  document.querySelector('#dataset thead tr').innerHTML=['Row',...data.original_features,data.target,'Split'].map(name=>`<th>${name}</th>`).join('');
  $('data-body').innerHTML=data.records.map((r,i)=>`<tr><td>${i+1}</td>${data.original_features.map(f=>`<td>${typeof r[f]==='number'?number(r[f]):r[f]}</td>`).join('')}<td>${money(r[data.target])}</td><td>${data.test_rows.includes(i+1)?'Test':'Train'}</td></tr>`).join('');
  const explanations=document.querySelectorAll('.methodology-grid article p');
  explanations[0].textContent=`Predict ${data.target} using ${data.title}. Compare predictive accuracy and model simplicity.`;
  explanations[1].textContent=`${data.records.length} records; ${data.original_features.length} original predictors; target unit: ${data.unit}.`;
  explanations[2].textContent='80/20 split, random_state=42. Feature selection is repeated inside training-only validation folds. State is one-hot encoded for 50 Startups; Boston uses numeric columns.';
  document.querySelector('footer > span').textContent=`${data.title} / REGRESSION EXPLORER`;
}
function init(){
    $('methods').innerHTML=Object.entries(meta).map(([key,m])=>`<button class="method-button" data-method="${key}" aria-pressed="false"><span aria-hidden="true">${m.icon}</span> &nbsp;${m.name}<small>${m.type}</small></button>`).join('');
    $('methods').addEventListener('click',e=>{const b=e.target.closest('[data-method]');if(b){method=b.dataset.method;render();}});
    $('feature-count').addEventListener('click',e=>{const b=e.target.closest('[data-k]');if(b){k=Number(b.dataset.k);render();}});
    $('compare').addEventListener('click',()=>{comparing=!comparing;render();});
    $('prediction-form').addEventListener('input',predict);$('prediction-form').addEventListener('submit',e=>e.preventDefault());
    $('export').addEventListener('click',()=>{const rows=['source_row,actual,predicted,residual',...active.residuals.map(r=>[r.row,r.actual,r.predicted,r.residual].join(','))];const url=URL.createObjectURL(new Blob([rows.join('\n')],{type:'text/csv'}));const link=document.createElement('a');link.href=url;link.download=`${data.problem || 'startups'}-${active.id}-residuals.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
    loadProblem('startups');
}
init();
