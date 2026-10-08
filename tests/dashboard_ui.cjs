const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map();
const context=vm.createContext({console,Date,Intl,Number,JSON,document:{getElementById(id){if(!elements.has(id))elements.set(id,{classList:{toggle(){}},replaceChildren(){this.innerHTML='';},innerHTML:'',textContent:''});return elements.get(id);}}});
const source=fs.readFileSync('project/serving_app/static/dashboard.js','utf8');
vm.runInContext(source.slice(0,source.lastIndexOf('init().catch')),context);
for (const [status,promoted,expected] of [['promoted',true,'운영 v2'],['gate_rejected',false,'검증 게이트 미통과'],['activation_failed',false,'모델 활성화 실패'],['failed',false,'학습 실패']]) {
  context.result={scenario:'drift',base_model_version:'1',served_model_version:promoted?'2':'1',model_name:'JejuSolarPredictor',completed_at:Date.now()/1000,drift_check:{status:'performance_degraded',ready:true,rmse:30,threshold:20},retraining:{status,promoted,rmse:15,reason:'test'}};
  vm.runInContext('renderSimulation(result)',context);
  assert.ok(elements.get('simulation-status').innerHTML.includes(expected),elements.get('simulation-status').innerHTML);
  if(!promoted)assert.ok(elements.get('pipeline-track').innerHTML.includes('기존 버전 유지'));
}
console.log('dashboard UI status checks passed');
(async()=>{
  context.document.querySelectorAll=()=>[];
  context.emptyMetrics={request_count:0,avg_latency_ms:0,success_rate:0,error_rate:0,series:[],drift:{count:0,rmse:null}};
  vm.runInContext('api=async()=>emptyMetrics',context);
  await vm.runInContext('loadMetrics()',context);
  assert.equal(elements.get('kpi-success').textContent,'—','요청이 없을 때 성공률 0%로 오해시키면 안 됩니다');
  assert.ok(elements.get('metrics-empty').textContent.includes('요청이 없습니다'));
  context.comparison={rmse:16.31,incumbent_rmse:441.65,parent_version:'3',validation_start:'2024-05-31T18:00:00',validation_end:'2024-06-07T17:00:00'};
  const comparison=vm.runInContext('comparisonText(comparison)',context);
  assert.ok(comparison.includes('441.65'));
  assert.ok(comparison.includes('16.31'));
  assert.ok(comparison.includes('개선'));
  console.log('dashboard operations UI checks passed');
})().catch(e=>{console.error(e);process.exitCode=1;});

for (const [value, recorded, expected] of [[0,true,'0%'],[12.345,true,'12.35%'],[null,true,'계산 불가'],[null,false,'미기록'],[150,true,'150%']]) {
  context.wapeValue=value;context.wapeRecorded=recorded;
  assert.equal(vm.runInContext('wapeText(wapeValue,wapeRecorded)',context),expected);
}
context.wapeRun={drift_check:{status:'ok',ready:true,rmse:20,wape_pct:12,threshold:30},completed_at:Date.now()/1000,scenario:'normal',base_model_version:'1',served_model_version:'1',model_name:'test'};
vm.runInContext('renderSimulation(wapeRun)',context);
assert.ok(elements.get('simulation-summary').innerHTML.includes('배치 WAPE'));
assert.ok(elements.get('simulation-summary').innerHTML.includes('12%'));

// Missing thresholds and incomplete observations must never appear healthy.
for(const [check,label] of [
  [{count:0,ready:false,status:'insufficient_data',rmse:null,wape_pct:null},'관측 대기'],
  [{count:4,ready:false,status:'insufficient_data',rmse:20,wape_pct:5},'관측 누적 중'],
  [{count:14,ready:false,status:'threshold_unavailable',rmse:20,wape_pct:5},'기준 확인 필요'],
  [{count:14,ready:true,status:'performance_degraded',rmse:20,wape_pct:5},'성능 저하'],
  [{count:14,ready:true,status:'ok',rmse:20,wape_pct:5},'정상 범위']
]) {
  context.monitorCheck=check;
  vm.runInContext('renderMonitoringSummary(monitorCheck)',context);
  assert.equal(elements.get('monitor-state').textContent,label);
}
context.boundaryModel={validation_end:'2024-08-03T00:00:00',simulation:true,version:'2',stage:'Production'};
elements.set('simulation-start',{value:'2024-07-21'});
vm.runInContext('renderModelSummary(boundaryModel)',context);
assert.equal(elements.get('simulation-start').min,'2024-08-04');
assert.equal(elements.get('simulation-start').value,'','An overlapping prefilled date must be cleared');
assert.ok(elements.get('summary-origin').textContent.includes('시연'));
elements.get('simulation-start').value='2024-08-30';
vm.runInContext('renderModelSummary(boundaryModel)',context);
assert.equal(elements.get('simulation-start').value,'2024-08-30','Refreshing must preserve user input after boundary');
context.gateResult={status:'gate_rejected',promoted:false,validation_start:'2024-08-30',validation_end:'2024-09-12',metrics:{validation:{model:{rmse:20,wape_pct:10},incumbent:{rmse:19,wape_pct:12},persistence:{rmse:30,wape_pct:15},weekly_mean:{rmse:25,wape_pct:13}}}};
vm.runInContext('renderGate(gateResult)',context);
assert.ok(elements.get('gate-comparison').innerHTML.includes('검증 게이트 미통과'));
assert.ok(elements.get('gate-comparison').innerHTML.includes('20 MWh'));
assert.ok(elements.get('gate-comparison').innerHTML.includes('10%'));
console.log('daily operations presentation checks passed');

// Batch errors use the evaluated model, not the newly promoted serving model.
context.errorBatch={scenario:'drift',base_model_version:'2',served_model_version:'3',records:[
  {timestamp:'2024-09-01T00:00:00',predicted:80,actual:100},
  {timestamp:'2024-08-30T00:00:00',predicted:130,actual:100},
  {timestamp:'2024-09-02T00:00:00',predicted:0,actual:0},
  {timestamp:'2024-09-03T00:00:00',predicted:null,actual:100}
]};
vm.runInContext('renderBatchErrors(errorBatch)',context);
let chart=elements.get('batch-error-chart').innerHTML;
assert.ok(chart.includes('과대 예측 +30 MWh'));
assert.ok(chart.includes('과소 예측 -20 MWh'));
assert.ok(chart.includes('오차 없음 0 MWh'));
assert.ok(!chart.includes('2024-09-03'));
assert.ok(chart.indexOf('2024-08-30')<chart.indexOf('2024-09-01'));
assert.ok(elements.get('batch-error-context').textContent.includes('평가 모델 v2'));
assert.ok(!elements.get('batch-error-context').textContent.includes('v3'));
context.errorBatch.records=[{timestamp:'2024-09-02',predicted:0,actual:0}];
vm.runInContext('renderBatchErrors(errorBatch)',context);
chart=elements.get('batch-error-chart').innerHTML;
assert.ok(!/NaN|Infinity/.test(chart));
context.errorBatch.records=[];
vm.runInContext('renderBatchErrors(errorBatch)',context);
assert.ok(!elements.get('batch-error-chart').innerHTML.includes('<svg'));
assert.ok(elements.get('batch-error-chart').innerHTML.includes('배치를 실행'));
console.log('batch error chart checks passed');

// Reevaluation must show the current detection result without claiming training.
for (const [status,label] of [['performance_degraded','성능 저하 감지'],['ok','정상 범위']]) {
  context.recheck={scenario:'drift',base_model_version:'3',served_model_version:'3',model_name:'test',completed_at:Date.now()/1000+10,drift_check:{status,ready:true,rmse:30,threshold:20,evaluation_only:true,new_count:0,retraining:null},retraining:null};
  vm.runInContext('renderSimulation(recheck)',context);
  const message=elements.get('simulation-status').innerHTML;
  assert.ok(message.includes('재평가'),message);
  assert.ok(message.includes(label),message);
  assert.ok(message.includes('재학습·모델 교체 미실행'),message);
  assert.ok(elements.get('pipeline-run-badge').textContent.includes('재평가'));
  assert.ok(!elements.get('pipeline-track').innerHTML.includes('요청 완료'));
}
console.log('simulation reevaluation UI checks passed');

context.wapeHistory={batches:[
  {scenario:'drift',start_timestamp:'2024-09-20',cutoff_timestamp:'2024-10-03',base_model_version:'2',wape_pct:150,evaluation_only:true},
  {scenario:'normal',start_timestamp:'2024-09-01',cutoff_timestamp:'2024-09-14',base_model_version:'1',wape_pct:0,evaluation_only:false},
  {scenario:'normal',start_timestamp:'2024-09-02',cutoff_timestamp:'2024-09-15',base_model_version:'1',wape_pct:null,evaluation_only:null}
]};
vm.runInContext('renderWapeHistory(wapeHistory)',context);
let wapeChart=elements.get('wape-history-chart').innerHTML;
assert.ok(wapeChart.includes('150%'),'WAPE over 100% must remain visible');
assert.ok(wapeChart.includes('0%'),'Zero error is valid');
assert.ok(!/NaN|Infinity/.test(wapeChart));
assert.ok(wapeChart.includes('wape-reevaluation'));
assert.ok(wapeChart.indexOf('2024-09-01')<wapeChart.indexOf('2024-09-20'));
assert.ok(elements.get('wape-history-rows').innerHTML.includes('계산 불가'));
assert.ok(elements.get('wape-history-rows').innerHTML.includes('구분 미기록'));
assert.ok(wapeChart.includes('평가 v2'));
context.wapeHistory.batches=[];
vm.runInContext('renderWapeHistory(wapeHistory)',context);
assert.ok(!elements.get('wape-history-chart').innerHTML.includes('<svg'));
console.log('batch WAPE chart checks passed');

const batch=(day,scenario,version,value,time=1)=>({start_timestamp:`2024-09-${day}`,cutoff_timestamp:`2024-10-${day}`,scenario,base_model_version:version,wape_pct:value,completed_at:time,evaluation_only:false});
context.lineHistory={batches:[
  batch('01','normal','1',99,1),batch('01','normal','1',10,2),
  batch('02','normal','1',20),batch('03','normal','2',30),batch('04','normal','2',40),
  batch('01','drift','1',50),batch('02','drift','1',60),
  batch('03','drift','1',null),batch('04','drift','1',70)
]};
vm.runInContext('renderWapeHistory(lineHistory)',context);
wapeChart=elements.get('wape-history-chart').innerHTML;
assert.ok(!wapeChart.includes('WAPE 99%'),'Only newest duplicate is plotted');
const lines=[...wapeChart.matchAll(/<polyline class="wape-line"[^>]*points="([^"]+)"/g)];
assert.equal(lines.length,3,'Two model segments for normal, one drift segment before missing value');
assert.ok(lines.every(m=>m[1].trim().split(/\s+/).length===2),'Never connect across model changes or missing values');
const markers=[...wapeChart.matchAll(/<circle class="wape-mark" cx="([^"]+)"/g)];
assert.equal(markers.length,7);
assert.equal(markers[0][1],markers[1][1],'Normal and drift from the same period share the x position');
console.log('WAPE line grouping and deduplication checks passed');
