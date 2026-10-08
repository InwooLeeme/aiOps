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
