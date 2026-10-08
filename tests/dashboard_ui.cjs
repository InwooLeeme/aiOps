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
