const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map();
const context=vm.createContext({console,Date,Intl,Number,JSON,document:{getElementById(id){if(!elements.has(id))elements.set(id,{classList:{toggle(){}},innerHTML:'',textContent:''});return elements.get(id);}}});
const source=fs.readFileSync('project/serving_app/static/dashboard.js','utf8');
vm.runInContext(source.slice(0,source.lastIndexOf('init().catch')),context);
for (const [status,promoted,expected] of [['promoted',true,'운영 v2'],['gate_rejected',false,'검증 게이트 미통과'],['activation_failed',false,'모델 활성화 실패'],['failed',false,'학습 실패']]) {
  context.result={scenario:'drift',base_model_version:'1',served_model_version:promoted?'2':'1',model_name:'JejuSolarPredictor',completed_at:Date.now()/1000,drift_check:{status:'performance_degraded',ready:true,rmse:30,threshold:20},retraining:{status,promoted,rmse:15,reason:'test'}};
  vm.runInContext('renderSimulation(result)',context);
  assert.ok(elements.get('simulation-status').innerHTML.includes(expected),elements.get('simulation-status').innerHTML);
  if(!promoted)assert.ok(elements.get('pipeline-track').innerHTML.includes('기존 버전 유지'));
}
console.log('dashboard UI status checks passed');
