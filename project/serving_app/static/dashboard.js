"use strict";
const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "—").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const number = (value, digits = 0) => Number.isFinite(value) ? value.toLocaleString("ko-KR", {maximumFractionDigits: digits}) : "—";
const money = (value) => Number.isFinite(value) ? `$${value.toFixed(2)}` : "—";
const windowLabels = {"5m":"최근 5분","1h":"최근 1시간","6h":"최근 6시간","24h":"최근 24시간"};
const state = {tab:"dashboard", window:"5m", settings:null, dataset:null, models:null, lastRun:null, busy:false, refreshing:false};
const stages = [
  ["📥","데이터 수집","요청 수신"],["📊","데이터 모니터링","예측 기록 누적"],
  ["🔍","드리프트 감지","RMSE vs 임계치"],["⚙️","재학습 트리거","조건 충족 시 시작"],
  ["📦","모델 학습","fine-tuning"],["🗂️","모델 등록","MLflow Registry"],["☁️","배포","Production 승격"]
];
function pill(text, kind="neutral", dot=false) { return `<span class="pill ${kind}${dot ? " dot" : ""}">${escapeHtml(text)}</span>`; }
function relativeTime(ts) {
  const seconds = Math.max(0, Math.floor(Date.now()/1000-ts));
  return seconds < 60 ? `${seconds}초 전` : seconds < 3600 ? `${Math.floor(seconds/60)}분 전` : seconds < 86400 ? `${Math.floor(seconds/3600)}시간 전` : `${Math.floor(seconds/86400)}일 전`;
}
function registeredTime(ts) {
  if (!ts) return "—";
  return new Intl.DateTimeFormat("ko-KR", {month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"}).format(new Date(ts*1000));
}
async function api(path, options={}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.method === "POST" ? 180000 : 15000);
  try {
    const response = await fetch(path, {...options, signal:controller.signal});
    const raw = await response.text();
    let data;
    try { data = JSON.parse(raw); } catch { data = {detail:raw || "응답이 없습니다"}; }
    if (!response.ok) {
      const error = new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? data));
      error.status = response.status; error.data = data; throw error;
    }
    return data;
  } finally { clearTimeout(timeout); }
}
function displayError(error, target) {
  $(target).textContent = `${error.status ? `HTTP ${error.status}: ` : "연결 오류: "}${error.message}`;
  $(target).hidden = false;
}
function showResult(id, value, isError=false) {
  $(id).hidden = false; $(id).classList.toggle("error", isError);
  $(id).textContent = JSON.stringify(value, null, 2);
}
function sparkline(id, values) {
  const data = values.filter(Number.isFinite);
  if (!data.length) { $(id).replaceChildren(); return; }
  const low = Math.min(...data), high = Math.max(...data), span = high-low || 1;
  const points = data.map((v,i) => `${i*138/Math.max(data.length-1,1)+1},${28-(v-low)/span*26}`).join(" ");
  $(id).innerHTML = `<svg viewBox="0 0 140 32" role="img" aria-label="실제 데이터 추이"><polyline fill="none" stroke="#3475ff" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round" points="${points}" /></svg>`;
}
function stat(label, value) { return `<div class="stat"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`; }
function updateClock() {
  const now = new Date();
  $("clock").dateTime = now.toISOString();
  $("clock").textContent = new Intl.DateTimeFormat("ko-KR", {year:"numeric",month:"numeric",day:"numeric",hour:"numeric",minute:"numeric",second:"numeric",hour12:false}).format(now);
}
function switchTab(tab, focus=false) {
  if (!["dashboard","simulation","datasets","system"].includes(tab)) tab = "dashboard";
  state.tab = tab;
  document.querySelectorAll("[role=tabpanel]").forEach((panel) => {panel.hidden = panel.id !== tab;});
  document.querySelectorAll("[data-tab]").forEach((button) => {
    const active = button.dataset.tab === tab;
    button.setAttribute("aria-selected", active); button.tabIndex = active ? 0 : -1;
    if (active && focus) button.focus();
  });
  if (tab === "datasets") loadDataset().catch((e) => displayError(e,"connection-error"));
  if (tab === "system") loadSystem().catch((e) => displayError(e,"connection-error"));
  if (tab === "dashboard" || tab === "simulation") loadMetrics().catch((e) => displayError(e,"connection-error"));
}
async function loadSystem() {
  const data = await api("/system/info"); state.settings = data;
  $("model-source").textContent = `MODEL_SOURCE=${data.model_source}`;
  $("sequence-length").textContent = data.seq_len;
  $("minimum-rows").textContent = data.seq_len + data.window_size;
  $("system-stats").innerHTML = [
    ["SEQ_LEN",data.seq_len],["RMSE 게이트($)",data.rmse_gate],["드리프트 윈도우",data.window_size],
    ["드리프트 임계값($)",data.rmse_threshold],["BASE EPOCHS",data.base_epochs],
    ["FINE-TUNE EPOCHS",data.fine_tune_epochs],["FINE-TUNE LR",data.fine_tune_lr],
    ["MODEL_SOURCE",data.model_source],["LOADING_MODE",data.loading_mode]
  ].map(([label,value]) => stat(label,value)).join("");
}
async function loadDataset(fillExample=false) {
  const data = await api("/data/preview"); state.dataset = data;
  const description = $("dataset-description"); description.replaceChildren();
  const filename = document.createElement("code"); filename.textContent = data.filename; description.append(filename);
  description.append(document.createTextNode(data.source === "upload" ? " · 최신 업로드 데이터입니다. 학습 시 이 파일을 사용합니다." : " · 샘플 미리보기입니다. 학습하려면 아래에서 CSV를 업로드하세요."));
  $("dataset-stats").innerHTML = [
    ["행 수",number(data.rows)],["시작일",data.start_date],["종료일",data.end_date],
    ["최저 종가",money(data.min_close)],["최고 종가",money(data.max_close)],["평균 거래량",number(data.avg_volume)]
  ].map(([label,value]) => stat(label,value)).join("");
  if (fillExample) $("prediction-input").value = JSON.stringify(data.example,null,2);
}
async function loadMetrics() {
  const data = await api(`/metrics/summary?window=${encodeURIComponent(state.window)}`);
  $("kpi-requests").textContent = number(data.request_count);
  $("kpi-latency").textContent = `${number(data.avg_latency_ms)} ms`;
  $("kpi-success").textContent = `${data.success_rate.toFixed(1)}%`;
  $("metric-latency").textContent = number(data.avg_latency_ms,2);
  $("metric-errors").textContent = number(data.error_rate,4);
  $("metric-requests").textContent = number(data.request_count);
  document.querySelectorAll(".window-caption").forEach((el) => {el.textContent=windowLabels[state.window];});
  sparkline("spark-requests",data.series.map((p)=>p.request_count));
  sparkline("spark-latency",data.series.map((p)=>p.avg_latency_ms));
  sparkline("spark-success",data.series.map((p)=>p.success_rate));
  $("kpi-drift").textContent = data.drift.rmse === null ? "미실행" : money(data.drift.rmse);
  $("drift-caption").textContent = data.drift.count ? `최근 ${data.drift.count}건 · 임계값 ${money(data.drift.threshold)}` : "Simulation 탭에서 배치 전송";
}
async function loadModels() {
  const [data, health] = await Promise.all([api("/models/overview"),api("/health")]);
  state.models = data;
  $("model-health").className = `pill dot ${health.model_loaded ? "ok" : "neutral"}`;
  $("model-health").textContent = health.model_loaded ? "Healthy" : "로딩 대기";
  const current = data.served_model ?? (data.source === "mlflow" ? data.production : null);
  $("current-version").textContent = current ? (current.stage === "Local" ? current.version : `v${current.version}`) : "미로딩";
  $("current-stage").textContent = current?.stage ?? (data.source === "local" ? "Local" : "—");
  $("current-mode").textContent = current?.mode === "fine-tune" ? "fine-tuning" : current?.mode ?? "—";
  const gate = state.settings?.rmse_gate;
  $("current-rmse").textContent = `${money(current?.rmse)}${gate !== undefined ? ` / 게이트 ${money(gate)}` : ""}`;
  $("current-time").textContent = registeredTime(current?.created_at);
  const notice = data.reload_required ? `Production은 v${data.production.version}로 승격됐지만 서버는 v${data.served_model.version}를 사용 중입니다. 서버를 재시작하면 최신 모델을 불러옵니다.` : !data.model_loaded ? "아직 모델이 로딩되지 않았습니다. 첫 예측 요청에서 선택한 모델을 불러옵니다." : data.source === "local" ? "로컬 baseline을 사용 중입니다. 등록 이력은 MLflow 저장소의 별도 기록입니다." : "";
  $("model-notice").textContent = notice; $("model-notice").hidden = !notice;
  $("registry-message").textContent = data.message ?? ""; $("registry-message").hidden = !data.message;
  $("kpi-rmse").textContent = money(current?.rmse);
  sparkline("spark-rmse",[...data.versions].reverse().map((v)=>v.rmse));
  $("model-history").innerHTML = data.versions.length ? data.versions.map((v)=>`<tr><td><strong>v${escapeHtml(v.version)}</strong></td><td>${escapeHtml(registeredTime(v.created_at))}</td><td>${pill(v.mode ?? "미기록")}</td><td>${money(v.rmse)}</td><td>${pill(v.stage,v.stage === "Production" ? "ok" : "neutral",v.stage === "Production")}</td></tr>`).join("") : '<tr><td colspan="5" class="table-empty">등록된 모델 버전이 없습니다.</td></tr>';
}
async function loadEvents() {
  const events = await api("/events/recent");
  $("recent-events").innerHTML = events.length ? events.slice(0,6).map((e)=>`<div class="event ${e.level === "WARNING" ? "warn" : ""}"><span class="event-icon" aria-hidden="true">${e.level === "WARNING" ? "!" : "i"}</span><div><p>${escapeHtml(e.message)}</p><small>${escapeHtml(relativeTime(e.timestamp))}</small></div></div>`).join("") : '<p class="empty">아직 기록된 이벤트가 없습니다.<br>Simulation 탭에서 배치를 전송해보세요.</p>';
}
function renderPipeline() {
  let statuses = stages.map(()=>["idle","대기 중"]);
  if (state.lastRun) {
    const check=state.lastRun.check;
    statuses = [["ok","완료"],["ok","완료"],["ok","정상"],["idle","불필요"],["idle","대기 중"],["idle","대기 중"],["idle","대기 중"]];
    if(check.status === "retrain_triggered") {
      statuses = [["ok","완료"],["ok","완료"],["warn","감지됨"],["ok","완료"],["ok","완료"],[check.promoted ? "ok" : "warn",check.promoted ? "등록 완료" : "게이트 미통과"],[check.promoted ? "ok" : "idle",check.promoted ? "승격 완료" : "기존 버전 유지"]];
    }
  }
  $("pipeline-run-badge").textContent = state.lastRun ? `마지막 실행 ${relativeTime(state.lastRun.ts)}` : "아직 실행 안 함";
  $("pipeline-track").innerHTML=stages.map(([icon,title,desc],i)=>`<div class="pipe-stage ${statuses[i][0]}"><div class="pipe-icon" aria-hidden="true">${icon}</div><div class="pipe-title">${title}</div><div class="pipe-desc">${desc}</div><div class="pipe-status">${statuses[i][1]}</div></div>`).join("");
}
function setBusy(value) {
  state.busy=value; document.querySelectorAll(".inference-action").forEach((button)=>{button.disabled=value;});
}
async function refreshDashboard() {
  if(state.refreshing) return;
  state.refreshing=true;
  try {
    const results=await Promise.allSettled([loadMetrics(),loadModels(),loadEvents()]);
    const failed=results.find((r)=>r.status === "rejected");
    if(failed) displayError(failed.reason,"connection-error"); else $("connection-error").hidden=true;
  } finally {state.refreshing=false;}
}
async function predict() {
  if(state.busy) return;
  let payload;
  try {payload=JSON.parse($("prediction-input").value);} catch {$("prediction-status").innerHTML=pill("JSON 형식을 확인하세요.","error");return;}
  setBusy(true);$("prediction-status").innerHTML=pill("예측 중…");$("prediction-result").hidden=true;
  try {
    const data=await api("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    $("prediction-status").innerHTML=pill(`예측 완료 · 다음날 종가 ${money(data.predicted_close)}`,"ok",true);
    showResult("prediction-result",data);
  } catch(e) {$("prediction-status").innerHTML=pill(`예측 실패${e.status ? ` (HTTP ${e.status})` : ""}`,"error",true);showResult("prediction-result",e.data ?? {detail:e.message},true);}
  finally {setBusy(false);await refreshDashboard();}
}
function randn(){let u=0,v=0;while(u===0)u=Math.random();while(v===0)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
async function sendBatch(kind) {
  if(state.busy) return;
  if(!state.settings || !state.dataset){$("drift-status").innerHTML=pill("서버 설정과 데이터부터 새로고침하세요.","warn");return;}
  setBusy(true);$("drift-status").innerHTML=pill("전송 중… 재학습이 실행되면 시간이 걸릴 수 있습니다.","warn");$("drift-result").hidden=true;
  const base=state.dataset.example.sequence.at(-1)?.close;
  const sigma=kind === "drift" ? .036 : .012;
  const prices=[];let logPrice=Math.log(base);
  for(let i=0;i<state.settings.seq_len+state.settings.window_size;i++){logPrice+=randn()*sigma;prices.push(Math.exp(logPrice));}
  try {
    const data=await api("/predict/batch-test",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prices})});
    const check=data.drift_check ?? {};
    state.lastRun={check,ts:Date.now()/1000};renderPipeline();
    const actual=prices.slice(state.settings.seq_len),predictions=data.predictions ?? [];
    const rmse=actual.length === predictions.length && actual.length ? Math.sqrt(actual.reduce((sum,p,i)=>sum+(p-predictions[i])**2,0)/actual.length) : null;
    let text="정상 · 드리프트 없음",level="ok";
    if(check.status === "retrain_triggered"){text=check.promoted ? `재학습 완료 · Production 승격 (RMSE ${money(check.rmse)})` : `드리프트 감지 · 재학습 게이트 미통과 (RMSE ${money(check.rmse)})`;level=check.promoted ? "ok" : "warn";}
    $("drift-status").innerHTML=`${pill(text,level,true)} ${pill(`배치 RMSE ${money(rmse)}`)}`;
    showResult("drift-result",data);
  } catch(e) {$("drift-status").innerHTML=pill(`배치 처리 실패${e.status ? ` (HTTP ${e.status})` : ""} · 서버 로그를 확인하세요.`,"error",true);showResult("drift-result",e.data ?? {detail:e.message},true);}
  finally {setBusy(false);await refreshDashboard();}
}
async function upload() {
  const file=$("upload-input").files[0];if(!file){$("upload-status").innerHTML=pill("CSV 파일을 먼저 선택하세요.","warn");return;}
  const form=new FormData();form.append("file",file);$("upload-file").disabled=true;$("upload-status").innerHTML=pill("업로드 중…");
  try {
    const data=await api("/data/upload",{method:"POST",body:form});
    $("upload-status").innerHTML=pill(`업로드 완료 · ${data.filename} (${number(data.rows)}행)`,"ok",true);
    await loadDataset(true);
  } catch(e){$("upload-status").innerHTML=pill(`업로드 실패: ${e.message}`,"error",true);}
  finally{$("upload-file").disabled=false;}
}
function onClick(id,action){$(id).addEventListener("click",()=>Promise.resolve().then(action).catch((e)=>displayError(e,"connection-error")));}
async function init() {
  updateClock();renderPipeline();
  document.querySelectorAll("[data-tab]").forEach((button)=>button.addEventListener("click",()=>{location.hash=button.dataset.tab;switchTab(button.dataset.tab);}));
  document.querySelector(".tabs").addEventListener("keydown",(e)=>{
    if(!["ArrowLeft","ArrowRight","Home","End"].includes(e.key))return;
    e.preventDefault();const names=["dashboard","simulation","datasets","system"],index=names.indexOf(state.tab);
    const next=e.key === "Home" ? 0 : e.key === "End" ? 3 : (index+(e.key === "ArrowRight" ? 1 : 3))%4;
    location.hash=names[next];switchTab(names[next],true);
  });
  window.addEventListener("hashchange",()=>switchTab(location.hash.slice(1)));
  document.querySelectorAll("[data-window]").forEach((button)=>button.addEventListener("click",()=>{
    state.window=button.dataset.window;document.querySelectorAll("[data-window]").forEach((b)=>{b.classList.toggle("selected",b===button);b.setAttribute("aria-pressed",b===button);});
    loadMetrics().catch((e)=>displayError(e,"connection-error"));
  }));
  onClick("refresh-models",refreshDashboard);onClick("refresh-metrics",loadMetrics);onClick("refresh-datasets",()=>loadDataset());onClick("refresh-system",loadSystem);
  onClick("regenerate-example",()=>loadDataset(true));onClick("run-prediction",predict);onClick("send-normal",()=>sendBatch("normal"));onClick("send-drift",()=>sendBatch("drift"));onClick("upload-file",upload);
  const initialized=await Promise.allSettled([loadSystem(),loadDataset(true)]);
  const failed=initialized.find((r)=>r.status === "rejected");
  await refreshDashboard();
  if(failed)displayError(failed.reason,"connection-error");
  switchTab(location.hash.slice(1));
  setInterval(updateClock,1000);
  setInterval(()=>{renderPipeline();if($("auto-refresh").checked && !state.busy && !document.hidden && ["dashboard","simulation"].includes(state.tab)){loadMetrics().catch((e)=>displayError(e,"connection-error"));}},5000);
  setInterval(()=>{if(!state.busy && !document.hidden && state.tab === "dashboard")refreshDashboard();},30000);
}
init().catch((e)=>displayError(e,"connection-error"));
