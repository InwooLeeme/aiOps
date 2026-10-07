"use strict";
const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "—").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const number = (value, digits = 0) => Number.isFinite(value) ? value.toLocaleString("ko-KR", {maximumFractionDigits: digits}) : "—";
const energy = (value) => Number.isFinite(value) ? `${number(value, 2)} MWh` : "—";
const windowLabels = {"5m":"최근 5분","1h":"최근 1시간","6h":"최근 6시간","24h":"최근 24시간"};
const state = {tab:"dashboard", window:"5m", settings:null, dataset:null, models:null, lastRun:null, busy:false, refreshing:false};
const stages = [
  ["📥","데이터 수집","요청 수신"],["📊","데이터 모니터링","예측 기록 누적"],
  ["🔍","성능 저하 감지","RMSE vs 임계치"],["⚙️","재학습 요청","감지 시 자동 실행"],
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
  $("minimum-rows").textContent = data.seq_len;
  $("system-stats").innerHTML = [
    ["입력 시간",data.seq_len],["입력 피처",data.n_features],["예측 단위",data.unit],["예측 시간",`${data.horizon_hours}시간 후`],["기준 모델 RMSE",energy(data.rmse_gate)],["드리프트 윈도우",data.window_size],
    ["성능 저하 임계값",energy(data.rmse_threshold)],["BASE EPOCHS",data.base_epochs],
    ["FINE-TUNE EPOCHS",data.fine_tune_epochs],["FINE-TUNE LR",data.fine_tune_lr],
    ["MODEL_SOURCE",data.model_source],["LOADING_MODE",data.loading_mode]
  ].map(([label,value]) => stat(label,value)).join("");
}
function shiftHours(timestamp, hours) {
  const parts = timestamp.slice(0,19).split(/[-T :]/).map(Number);
  return new Date(Date.UTC(parts[0],parts[1]-1,parts[2],(parts[3] || 0)+hours,parts[4] || 0)).toISOString().slice(0,16);
}
async function loadDataset(fillExample=false) {
  let data;
  try { data = await api("/data/preview"); }
  catch (error) { if (error.status === 404) data = {exists:false}; else throw error; }
  state.dataset = data;
  const description = $("dataset-description"); description.replaceChildren();
  if (!data.exists) {
    description.textContent = "업로드된 제주 태양광 데이터가 없습니다. CSV를 업로드하세요.";
    $("dataset-stats").replaceChildren();
    if (fillExample) $("prediction-input").value = "";
    $("prediction-status").textContent = "데이터를 업로드한 뒤 실제 입력 구간을 불러오세요.";
    return;
  }
  const filename = document.createElement("code"); filename.textContent = data.filename; description.append(filename);
  description.append(document.createTextNode(` · ${data.region} 시간별 데이터 · 시각은 KST입니다.`));
  $("dataset-stats").innerHTML = [
    ["행 수",number(data.rows)],["시작 시각",data.start_date],["종료 시각",data.end_date],
    ["최소 발전량",energy(data.min_generation_mwh)],["최대 발전량",energy(data.max_generation_mwh)],
    ["설비용량",`${number(data.capacity_mw,2)} MW`],["누락 시간",number(data.missing_hours)],
    ["발전량 결측",number(data.missing_targets)],
    ...(Number.isFinite(data.excluded_windows) ? [["제외된 입력 구간",number(data.excluded_windows)]] : [])
  ].map(([label,value]) => stat(label,value)).join("");
  if (fillExample) {
    $("prediction-input").value = data.example ? JSON.stringify(data.example,null,2) : "";
    $("prediction-status").textContent = data.example ? "실제 데이터에서 연속 72시간 입력을 불러왔습니다." : "결측값 없이 연속된 72시간 구간이 없습니다. 데이터 누락을 확인하세요.";
  }
  if (fillExample || !$("replay-start").value) {
    const first = shiftHours(data.start_date, state.settings?.seq_len ?? 72);
    $("replay-start").value = first < "2024-01-01T00:00" ? "2024-01-01T00:00" : first;
  }
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
  $("kpi-drift").textContent = data.drift.rmse === null ? "미실행" : energy(data.drift.rmse);
  $("drift-caption").textContent = data.drift.count ? `최근 ${data.drift.count}건 · ${data.drift.ready ? "평가 가능" : "판정 대기"} · 임계값 ${energy(data.drift.threshold)}` : "Simulation 탭에서 과거 구간 평가";
}
async function loadModels() {
  const [data, health] = await Promise.all([api("/models/overview"),api("/health")]);
  state.models = data;
  $("current-model-name").textContent = data.model_name;
  const current = data.served_model ?? (data.source === "mlflow" ? data.production : null);
  const experimental = data.source === "local" && current?.gate_passed === false;
  $("model-health").className = `pill dot ${experimental ? "warn" : health.model_loaded ? "ok" : "neutral"}`;
  $("model-health").textContent = experimental ? "검증 미통과 · 실험용" : health.model_loaded ? "로딩됨" : "로딩 대기";
  $("current-version").textContent = current ? (current.stage === "Local" ? current.version : `v${current.version}`) : "미로딩";
  $("current-stage").textContent = current?.stage ?? (data.source === "local" ? "Local" : "—");
  $("current-mode").textContent = current?.mode === "fine-tune" ? "fine-tuning" : current?.mode ?? "—";
  const gate = state.settings?.rmse_gate;
  $("current-rmse").textContent = `${energy(current?.rmse)}${Number.isFinite(gate) ? ` / 게이트 ${energy(gate)}` : ""}`;
  $("current-time").textContent = registeredTime(current?.created_at);
  const notice = experimental ? "검증 기준을 통과하지 못한 실험용 로컬 모델입니다. Production으로 승격되지 않았습니다." : data.reload_required ? `Production은 v${data.production.version}로 승격됐지만 서버는 v${data.served_model.version}를 사용 중입니다. 서버를 재시작하면 최신 모델을 불러옵니다.` : !data.model_loaded ? "아직 모델이 로딩되지 않았습니다. 첫 예측 요청에서 선택한 모델을 불러옵니다." : data.source === "local" ? "로컬 baseline을 사용 중입니다. 등록 이력은 MLflow 저장소의 별도 기록입니다." : "";
  $("model-notice").textContent = notice; $("model-notice").hidden = !notice;
  $("registry-message").textContent = data.message ?? ""; $("registry-message").hidden = !data.message;
  $("kpi-rmse").textContent = energy(current?.rmse);
  sparkline("spark-rmse",[...data.versions].reverse().map((v)=>v.rmse));
  $("model-history").innerHTML = data.versions.length ? data.versions.map((v)=>`<tr><td><strong>v${escapeHtml(v.version)}</strong></td><td>${escapeHtml(registeredTime(v.created_at))}</td><td>${pill(v.mode ?? "미기록")}</td><td>${energy(v.rmse)}</td><td>${pill(v.stage,v.stage === "Production" ? "ok" : "neutral",v.stage === "Production")}</td></tr>`).join("") : '<tr><td colspan="5" class="table-empty">등록된 모델 버전이 없습니다.</td></tr>';
}
async function loadEvents() {
  const events = await api("/events/recent");
  $("recent-events").innerHTML = events.length ? events.slice(0,6).map((e)=>`<div class="event ${e.level === "WARNING" ? "warn" : ""}"><span class="event-icon" aria-hidden="true">${e.level === "WARNING" ? "!" : "i"}</span><div><p>${escapeHtml(e.message)}</p><small>${escapeHtml(relativeTime(e.timestamp))}</small></div></div>`).join("") : '<p class="empty">아직 기록된 이벤트가 없습니다.<br>Simulation 탭에서 과거 구간을 평가해보세요.</p>';
}
function trainingOutcome(trained) {
  return ({promoted:"운영 모델 교체 완료",gate_rejected:"검증 게이트 미통과",activation_failed:"모델 활성화 실패",failed:"학습 실패",blocked:"재학습 조건 미충족"})[trained?.status] ?? "재학습 결과 확인 필요";
}
function renderPipeline() {
  let statuses = stages.map(()=>["idle","대기 중"]);
  if (state.lastRun) {
    const check=state.lastRun.check;
    const drift=check.status === "performance_degraded";
    statuses = [["ok","완료"],["ok","완료"],[check.ready ? (drift ? "warn" : "ok") : "idle",check.ready ? (drift ? "감지됨" : "정상") : "판정 대기"],["idle","별도 실행"],["idle","대기 중"],["idle","대기 중"],["idle","대기 중"]];
    const trained=state.lastRun.kind === "retrain" ? check : state.lastRun.retraining ?? check.retraining;
    if (trained) {
      const trainedOk=["promoted","gate_rejected","activation_failed"].includes(trained.status);
      statuses[3]=["ok","요청 완료"];
      statuses[4]=[trainedOk ? "ok" : "warn",trainedOk ? "학습 완료" : trainingOutcome(trained)];
      statuses[5]=[trained.promoted ? "ok" : "warn",trained.promoted ? "등록 완료" : trainingOutcome(trained)];
      statuses[6]=[trained.promoted ? "ok" : "idle",trained.promoted ? "운영 모델 교체 완료" : "기존 버전 유지"];
    } else if(check.ready) statuses[3]=["idle",drift ? "미실행" : "불필요"];

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
    const results=await Promise.allSettled([loadMetrics(),loadSystem().then(loadModels),loadEvents()]);
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
    $("prediction-status").innerHTML=pill(`예측 완료 · ${data.target_timestamp} 발전량 ${energy(data.predicted_generation_mwh)}`,"ok",true);
    showResult("prediction-result",data);
  } catch(e) {$("prediction-status").innerHTML=pill(`예측 실패${e.status ? ` (HTTP ${e.status})` : ""}`,"error",true);showResult("prediction-result",e.data ?? {detail:e.message},true);}
  finally {setBusy(false);await refreshDashboard();}
}
async function sendBatch() {
  if(state.busy) return;
  if(!state.dataset?.exists){$("drift-status").innerHTML=pill("제주 태양광 CSV를 먼저 업로드하세요.","warn");return;}
  const start=$("replay-start").value,limit=Number($("replay-limit").value);
  if(!start || !Number.isInteger(limit) || limit < 1 || limit > 744){$("drift-status").innerHTML=pill("시작 시각과 평가 건수(1~744)를 확인하세요.","warn");return;}
  setBusy(true);$("drift-status").innerHTML=pill("실제 과거 구간 평가 중…");$("drift-result").hidden=true;
  try {
    const data=await api("/predict/batch-test",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({start_timestamp:start,limit})});
    const check=data.drift_check ?? {};
    state.lastRun={kind:"replay",check,retraining:check.retraining,ts:Date.now()/1000};renderPipeline();
    const latest = data.records?.at(-1)?.timestamp;
    if (latest) {
      $("retrain-cutoff").value = latest.slice(0,16);
      $("retrain-cutoff").max = latest.slice(0,16);
    }
    let text="판정 대기",level="neutral";
    if(check.status === "ok"){text="예측 오차가 임계값 이내입니다";level="ok";}
    else if(check.status === "performance_degraded"){text="예측 성능 저하 · 데이터 변화와 재학습 시점을 검토하세요";level="warn";}
    else if(check.status === "threshold_unavailable"){text="판정 대기 · 모델 검증 임계값이 없습니다";}
    else if(check.status === "insufficient_data"){text="판정 대기 · 최근 168시간의 완전한 예측 기록이 필요합니다";}
    else if(check.reason){text=`판정 대기 · ${check.reason}`;}
    if(check.retraining){text=trainingOutcome(check.retraining);level=check.retraining.promoted ? "ok" : "warn";}
    $("drift-status").innerHTML=`${pill(text,level,true)} ${pill(`최근 ${number(check.count)}건 RMSE ${energy(check.rmse)}`)}`;
    showResult("drift-result",data);
  } catch(e) {$("drift-status").innerHTML=pill(`평가 실패${e.status ? ` (HTTP ${e.status})` : ""}`,"error",true);showResult("drift-result",e.data ?? {detail:e.message},true);}
  finally {setBusy(false);await refreshDashboard();}
}
function renderSimulation(data) {
  const check=data.drift_check, trained=data.retraining;
  state.lastRun={kind:"simulation",check,retraining:trained,reloadError:data.reload_error,ts:data.completed_at ?? Date.now()/1000};
  renderPipeline();
  let text="정상 · 자동 재학습 불필요", level="ok";
  if(trained?.promoted){text=`자동 재학습 완료 → 운영 v${data.served_model_version} 승격·서빙 교체 완료`;}
  else if(trained){text=`${trainingOutcome(trained)} · 기존 운영 모델 유지${trained.reason ? ': '+trained.reason : ''}`;level="warn";}

  $("simulation-status").innerHTML=pill(`[시뮬레이션] ${text}`,level,true);
  $("simulation-summary").innerHTML=[
    ["변형 전 기준 모델",`운영 v${data.base_model_version}`],
    ["배치 RMSE",energy(check.rmse)],["감지 임계값",energy(check.threshold)],
    ["재학습 검증 RMSE",energy(trained?.rmse)],["실행 후 서빙 모델",`${data.model_name} v${data.served_model_version}`],
    ["데이터",data.scenario === "drift" ? "출력 제한 합성 데이터" : "원본 관측값"]
  ].map(([label,value])=>stat(label,value)).join("");
  showResult("simulation-result",data);
}
async function loadSimulation() {
  const data=await api("/simulation/status");
  if(data.exists) renderSimulation(data);
}
async function sendSimulation(scenario) {
  if(state.busy) return;
  const start=$("simulation-start").value;
  if(!start){$("simulation-status").textContent="시뮬레이션 시작 시각을 입력하세요.";return;}
  setBusy(true);
  $("simulation-summary").replaceChildren();$("simulation-result").hidden=true;
  $("simulation-status").innerHTML=pill("배치 평가 중… 성능 저하가 감지되면 자동으로 재학습·검증합니다.","warn",true);
  try {
    const data=await api("/simulation/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({scenario,start_timestamp:start})});
    renderSimulation(data);
  } catch(e){$("simulation-status").innerHTML=pill(`시뮬레이션 실패: ${e.message}`,"error",true);showResult("simulation-result",e.data ?? {detail:e.message},true);}
  finally{setBusy(false);await refreshDashboard();}
}
async function retrain() {
  if(state.busy) return;
  const cutoff=$("retrain-cutoff").value;
  if(!cutoff){$("retrain-status").innerHTML=pill("과거 구간을 먼저 평가해 학습 종료 시각을 확인하세요.","warn");return;}
  setBusy(true);$("retrain-status").innerHTML=pill("선택한 시각까지의 데이터로 재학습 중…");$("retrain-result").hidden=true;
  try {
    const data=await api("/retrain",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({cutoff_timestamp:cutoff})});
    if (typeof data.promoted === "boolean") {
      state.lastRun={kind:"retrain",check:data,ts:Date.now()/1000};renderPipeline();
      $("retrain-status").innerHTML=pill(trainingOutcome(data),data.promoted ? "ok" : "warn",true);
    } else {
      $("retrain-status").innerHTML=pill(`재학습 미실행 · ${data.reason ?? data.status ?? "응답을 확인하세요"}`,"warn",true);
    }
    showResult("retrain-result",data);
    await loadSystem();
  } catch(e) {$("retrain-status").innerHTML=pill("재학습 실패","error",true);showResult("retrain-result",e.data ?? {detail:e.message},true);}
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
  onClick("regenerate-example",()=>loadDataset(true));onClick("run-prediction",predict);onClick("send-replay",sendBatch);onClick("run-retrain",retrain);onClick("upload-file",upload);
  onClick("send-normal",()=>sendSimulation("normal"));onClick("send-drift",()=>sendSimulation("drift"));
  const initialized=await Promise.allSettled([loadSystem(),loadDataset(true)]);
  const failed=initialized.find((r)=>r.status === "rejected");
  await refreshDashboard();
  await loadSimulation();
  if(failed)displayError(failed.reason,"connection-error");
  switchTab(location.hash.slice(1));
  setInterval(updateClock,1000);
  setInterval(()=>{renderPipeline();if($("auto-refresh").checked && !state.busy && !document.hidden && ["dashboard","simulation"].includes(state.tab)){loadMetrics().catch((e)=>displayError(e,"connection-error"));}},5000);
  setInterval(()=>{if(!state.busy && !document.hidden && state.tab === "dashboard")refreshDashboard();},30000);
}
init().catch((e)=>displayError(e,"connection-error"));
