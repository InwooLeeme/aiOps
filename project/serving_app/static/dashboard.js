"use strict";
const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "—").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const number = (value, digits = 0) => Number.isFinite(value) ? value.toLocaleString("ko-KR", {maximumFractionDigits: digits}) : "—";
const energy = (value) => Number.isFinite(value) ? `${number(value, 2)} MWh` : "—";
const wapeText = (value, recorded = value !== undefined) => Number.isFinite(value) ? `${number(value, 2)}%` : recorded ? "계산 불가" : "미기록";
const windowLabels = {"5m":"최근 5분","1h":"최근 1시간","6h":"최근 6시간","24h":"최근 24시간"};
const state = {tab:"dashboard", window:"5m", settings:null, dataset:null, models:null, lastRun:null, busy:false, refreshing:false};
const stages = [
  ["01","관측값 연결","저장된 예측과 비교"],["02","데이터 모니터링","예측 기록 누적"],
  ["03","성능 저하 감지","RMSE vs 임계치"],["04","재학습 요청","감지 시 자동 실행"],
  ["05","모델 학습","후보 학습·검증"],["06","모델 등록","MLflow Registry"],["07","배포","Production 승격"]
];
function pill(text, kind="neutral", dot=false) { return `<span class="pill ${kind}${dot ? " dot" : ""}">${escapeHtml(text)}</span>`; }
function relativeTime(ts) {
  const seconds = Math.max(0, Math.floor(Date.now()/1000-ts));
  return seconds < 60 ? `${seconds}초 전` : seconds < 3600 ? `${Math.floor(seconds/60)}분 전` : seconds < 86400 ? `${Math.floor(seconds/3600)}시간 전` : `${Math.floor(seconds/86400)}일 전`;
}
function registeredTime(ts) {
  if (!ts) return "—";
  return new Intl.DateTimeFormat("ko-KR", {month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",timeZone:"Asia/Seoul"}).format(new Date(ts*1000));
}
function dataTime(value) { return value ? String(value).slice(0,10) : "미기록"; }
function validationPeriod(model) { return model?.validation_start && model?.validation_end ? `${dataTime(model.validation_start)} ~ ${dataTime(model.validation_end)}` : "검증 기간 미기록"; }
function comparisonText(model) {
  if (!Number.isFinite(model?.incumbent_rmse) || !Number.isFinite(model?.rmse)) return "기존 모델 비교 미기록";
  const difference=model.incumbent_rmse-model.rmse;
  const change=model.incumbent_rmse > 0 ? `${number(Math.abs(difference)/model.incumbent_rmse*100,1)}% ${difference > 0 ? "개선" : difference < 0 ? "악화" : "동일"}` : `차이 ${energy(difference)}`;
  return `${model.parent_version ? `v${model.parent_version} ` : "기존 "}${energy(model.incumbent_rmse)} → ${energy(model.rmse)} · ${change}`;
}
const modeLabel=(mode)=>({"fine-tune":"자동 재학습","fine_tune":"자동 재학습","scratch":"초기 학습","seasonal_upgrade":"모델 개선","seasonal-upgrade":"모델 개선"})[mode] ?? mode ?? "미기록";
function renderDataContext(data) {
  const context=data.forecast_context;
  const status=!data.example ? "예측 가능한 연속 관측 구간 없음" : context === "current" ? "최근 관측 구간" : context === "future_input" ? "미래 시각 포함 · 입력 확인 필요" : "과거 데이터 · 현재 시각 예측 아님";
  const text=`${status} · 입력 기준 ${dataTime(data.input_end_timestamp)} → 예측 대상 ${dataTime(data.target_timestamp)} (KST)`;
  $("data-context").textContent=text;
  $("prediction-context").textContent=text;
}
function renderGate(trained) {
  if(!trained){$("gate-comparison").innerHTML='<div class="gate-empty"><strong>모델 비교 대기</strong><span>재학습이 끝나면 동일 기간의 후보·기존 모델·기준 예측을 비교합니다.</span></div>';return;}
  const scores=trained.metrics?.validation;
  const registered=state.models?.versions?.find(v=>String(v.version)===String(trained.version));
  const period={validation_start:trained.validation_start ?? registered?.validation_start,validation_end:trained.validation_end ?? registered?.validation_end};
  const rows=[["후보 모델",scores?.model],["기존 운영 모델",scores?.incumbent],["전날 발전량 유지",scores?.persistence],["최근 7일 평균",scores?.weekly_mean]];
  $("gate-comparison").innerHTML=`<div class="gate-heading"><div><h3>동일 기간 검증 비교</h3><p>${escapeHtml(validationPeriod(period))} · 낮을수록 좋음</p></div>${pill(trainingOutcome(trained),trained.promoted ? "ok" : "warn")}</div><div class="table-wrap"><table><thead><tr><th>비교 대상</th><th>RMSE · 승격 기준</th><th>WAPE · 보조 지표</th></tr></thead><tbody>${rows.map(([name,score])=>`<tr><td>${escapeHtml(name)}</td><td>${energy(score?.rmse)}</td><td>${wapeText(score?.wape_pct)}</td></tr>`).join("")}</tbody></table></div><p class="inline-note">후보 RMSE가 기존 모델과 두 기준 예측보다 모두 낮아야 검증을 통과합니다.${trained.reason ? ` ${escapeHtml(trained.reason)}` : ""}</p>`;
}
function renderMonitoringSummary(check) {
  const label=check.status === "threshold_unavailable" ? "기준 확인 필요" : !check.count ? "관측 대기" : !check.ready ? "관측 누적 중" : check.status === "performance_degraded" ? "성능 저하" : "정상 범위";
  $("observation-count").textContent=`${number(check.count)} / 14일`;
  $("monitor-state").textContent=label;
  $("monitor-state").className=`metric-sub ${check.status === "performance_degraded" || check.status === "threshold_unavailable" ? "status-warn" : check.ready ? "status-ok" : ""}`;
}
function renderModelSummary(current) {
  $("summary-model").textContent=current ? (current.stage === "Local" ? "로컬 모델" : `v${current.version}`) : "미로딩";
  $("summary-origin").textContent=current ? `${current.stage ?? "스테이지 미확인"} · ${current.simulation ? "시연 데이터 학습" : "관측 데이터 학습"}` : "모델을 확인하세요";
  const end=current?.validation_end;
  const input=$("simulation-start");
  if(end){
    const day=new Date(`${end.slice(0,10)}T00:00:00Z`);day.setUTCDate(day.getUTCDate()+1);
    if(Number.isFinite(day.getTime())){
      input.min=day.toISOString().slice(0,10);
      if(input.value && input.value < input.min)input.value="";
      $("simulation-boundary").textContent=`현재 모델 검증 종료: ${dataTime(end)}. ${input.min} 이후 중 결측 없는 구간을 선택하세요.`;
      return;
    }
  }
  input.min="";
  $("simulation-boundary").textContent="모델 검증 종료일을 확인할 수 없습니다. 모델 정보를 먼저 확인하세요.";
}
async function loadForecasts() {
  const data=await api("/predictions/recent");
  const check=data.monitoring;
  renderMonitoringSummary(check);
  $("kpi-drift").textContent=check.rmse === null ? "관측 대기" : energy(check.rmse);
  $("operating-wape").textContent=`WAPE ${check.count ? wapeText(check.wape_pct) : "관측 대기"}`;
  $("drift-caption").textContent=`현재 모델 · ${check.count}/14일 · ${check.ready ? "평가 가능" : "관측 누적 중"}`;
  const last=data.last_evaluation;
  if(last?.check && (!state.lastRun || last.completed_at >= state.lastRun.ts)){
    state.lastRun={kind:"observation",check:last.check,retraining:last.check.retraining,ts:last.completed_at};renderPipeline();renderGate(last.check.retraining);
  }
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
  $("model-source").textContent = data.model_source === "mlflow" ? "MLflow Registry" : "로컬 모델";
  $("sequence-length").textContent = data.seq_len;
  $("minimum-rows").textContent = data.seq_len;
  $("system-stats").innerHTML = [
    ["입력 관측 기간",`${data.seq_len}일`],["입력 피처",data.n_features],["예측 단위",data.unit],["예측 대상",`다음 ${data.horizon_days}일 총발전량`],["기준 모델 RMSE",energy(data.rmse_gate)],["성능 평가 기간",`${data.window_size}일`],
    ["성능 저하 임계값",energy(data.rmse_threshold)],["BASE EPOCHS",data.base_epochs],
    ["FINE-TUNE EPOCHS",data.fine_tune_epochs],["FINE-TUNE LR",data.fine_tune_lr],
    ["MODEL_SOURCE",data.model_source],["LOADING_MODE",data.loading_mode]
  ].map(([label,value]) => stat(label,value)).join("");
}
async function loadDataset(fillExample=false) {
  let data;
  try { data = await api("/data/preview"); }
  catch (error) { if (error.status === 404) data = {exists:false}; else throw error; }
  state.dataset = data;
  renderDataContext(data);
  const description = $("dataset-description"); description.replaceChildren();
  if (!data.exists) {
    description.textContent = "업로드된 제주 태양광 데이터가 없습니다. CSV를 업로드하세요.";
    $("dataset-stats").replaceChildren();
    if (fillExample) $("prediction-input").value = "";
    $("prediction-status").textContent = "데이터를 업로드한 뒤 실제 입력 구간을 불러오세요.";
    return;
  }
  const filename = document.createElement("code"); filename.textContent = data.filename; description.append(filename);
  description.append(document.createTextNode(` · ${data.source === "sample" ? "기본 샘플" : "업로드"} · ${data.region} 일별 데이터 · 날짜는 KST입니다.`));
  $("dataset-stats").innerHTML = [
    ["행 수",number(data.rows)],["시작 날짜",dataTime(data.start_date)],["종료 날짜",dataTime(data.end_date)],
    ["최소 발전량",energy(data.min_generation_mwh)],["최대 발전량",energy(data.max_generation_mwh)],
    ["설비용량",`${number(data.capacity_mw,2)} MW`],["누락 날짜",number(data.missing_days)], ["완전한 관측일",number(data.complete_days)], ["결측 포함 날짜",number(data.incomplete_days)],
    ["발전량 결측",number(data.missing_targets)],
    ...(Number.isFinite(data.excluded_windows) ? [["제외된 입력 구간",number(data.excluded_windows)]] : [])
  ].map(([label,value]) => stat(label,value)).join("");
  if (fillExample) {
    $("prediction-input").value = data.example ? JSON.stringify(data.example,null,2) : "";
    $("prediction-status").textContent = data.example ? "실제 데이터에서 연속 14일 입력을 불러왔습니다." : "결측값 없이 연속된 14일 구간이 없습니다. 데이터 누락을 확인하세요.";
  }
}
async function loadMetrics() {
  const data = await api(`/metrics/summary?window=${encodeURIComponent(state.window)}`);
  $("kpi-requests").textContent = number(data.request_count);
  $("kpi-latency").textContent = data.request_count ? `${number(data.avg_latency_ms)} ms` : "—";
  $("kpi-success").textContent = data.request_count ? `${data.success_rate.toFixed(1)}%` : "—";
  $("metric-latency").textContent = data.request_count ? number(data.avg_latency_ms,2) : "—";
  $("metric-errors").textContent = data.request_count ? `${number(data.error_rate*100,1)}%` : "—";
  $("metric-requests").textContent = number(data.request_count);
  const empty=data.request_count ? "" : `${windowLabels[state.window]} 동안 집계할 요청이 없습니다. 예측 또는 시연 요청이 완료되면 표시됩니다.`;
  $("requests-empty").textContent=empty;$("metrics-empty").textContent=empty;
  document.querySelectorAll(".window-caption").forEach((el) => {el.textContent=windowLabels[state.window];});
  sparkline("spark-requests",data.series.map((p)=>p.request_count));
  sparkline("spark-latency",data.series.map((p)=>p.request_count ? p.avg_latency_ms : null));
  sparkline("spark-success",data.series.map((p)=>p.request_count ? p.success_rate : null));

}
async function loadModels() {
  const [data, health] = await Promise.all([api("/models/overview"),api("/health")]);
  state.models = data;
  $("current-model-name").textContent = data.model_name;
  const current = data.served_model ?? (data.source === "mlflow" ? data.production : null);
  renderModelSummary(current);
  const experimental = data.source === "local" && current?.gate_passed === false;
  $("model-health").className = `pill dot ${experimental ? "warn" : health.model_loaded ? "ok" : "neutral"}`;
  $("model-health").textContent = experimental ? "검증 미통과 · 실험용" : health.model_loaded ? "로딩됨" : "로딩 대기";
  $("current-version").textContent = current ? (current.stage === "Local" ? current.version : `v${current.version}`) : "미로딩";
  $("current-stage").textContent = current?.stage ?? (data.source === "local" ? "Local" : "—");
  $("current-mode").textContent = modeLabel(current?.mode);
  $("current-rmse").textContent = energy(current?.rmse);
  $("current-wape").textContent=wapeText(current?.wape_pct,current?.wape_recorded ?? false);
  $("validation-wape").textContent=`WAPE ${wapeText(current?.wape_pct,current?.wape_recorded ?? false)}`;
  $("current-period").textContent=validationPeriod(current);
  $("rmse-caption").textContent=validationPeriod(current);
  $("current-time").textContent = registeredTime(current?.created_at);
  const notice = current?.simulation ? "시연 데이터로 학습한 모델이 서빙 중입니다. 이 모델의 예측은 실데이터 운영 성능 집계에서 제외합니다." : experimental ? "검증 기준을 통과하지 못한 실험용 로컬 모델입니다. Production으로 승격되지 않았습니다." : data.reload_required ? `Production은 v${data.production.version}로 승격됐지만 서버는 v${data.served_model.version}를 사용 중입니다. 서버를 재시작하면 최신 모델을 불러옵니다.` : !data.model_loaded ? "아직 모델이 로딩되지 않았습니다. 첫 예측 요청에서 선택한 모델을 불러옵니다." : data.source === "local" ? "로컬 baseline을 사용 중입니다. 등록 이력은 MLflow 저장소의 별도 기록입니다." : "";
  $("model-notice").textContent = notice; $("model-notice").hidden = !notice;
  $("registry-message").textContent = data.message ?? ""; $("registry-message").hidden = !data.message;
  $("kpi-rmse").textContent = energy(current?.rmse);
  $("spark-rmse").replaceChildren();
  $("model-history").innerHTML = data.versions.length ? data.versions.map((v)=>`<tr><td><strong>v${escapeHtml(v.version)}</strong></td><td>${escapeHtml(registeredTime(v.created_at))}</td><td>${pill(modeLabel(v.mode))}<br><small>${v.simulation ? "시연 데이터" : "학습 데이터"}</small></td><td class="period-cell">${escapeHtml(validationPeriod(v))}</td><td>${energy(v.rmse)}<br><small>WAPE ${wapeText(v.wape_pct,v.wape_recorded ?? false)}</small></td><td>${escapeHtml(comparisonText(v))}</td><td>${pill(v.stage,v.stage === "Production" ? "ok" : "neutral",v.stage === "Production")}</td></tr>`).join("") : '<tr><td colspan="7" class="table-empty">등록된 모델 버전이 없습니다.</td></tr>';
}
async function loadEvents() {
  const events = await api("/events/recent");
  $("recent-events").innerHTML = events.length ? events.slice(0,6).map((e)=>`<div class="event ${e.level === "WARNING" ? "warn" : ""}"><span class="event-icon" aria-hidden="true">${e.level === "WARNING" ? "!" : "i"}</span><div><p>${escapeHtml(e.message)}</p><small>${escapeHtml(relativeTime(e.timestamp))}</small></div></div>`).join("") : '<p class="empty">아직 기록된 이벤트가 없습니다.<br>Simulation 탭에서 정상·드리프트 배치를 실행해보세요.</p>';
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
    const trained=state.lastRun.retraining ?? check.retraining;
    if (trained) {
      const trainedOk=["promoted","gate_rejected","activation_failed"].includes(trained.status);
      statuses[3]=["ok","요청 완료"];
      statuses[4]=[trainedOk ? "ok" : "warn",trainedOk ? "학습 완료" : trainingOutcome(trained)];
      statuses[5]=[trained.promoted ? "ok" : "warn",trained.promoted ? "등록 완료" : trainingOutcome(trained)];
      statuses[6]=[trained.promoted ? "ok" : "idle",trained.promoted ? "운영 모델 교체 완료" : "기존 버전 유지"];
    } else if(check.ready) statuses[3]=["idle",drift ? "미실행" : "불필요"];

  }
  $("pipeline-run-badge").textContent = state.lastRun ? `${state.lastRun.kind === "simulation" ? "시연" : "운영 관측"} · ${relativeTime(state.lastRun.ts)}` : "아직 실행 안 함";
  $("pipeline-track").innerHTML=stages.map(([icon,title,desc],i)=>`<div class="pipe-stage ${statuses[i][0]}"><div class="pipe-icon" aria-hidden="true">${icon}</div><div class="pipe-title">${title}</div><div class="pipe-desc">${desc}</div><div class="pipe-status">${statuses[i][1]}</div></div>`).join("");
}
function setBusy(value) {
  state.busy=value; document.querySelectorAll(".inference-action").forEach((button)=>{button.disabled=value;});
}
async function refreshDashboard() {
  if(state.refreshing) return;
  state.refreshing=true;
  try {
    const results=await Promise.allSettled([loadMetrics(),loadSystem().then(loadModels),loadEvents(),loadForecasts()]);
    const failed=results.find((r)=>r.status === "rejected");
    if(failed) displayError(failed.reason,"connection-error"); else $("connection-error").hidden=true;
  } finally {state.refreshing=false;}
}
function renderPrediction(data, payload) {
  const model=state.models?.versions?.find(v=>String(v.version)===String(data.model_version));
  const historical=data.forecast_context === "historical";
  const contextLabel=historical ? "과거 데이터 점검" : data.monitoring_eligible ? "운영 예측" : "운영 집계 제외";
  const simulation=model?.simulation || data.exclusion_reason === "simulation_model";
  const rows=[
    ["예측 대상일",`${dataTime(data.target_timestamp)} · KST`],
    ["입력 관측 기간",`${dataTime(payload.sequence?.[0]?.timestamp)} ~ ${dataTime(data.input_end_timestamp)} · ${payload.sequence?.length ?? "—"}일`],
    ["사용 모델",`v${data.model_version}`]
  ];
  $("prediction-result-card").innerHTML=`<div class="prediction-card-head"><h3>다음 날 예상 발전량</h3><div>${pill(contextLabel,"neutral")}${simulation ? pill("시연 데이터 학습 모델","warn") : ""}</div></div><p class="prediction-energy">${number(data.predicted_generation_mwh,2)} <span>MWh</span></p><dl class="prediction-facts">${rows.map(([label,value])=>`<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>`;
  $("prediction-result-card").hidden=false;
}
async function predict() {
  if(state.busy) return;
  $("prediction-result-card").hidden=true;
  $("prediction-result-card").innerHTML="";
  $("prediction-result").hidden=true;
  $("prediction-details").open=false;
  let payload;
  try {payload=JSON.parse($("prediction-input").value);} catch {$("prediction-status").innerHTML=pill("JSON 형식을 확인하세요.","error");return;}
  setBusy(true);$("prediction-status").innerHTML=pill("예측 중…");$("prediction-result").hidden=true;
  try {
    const data=await api("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    renderPrediction(data,payload);
    $("prediction-status").innerHTML=pill("예측 결과 조회 완료","ok",true);
    showResult("prediction-result",data);
  } catch(e) {
    const detail=e.data?.detail;
    const reason=Array.isArray(detail) ? detail.map(item=>`${(item.loc ?? []).filter(v=>v!=="body").join(" · ")}: ${item.msg}`).join(" / ") : e.message;
    $("prediction-status").innerHTML=pill(`예측 실패${e.status ? ` (HTTP ${e.status})` : ""}`,"error",true)+`<p class="inline-note">${escapeHtml(reason)}</p>`;
    showResult("prediction-result",e.data ?? {detail:e.message},true);
  }
  finally {setBusy(false);await refreshDashboard();}
}
function renderBatchErrors(data) {
  const records=(data?.records ?? []).filter(r=>typeof r.timestamp === "string" && Number.isFinite(r.predicted) && Number.isFinite(r.actual) && Number.isFinite(r.predicted-r.actual)).sort((a,b)=>a.timestamp.localeCompare(b.timestamp)).slice(-14);
  if(!records.length){
    $("batch-error-context").textContent="표시할 날짜별 예측·실제값이 없습니다.";
    $("batch-error-chart").innerHTML='<p class="empty">Simulation에서 정상 또는 드리프트 배치를 실행하면 날짜별 오차를 표시합니다.</p>';
    return;
  }
  const kind=data.scenario === "drift" ? "드리프트 합성 배치" : "정상 원본 배치";
  $("batch-error-context").textContent=`평가 모델 v${data.base_model_version ?? "미기록"} · ${kind} · ${dataTime(records[0].timestamp)} ~ ${dataTime(records.at(-1).timestamp)} · ${records.length}일`;
  const extent=Math.max(1,...records.map(r=>Math.abs(r.predicted-r.actual)));
  const limit=Math.ceil(extent/10)*10, zero=150, half=108, left=72, width=820, step=width/records.length;
  const grid=[limit,0,-limit].map(value=>{
    const y=zero-value/limit*half;
    return `<line class="${value===0 ? "error-zero" : "error-grid"}" x1="${left}" x2="${left+width}" y1="${y}" y2="${y}"/><text class="error-axis" x="${left-12}" y="${y+4}" text-anchor="end">${value>0 ? "+" : ""}${number(value)}</text>`;
  }).join("");
  const bars=records.map((r,i)=>{
    const error=r.predicted-r.actual, x=left+step*(i+.5), h=Math.max(2,Math.abs(error)/limit*half), y=error>0 ? zero-h : error<0 ? zero : zero-1;
    const label=`${dataTime(r.timestamp)} · 예측 ${energy(r.predicted)} · 실제 ${energy(r.actual)} · ${error>0 ? "과대 예측 +" : error<0 ? "과소 예측 " : "오차 없음 "}${energy(error)}`;
    return `<g class="error-bar" tabindex="0" role="img" aria-label="${escapeHtml(label)}"><title>${escapeHtml(label)}</title><rect class="${error>0 ? "error-positive" : error<0 ? "error-negative" : "error-neutral"}" x="${x-Math.min(32,step*.6)/2}" y="${y}" width="${Math.min(32,step*.6)}" height="${h}" rx="3"/><text class="error-axis" x="${x}" y="278" text-anchor="middle">${escapeHtml(dataTime(r.timestamp).slice(5))}</text><text class="error-detail" x="${left}" y="308">${escapeHtml(label)}</text></g>`;
  }).join("");
  $("batch-error-chart").innerHTML=`<svg viewBox="0 0 920 324" role="group" aria-label="날짜별 예측 오차 막대 그래프. 0 위는 과대 예측, 아래는 과소 예측"><text class="error-axis" x="16" y="20">MWh</text>${grid}${bars}</svg>`;
}
function renderSimulation(data) {
  renderBatchErrors(data);
  const check=data.drift_check, trained=data.retraining;
  const ts=data.completed_at ?? Date.now()/1000;
  if(!state.lastRun || ts >= state.lastRun.ts){state.lastRun={kind:"simulation",check,retraining:trained,reloadError:data.reload_error,ts};renderPipeline();renderGate(trained);}
  let text="정상 · 자동 재학습 불필요", level="ok";
  if(trained?.promoted){text=`자동 재학습 완료 → 운영 v${data.served_model_version} 승격·서빙 교체 완료`;}
  else if(trained){text=`${trainingOutcome(trained)} · 기존 운영 모델 유지${trained.reason ? ': '+trained.reason : ''}`;level="warn";}

  $("simulation-status").innerHTML=pill(`[시뮬레이션] ${text}`,level,true)+`<p class="inline-note">최근 실행 구간: ${escapeHtml(dataTime(data.start_timestamp))} ~ ${escapeHtml(dataTime(data.cutoff_timestamp))} · 선택 중인 날짜와 다를 수 있습니다.</p>`;
  $("simulation-summary").innerHTML=[
    ["변형 전 기준 모델",`운영 v${data.base_model_version}`],
    ["배치 RMSE",energy(check.rmse)],["배치 WAPE",wapeText(check.wape_pct)],["감지 임계값",energy(check.threshold)],
    ...(trained ? [["재학습 검증 RMSE",energy(trained.rmse)],["재학습 검증 WAPE",wapeText(trained.metrics?.validation?.model?.wape_pct)]] : []),["실행 후 서빙 모델",`${data.model_name} v${data.served_model_version}`],
    ["데이터",data.scenario === "drift" ? "출력 제한 합성 데이터" : "원본 관측값"]
  ].map(([label,value])=>stat(label,value)).join("");
  showResult("simulation-result",data);
}
async function loadSimulation() {
  const data=await api("/simulation/status");
  if(data.exists) renderSimulation(data);
  else renderBatchErrors(null);
}
async function sendSimulation(scenario) {
  if(state.busy) return;
  const start=$("simulation-start").value;
  if(!start){$("simulation-status").textContent="평가 시작 날짜를 선택하세요.";return;}
  setBusy(true);
  $("simulation-summary").replaceChildren();$("simulation-result").hidden=true;
  $("simulation-status").innerHTML=pill("배치 평가 중… 성능 저하가 감지되면 자동으로 재학습·검증합니다.","warn",true);
  try {
    const data=await api("/simulation/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({scenario,start_timestamp:start})});
    renderSimulation(data);
  } catch(e){$("simulation-status").innerHTML=pill(`시뮬레이션 실패: ${e.message}`,"error",true);showResult("simulation-result",e.data ?? {detail:e.message},true);}
  finally{setBusy(false);await refreshDashboard();}
}
async function upload() {
  const file=$("upload-input").files[0];if(!file){$("upload-status").innerHTML=pill("CSV 파일을 먼저 선택하세요.","warn");return;}
  const form=new FormData();form.append("file",file);$("upload-file").disabled=true;$("upload-status").innerHTML=pill("업로드 중…");
  try {
    const data=await api("/data/upload",{method:"POST",body:form});
    $("upload-status").innerHTML=pill(`업로드 완료 · ${data.filename} (${number(data.rows)}행) · 예측 ${number(data.feedback?.matched ?? 0)}건에 실제값 연결${data.feedback?.status === "busy" ? " · 평가 대기: 작업 종료 후 다시 업로드하세요" : ""}`,"ok",true);
    if(data.feedback?.message){$("upload-status").innerHTML+=` ${pill(data.feedback.message,"warn")}`;}
    if(data.feedback?.check?.retraining){$("upload-status").innerHTML+=` ${pill(trainingOutcome(data.feedback.check.retraining),data.feedback.check.retraining.promoted ? "ok" : "warn")}`;}
    await loadDataset(true);await refreshDashboard();
  } catch(e){$("upload-status").innerHTML=pill(`업로드 실패: ${e.message}`,"error",true);}
  finally{$("upload-file").disabled=false;}
}
function onClick(id,action){$(id).addEventListener("click",()=>Promise.resolve().then(action).catch((e)=>displayError(e,"connection-error")));}
async function init() {
  updateClock();renderPipeline();renderGate(null);
  document.querySelectorAll("[data-tab]").forEach((button)=>button.addEventListener("click",()=>{location.hash=button.dataset.tab;switchTab(button.dataset.tab);}));
  document.querySelector(".tabs").addEventListener("keydown",(e)=>{
    if(!["ArrowLeft","ArrowRight","Home","End"].includes(e.key))return;
    e.preventDefault();const names=["dashboard","simulation","datasets","system"],index=names.indexOf(state.tab);
    const next=e.key === "Home" ? 0 : e.key === "End" ? 3 : (index+(e.key === "ArrowRight" ? 1 : 3))%4;
    location.hash=names[next];switchTab(names[next],true);
  });
  window.addEventListener("hashchange",()=>switchTab(location.hash.slice(1)));
  document.querySelectorAll("[data-window]").forEach((button)=>button.addEventListener("click",()=>{
    state.window=button.dataset.window;document.querySelectorAll("[data-window]").forEach((b)=>{const selected=b.dataset.window===state.window;b.classList.toggle("selected",selected);b.setAttribute("aria-pressed",selected);});
    loadMetrics().catch((e)=>displayError(e,"connection-error"));
  }));
  onClick("refresh-models",refreshDashboard);onClick("refresh-metrics",loadMetrics);onClick("refresh-datasets",()=>loadDataset());onClick("refresh-system",loadSystem);
  onClick("regenerate-example",()=>loadDataset(true));onClick("run-prediction",predict);onClick("upload-file",upload);
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
