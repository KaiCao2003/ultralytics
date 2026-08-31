const state = { browser: null, defaults: {}, current: null, poller: null };

const byId = (id) => document.getElementById(id);

function csrfToken() {
  const prefix = "headplate_yolo_csrf=";
  const value = document.cookie.split("; ").find((item) => item.startsWith(prefix));
  return value ? decodeURIComponent(value.slice(prefix.length)) : "";
}

async function api(path, options = {}) {
  const request = { credentials: "same-origin", ...options };
  request.headers = { ...(options.headers || {}) };
  if (request.method && request.method !== "GET") {
    request.headers["Content-Type"] = "application/json";
    request.headers["X-CSRF-Token"] = csrfToken();
  }
  const response = await fetch(path, request);
  if (response.status === 401) {
    window.location.assign("login");
    throw new Error("Authentication required");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

function showError(error) {
  const alert = byId("alert");
  alert.textContent = error ? String(error.message || error) : "";
  alert.hidden = !error;
}

function setHidden(id, hidden) {
  byId(id).hidden = hidden;
}

function clearProject() {
  state.current = null;
  setHidden("empty-state", false);
  setHidden("project-view", true);
  stopPolling();
}

function renderFolderBrowser(browser) {
  state.browser = browser;
  state.defaults = browser.defaults;
  byId("root-path").textContent = browser.root;
  byId("current-folder-path").textContent =
    browser.current === "." ? browser.root : `${browser.root}/${browser.current}`;
  const select = byId("folder-select");
  select.replaceChildren(new Option(browser.folders.length ? "Select a subfolder" : "No subfolders", ""));
  browser.folders.forEach((folder) => select.add(new Option(folder.name, folder.path)));
  select.disabled = browser.folders.length === 0;
  byId("open-folder-button").disabled = true;
  byId("up-folder-button").disabled = browser.parent === null;
  byId("use-folder-button").disabled = browser.current === ".";
}

async function loadFolder(folder = ".", preserveProject = false) {
  try {
    showError(null);
    const browser = await api(`api/folders?folder=${encodeURIComponent(folder)}`);
    renderFolderBrowser(browser);
    if (preserveProject && state.current?.project === browser.current) await loadProject(browser.current);
    else clearProject();
  } catch (error) {
    showError(error);
  }
}

async function loadProject(name) {
  if (!name) {
    clearProject();
    return;
  }
  try {
    showError(null);
    state.current = await api(`api/project?project=${encodeURIComponent(name)}`);
    renderProject();
  } catch (error) {
    showError(error);
  }
}

function renderProject() {
  const project = state.current;
  const workflow = project.state;
  setHidden("empty-state", true);
  setHidden("project-view", false);
  byId("project-name").textContent = project.project;
  byId("video-count").textContent = String(project.videos.length);
  byId("round-number").textContent = workflow ? String(workflow.current_round || "—") : "—";
  ["initialize-form", "prepare-panel", "annotation-panel", "running-panel", "complete-panel"].forEach((id) =>
    setHidden(id, true),
  );

  if (!workflow) {
    byId("stage-badge").textContent = "Not initialized";
    byId("stage-message").textContent = "Choose videos and training settings";
    byId("action-heading").textContent = "Configure project";
    setHidden("initialize-form", false);
    setHidden("progress-shell", true);
    renderTimeline(0);
    renderConfig(project.videos);
    stopPolling();
    return;
  }

  byId("stage-badge").textContent = workflow.stage.replaceAll("_", " ");
  byId("stage-message").textContent = workflow.message || "";
  renderProgress(workflow);
  renderTimeline(stageStep(workflow));
  if (workflow.error) showError(new Error(workflow.error));

  if (workflow.stage === "CONFIGURED") {
    byId("action-heading").textContent = "Prepare the first labeling round";
    setHidden("prepare-panel", false);
  } else if (workflow.stage === "RUNNING" || workflow.active_job) {
    byId("action-heading").textContent = "Workflow running";
    setHidden("running-panel", false);
  } else if (workflow.stage.startsWith("WAITING_ROUND_")) {
    byId("action-heading").textContent = `Complete Label Studio Round ${workflow.current_round}`;
    renderAnnotationPanel(project, workflow);
    setHidden("annotation-panel", false);
  } else if (workflow.stage === "COMPLETE") {
    byId("action-heading").textContent = "Final results ready";
    renderArtifacts(workflow);
    setHidden("complete-panel", false);
  }
  loadLog(project.project);
  if (workflow.active_job || workflow.stage === "RUNNING") startPolling();
  else stopPolling();
}

function renderConfig(videos) {
  const container = byId("video-options");
  container.replaceChildren();
  if (!videos.length) {
    const empty = document.createElement("p");
    empty.className = "muted folder-empty-message";
    empty.textContent = "No videos here yet. Add video files to this folder on disk, then press Refresh.";
    container.append(empty);
  }
  videos.forEach((video, index) => {
    const label = document.createElement("label");
    label.className = "check-item";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = "video";
    input.value = video;
    input.checked = true;
    input.id = `video-${index}`;
    label.append(input, document.createTextNode(video));
    container.append(label);
  });
  byId("initialize-button").disabled = videos.length === 0;
  const defaults = state.defaults;
  byId("round1-frames").value = defaults.round1_frames;
  byId("review-frames").value = defaults.review_frames;
  byId("epochs").value = defaults.epochs;
  byId("imgsz").value = defaults.imgsz;
  byId("batch").value = defaults.batch;
  byId("device").value = defaults.device;
  byId("base-model").value = defaults.base_model;
  byId("confidence").value = defaults.conf;
  byId("json-key").value = defaults.json_key;
}

function renderProgress(workflow) {
  const visible = workflow.stage === "RUNNING" || workflow.active_job;
  setHidden("progress-shell", !visible);
  const percent = Math.max(0, Math.min(100, Math.round(Number(workflow.progress || 0) * 100)));
  byId("progress-bar").style.width = `${percent}%`;
  byId("progress-label").textContent = `${percent}%`;
}

function stageStep(workflow) {
  if (workflow.stage === "COMPLETE") return 4;
  if (workflow.current_round >= 2) return workflow.stage.startsWith("WAITING") ? 3 : 4;
  if (workflow.current_round === 1) return workflow.stage.startsWith("WAITING") ? 1 : 2;
  return 0;
}

function renderTimeline(active) {
  byId("timeline")
    .querySelectorAll("li")
    .forEach((item, index) => {
      item.classList.toggle("is-done", index < active);
      item.classList.toggle("is-active", index === active);
    });
}

function renderAnnotationPanel(project, workflow) {
  const record = workflow.rounds[String(workflow.current_round)] || {};
  byId("import-path").textContent = record.label_studio_import || "—";
  byId("config-path").textContent = record.label_studio_dir ? `${record.label_studio_dir}/label_config.xml` : "—";
  const select = byId("annotation-select");
  select.replaceChildren(new Option("Select exported JSON", ""));
  project.annotations.forEach((name) => select.add(new Option(name, name)));
  byId("process-button").disabled = project.annotations.length === 0;
}

function renderArtifacts(workflow) {
  const container = byId("artifact-list");
  container.replaceChildren();
  const record = workflow.rounds[String(workflow.current_round)] || {};
  (record.artifacts || []).forEach((artifact) => {
    const item = document.createElement("div");
    item.className = "artifact";
    const title = document.createElement("strong");
    title.textContent = artifact.video;
    item.append(title);
    const summary = document.createElement("span");
    summary.className = "artifact__summary";
    summary.textContent = `${artifact.detected_frames ?? artifact.valid_frames}/${artifact.frames} measured · ${artifact.held_frames ?? 0} held`;
    item.append(summary);
    [
      ["Pose CSV", artifact.csv],
      ["Position CSV", artifact.position_csv],
      ["HD JSON", artifact.json],
      ["Overlay", artifact.overlay],
    ].forEach(([label, path]) => {
      if (!path) return;
      const line = document.createElement("span");
      line.textContent = `${label}: ${path}`;
      item.append(line);
    });
    container.append(item);
  });
  if (record.model) {
    const item = document.createElement("div");
    item.className = "artifact";
    const title = document.createElement("strong");
    title.textContent = "Final model";
    const path = document.createElement("span");
    path.textContent = record.model;
    item.append(title, path);
    container.append(item);
  }
}

async function loadLog(project) {
  try {
    const payload = await api(`api/log?project=${encodeURIComponent(project)}`);
    byId("worker-log").textContent = payload.lines.length ? payload.lines.join("") : "No job log yet.";
  } catch (error) {
    byId("worker-log").textContent = String(error.message || error);
  }
}

function startPolling() {
  if (state.poller) return;
  state.poller = window.setInterval(() => {
    if (state.current) loadProject(state.current.project);
  }, 3000);
}

function stopPolling() {
  if (state.poller) window.clearInterval(state.poller);
  state.poller = null;
}

async function initializeProject(event) {
  event.preventDefault();
  try {
    showError(null);
    const videos = [...document.querySelectorAll('input[name="video"]:checked')].map((item) => item.value);
    const config = {
      ...state.defaults,
      round1_frames: Number(byId("round1-frames").value),
      review_frames: Number(byId("review-frames").value),
      epochs: Number(byId("epochs").value),
      imgsz: Number(byId("imgsz").value),
      batch: Number(byId("batch").value),
      device: byId("device").value.trim(),
      base_model: byId("base-model").value.trim(),
      conf: Number(byId("confidence").value),
      json_key: byId("json-key").value.trim(),
    };
    state.current = await api("api/initialize", {
      method: "POST",
      body: JSON.stringify({ project: state.current.project, videos, config }),
    });
    renderProject();
  } catch (error) {
    showError(error);
  }
}

async function createFolder(event) {
  event.preventDefault();
  try {
    showError(null);
    const browser = await api("api/folders", {
      method: "POST",
      body: JSON.stringify({ parent: state.browser.current, name: byId("new-folder-name").value }),
    });
    byId("new-folder-name").value = "";
    renderFolderBrowser(browser);
    await loadProject(browser.current);
  } catch (error) {
    showError(error);
  }
}

async function runAction(path, payload) {
  try {
    showError(null);
    state.current = await api(path, { method: "POST", body: JSON.stringify(payload) });
    renderProject();
  } catch (error) {
    showError(error);
  }
}

byId("folder-select").addEventListener("change", (event) => {
  byId("open-folder-button").disabled = !event.target.value;
});
byId("open-folder-button").addEventListener("click", () => {
  const folder = byId("folder-select").value;
  if (folder) loadFolder(folder);
});
byId("up-folder-button").addEventListener("click", () => {
  if (state.browser?.parent !== null) loadFolder(state.browser.parent);
});
byId("use-folder-button").addEventListener("click", () => loadProject(state.browser.current));
byId("refresh-button").addEventListener("click", () => loadFolder(state.browser?.current || ".", true));
byId("create-folder-form").addEventListener("submit", createFolder);
byId("initialize-form").addEventListener("submit", initializeProject);
byId("prepare-button").addEventListener("click", () => runAction("api/prepare", { project: state.current.project }));
byId("process-button").addEventListener("click", () => {
  const annotations = byId("annotation-select").value;
  if (!annotations) return showError(new Error("Select the Label Studio export JSON"));
  runAction("api/process", { project: state.current.project, annotations });
});
byId("logout-button").addEventListener("click", async () => {
  await api("logout", { method: "POST", body: "{}" });
  window.location.assign("login");
});

loadFolder();
