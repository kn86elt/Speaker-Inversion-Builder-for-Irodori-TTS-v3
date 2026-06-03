// Speaker Inversion Builder – frontend
// WaveSurfer loaded globally via wavesurfer.min.js (UMD)

// ─── Global state ────────────────────────────────────────────────────────────
var state = { files: {}, segments: [], markers: {} };
var currentFileId = null;
var wavesurfer = null;
var splitMarkers = [];   // seconds within current file
var loopEnabled = false;
var selState = { active: false, confirmed: false, startT: 0, endT: 0, startX: 0, hasDrag: false };
var settings = { irodori_root: '', uv_exe: '', checkpoint_path: '' };
var runsData = [];
var suggestedIrodoriRoot = '';

// ─── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
  bindUI();
  loadSettings();
  refreshState();
  refreshRuns();
  refreshDatasets();
  refreshDataDatasetProjects();
});

// ─── Spinner helper ───────────────────────────────────────────────────────────
function withSpinner(btn, fn) {
  var orig = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  var p;
  try {
    p = fn();
  } catch (e) {
    btn.disabled = false;
    btn.innerHTML = orig;
    setStatus('Error: ' + e.message, true);
    return Promise.reject(e);
  }
  if (p && typeof p.finally === 'function') {
    return p.catch(function (e) {
      setStatus('Error: ' + e.message, true);
    }).finally(function () {
      btn.disabled = false;
      btn.innerHTML = orig;
    });
  }
  btn.disabled = false;
  btn.innerHTML = orig;
  return Promise.resolve(p);
}

// ─── Global status bar ────────────────────────────────────────────────────────
function setStatus(msg, isError) {
  var bar = document.getElementById('status-bar');
  if (!bar) return;
  bar.textContent = msg;
  bar.style.color = isError ? 'var(--danger)' : 'var(--success)';
  bar.style.display = msg ? 'block' : 'none';
  if (!isError && msg) {
    clearTimeout(bar._t);
    bar._t = setTimeout(function () { bar.style.display = 'none'; }, 4000);
  }
}

// ─── Tab switching ────────────────────────────────────────────────────────────
function bindUI() {
  document.querySelectorAll('.tab').forEach(function (tab) {
    tab.addEventListener('click', function () {
      document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('active'); });
      document.querySelectorAll('.tab-content').forEach(function (c) { c.classList.remove('active'); });
      tab.classList.add('active');
      document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
      if (tab.dataset.tab === 'test') refreshRuns();
    });
  });

  // Upload
  var dropArea = document.getElementById('drop-area');
  var fileInput = document.getElementById('file-input');
  var datasetFolderInput = document.getElementById('dataset-folder-input');
  dropArea.addEventListener('click', function () { fileInput.click(); });
  dropArea.addEventListener('dragover', function (e) { e.preventDefault(); dropArea.classList.add('dragover'); });
  dropArea.addEventListener('dragleave', function () { dropArea.classList.remove('dragover'); });
  dropArea.addEventListener('drop', function (e) { e.preventDefault(); dropArea.classList.remove('dragover'); handleFileUpload(e.dataTransfer.files); });
  fileInput.addEventListener('change', function (e) { handleFileUpload(e.target.files); });
  document.getElementById('btn-load-project-state').addEventListener('click', function () { datasetFolderInput.click(); });
  datasetFolderInput.addEventListener('change', function (e) { importDatasetFolder(e.target.files); });
  document.getElementById('btn-clear-data-prep').addEventListener('click', clearDataPrep);

  // Playback controls
  document.getElementById('btn-play-pause').addEventListener('click', function () { if (wavesurfer) wavesurfer.playPause(); });
  document.getElementById('btn-loop').addEventListener('click', function () {
    loopEnabled = !loopEnabled;
    document.getElementById('btn-loop').style.color = loopEnabled ? 'var(--accent)' : '';
  });
  document.getElementById('volume-slider').addEventListener('input', function (e) {
    if (wavesurfer) wavesurfer.setVolume(parseFloat(e.target.value));
  });

  // Waveform interaction overlay
  bindWaveformInteraction();

  // Selection toolbar
  document.getElementById('btn-play-sel').addEventListener('click', playSelection);
  document.getElementById('btn-extract-sel').addEventListener('click', extractSelection);
  document.getElementById('btn-markers-at-sel').addEventListener('click', markersAtSelection);
  document.getElementById('btn-delete-sel').addEventListener('click', deleteSelection);
  document.getElementById('btn-clear-sel').addEventListener('click', clearSelection);

  // Segment controls
  document.getElementById('btn-add-whole').addEventListener('click', addWholeFileAsSegment);
  document.getElementById('btn-auto-markers').addEventListener('click', function () {
    withSpinner(this, detectAutoMarkers);
  });
  document.getElementById('btn-add-marker').addEventListener('click', addMarkerAtCurrent);
  document.getElementById('btn-clear-markers').addEventListener('click', clearMarkers);
  document.getElementById('btn-apply-split').addEventListener('click', function () {
    withSpinner(this, applyManualSplit);
  });

  // Segments tab
  document.getElementById('btn-transcribe-all').addEventListener('click', transcribeAll);

  // Training tab
  document.getElementById('btn-build-dataset').addEventListener('click', function () {
    withSpinner(this, function () { return buildDataset(false); });
  });
  document.getElementById('btn-refresh-data-datasets').addEventListener('click', refreshDatasets);
  document.getElementById('btn-refresh-data-datasets').addEventListener('click', refreshDataDatasetProjects);
  document.getElementById('btn-prepare-manifest').addEventListener('click', function () {
    withSpinner(this, prepareManifest);
  });
  document.getElementById('btn-train').addEventListener('click', startTraining);

  // Test tab
  document.getElementById('btn-refresh-runs').addEventListener('click', refreshRuns);
  document.getElementById('embed-select').addEventListener('change', updateStepEmbedList);
  document.getElementById('btn-generate').addEventListener('click', function () {
    withSpinner(this, generateAudio);
  });

  // Settings
  document.getElementById('btn-save-settings').addEventListener('click', saveSettings);
  document.getElementById('btn-open-root-modal').addEventListener('click', function () { showRootModal(false); });
  document.getElementById('btn-browse-root-settings').addEventListener('click', function () {
    browseIrodoriRoot('set-irodori-root', 'settings-status', this);
  });
  document.getElementById('btn-browse-root-modal').addEventListener('click', function () {
    browseIrodoriRoot('modal-irodori-root', 'root-modal-status', this);
  });
  document.getElementById('btn-save-root-modal').addEventListener('click', saveRootModal);
  document.getElementById('btn-close-root-modal').addEventListener('click', function () {
    if (!settings.irodori_root) return;
    hideRootModal();
  });
  updateSelectionToolbar();
}

// ─── Waveform interaction overlay ─────────────────────────────────────────────
function bindWaveformInteraction() {
  var overlay = document.getElementById('wf-interact');
  var selBox = document.getElementById('wf-sel');
  var DRAG_PX = 6;

  overlay.addEventListener('mousedown', function (e) {
    if (!wavesurfer) return;
    selState.active = true;
    selState.confirmed = false;
    selState.hasDrag = false;
    selState.startX = e.clientX;
    selState.startT = xToTime(e, overlay);
    selState.endT = selState.startT;
    document.getElementById('wf-sel').classList.add('hidden');
    updateSelectionToolbar();
  });

  overlay.addEventListener('mousemove', function (e) {
    if (!selState.active || !wavesurfer) return;
    if (selState.hasDrag || Math.abs(e.clientX - selState.startX) > DRAG_PX) {
      selState.hasDrag = true;
      selState.endT = clampTime(xToTime(e, overlay));
      updateSelBox(selBox);
      updateSelectionToolbar();
    }
  });

  function finishSelection(e) {
    if (!selState.active) return;
    if (!wavesurfer) { clearSelection(); return; }
    var finalT = clampTime(xToTime(e, overlay));
    if (selState.hasDrag || Math.abs(e.clientX - selState.startX) > DRAG_PX) {
      selState.hasDrag = true;
      selState.endT = finalT;
      updateSelBox(selBox);
    }
    if (!selState.hasDrag) {
      // plain click -> seek
      wavesurfer.setTime(clampTime(xToTime(e, overlay)));
      clearSelection();
      return;
    }
    selState.active = false;
    selState.endT = finalT;
    if (selEnd() - selStart() <= 0.001) {
      clearSelection();
    } else {
      selState.confirmed = true;
      updateSelBox(selBox);
      showSelToolbar();
    }
  }

  overlay.addEventListener('mouseup', finishSelection);
  document.addEventListener('mouseup', finishSelection);
  window.addEventListener('blur', function () {
    if (selState.active) clearSelection();
  });
}

function xToTime(e, el) {
  var rect = el.getBoundingClientRect();
  var ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  return ratio * (wavesurfer ? wavesurfer.getDuration() : 0);
}
function clampTime(t) {
  return Math.max(0, Math.min(wavesurfer ? wavesurfer.getDuration() : 0, t));
}
function selStart() { return Math.min(selState.startT, selState.endT); }
function selEnd()   { return Math.max(selState.startT, selState.endT); }
function hasSelectionRange() {
  return !!(wavesurfer && selState.confirmed && selEnd() - selStart() > 0.001);
}

function updateSelBox(box) {
  var dur = wavesurfer ? wavesurfer.getDuration() : 1;
  var s = selStart() / dur * 100;
  var w = (selEnd() - selStart()) / dur * 100;
  box.style.left = s + '%';
  box.style.width = w + '%';
  box.classList.remove('hidden');
}

function showSelToolbar() {
  updateSelectionToolbar();
  return;
  var tb = document.getElementById('sel-toolbar');
  var info = document.getElementById('sel-info');
  var dur = selEnd() - selStart();
  info.textContent = selStart().toFixed(2) + 's – ' + selEnd().toFixed(2) + 's (' + dur.toFixed(2) + 's)';
  tb.classList.remove('hidden');
}

function updateSelectionToolbar() {
  var info = document.getElementById('sel-info');
  var hasSelection = hasSelectionRange();
  ['btn-play-sel', 'btn-extract-sel', 'btn-markers-at-sel', 'btn-delete-sel', 'btn-clear-sel'].forEach(function (id) {
    var btn = document.getElementById(id);
    if (btn) btn.disabled = !hasSelection;
  });
  if (!info) return;
  if (!hasSelection) {
    info.textContent = 'No range selected';
    return;
  }
  var dur = selEnd() - selStart();
  info.textContent = selStart().toFixed(2) + 's - ' + selEnd().toFixed(2) + 's (' + dur.toFixed(2) + 's)';
}

function clearSelection() {
  selState.active = false;
  selState.confirmed = false;
  selState.hasDrag = false;
  document.getElementById('wf-sel').classList.add('hidden');
  document.getElementById('sel-toolbar').classList.remove('hidden');
  updateSelectionToolbar();
}

function playSelection() {
  if (!hasSelectionRange()) return;
  var s = selStart(), e = selEnd();
  wavesurfer.setTime(s);
  wavesurfer.play();
  var stopFn = function (t) { if (t >= e) { wavesurfer.pause(); wavesurfer.un('timeupdate', stopFn); } };
  wavesurfer.on('timeupdate', stopFn);
}

function extractSelection() {
  if (!currentFileId || !hasSelectionRange()) return;
  var start = selStart(), end = selEnd();
  var btn = document.getElementById('btn-extract-sel');
  withSpinner(btn, function () {
    return api('/api/file/' + encodeURIComponent(currentFileId) + '/extract_range', 'POST', { start: start, end: end })
      .then(function () { clearSelection(); refreshState(); });
  });
}

function deleteSelection() {
  if (!currentFileId || !hasSelectionRange()) return;
  if (!confirm('選択範囲を削除します。続行しますか？')) return;
  var fileId = currentFileId;
  var start = selStart(), end = selEnd();
  var btn = document.getElementById('btn-delete-sel');
  withSpinner(btn, function () {
    return api('/api/file/' + encodeURIComponent(fileId) + '/delete_range', 'POST', { start: start, end: end })
      .then(function () {
        clearMarkers();
        clearSelection();
        return refreshStatePromise();
      })
      .then(function () {
        selectFile(fileId);
        markDatasetDirty();
      });
  });
}

function markersAtSelection() {
  if (!hasSelectionRange()) return;
  [selStart(), selEnd()].forEach(function (t) {
    if (splitMarkers.indexOf(t) === -1) splitMarkers.push(t);
  });
  splitMarkers.sort(function (a, b) { return a - b; });
  renderMarkers();
  renderMarkerOverlays();
  persistMarkers();
}

// ─── Settings ─────────────────────────────────────────────────────────────────
function loadSettings() {
  api('/api/settings').then(function (data) {
    suggestedIrodoriRoot = data.suggested_irodori_root || '';
    settings.irodori_root = data.irodori_root || '';
    settings.uv_exe = data.uv_exe || '';
    settings.checkpoint_path = data.checkpoint_path || data.default_checkpoint || '';
    document.getElementById('set-irodori-root').value = settings.irodori_root;
    document.getElementById('set-irodori-root').placeholder = suggestedIrodoriRoot || 'C:/path/to/Irodori-TTS-v3';
    document.getElementById('set-uv-exe').value = settings.uv_exe;
    document.getElementById('set-checkpoint').value = settings.checkpoint_path;
    if (!settings.irodori_root) showRootModal(true);
  }).catch(function () {});
}
function saveSettings() {
  var st = document.getElementById('settings-status');
  var payload = {
    irodori_root: document.getElementById('set-irodori-root').value.trim(),
    uv_exe: document.getElementById('set-uv-exe').value.trim(),
    checkpoint_path: document.getElementById('set-checkpoint').value.trim(),
  };
  st.textContent = 'Saving...';
  st.style.color = 'var(--dim)';
  return api('/api/settings', 'POST', payload).then(function (res) {
    settings.irodori_root = res.irodori_root || payload.irodori_root;
    settings.uv_exe = res.uv_exe || payload.uv_exe;
    settings.checkpoint_path = res.checkpoint_path || payload.checkpoint_path;
    document.getElementById('set-irodori-root').value = settings.irodori_root;
    document.getElementById('set-uv-exe').value = settings.uv_exe;
    document.getElementById('set-checkpoint').value = settings.checkpoint_path;
    st.textContent = 'Saved';
    st.style.color = 'var(--success)';
    setTimeout(function () { st.textContent = ''; }, 2000);
    return res;
  }).catch(function (e) {
    st.textContent = 'Error: ' + e.message;
    st.style.color = 'var(--danger)';
    throw e;
  });
}

function showRootModal(required) {
  var modal = document.getElementById('irodori-root-modal');
  var input = document.getElementById('modal-irodori-root');
  var status = document.getElementById('root-modal-status');
  input.value = settings.irodori_root || document.getElementById('set-irodori-root').value.trim() || '';
  input.placeholder = suggestedIrodoriRoot || 'C:/path/to/Irodori-TTS-v3';
  status.textContent = required ? '初回設定が必要です' : '';
  document.getElementById('btn-close-root-modal').disabled = required && !settings.irodori_root;
  modal.classList.remove('hidden');
  setTimeout(function () { input.focus(); }, 0);
}

function hideRootModal() {
  document.getElementById('irodori-root-modal').classList.add('hidden');
}

function browseIrodoriRoot(inputId, statusId, btn) {
  var status = document.getElementById(statusId);
  var run = function () {
    if (status) {
      status.textContent = 'フォルダを選択してください...';
      status.style.color = 'var(--dim)';
    }
    return api('/api/browse/irodori_root').then(function (res) {
      if (!res.path) {
        if (status) status.textContent = '';
        return res;
      }
      document.getElementById(inputId).value = res.path;
      if (status) {
        status.textContent = '選択しました';
        status.style.color = 'var(--success)';
      }
      return res;
    }).catch(function (e) {
      if (status) {
        status.textContent = '参照エラー: ' + e.message;
        status.style.color = 'var(--danger)';
      }
      throw e;
    });
  };
  return btn ? withSpinner(btn, run) : run();
}

function saveRootModal() {
  var root = document.getElementById('modal-irodori-root').value.trim();
  var status = document.getElementById('root-modal-status');
  if (!root) {
    status.textContent = 'パスを入力してください';
    status.style.color = 'var(--danger)';
    return;
  }
  document.getElementById('set-irodori-root').value = root;
  status.textContent = 'Saving...';
  status.style.color = 'var(--dim)';
  saveSettings().then(function () {
    status.textContent = 'Saved';
    status.style.color = 'var(--success)';
    hideRootModal();
  }).catch(function (e) {
    status.textContent = 'Error: ' + e.message;
    status.style.color = 'var(--danger)';
  });
}

function ensureIrodoriRoot() {
  if (settings.irodori_root) return true;
  showRootModal(true);
  setStatus('Irodori-TTS root path is required', true);
  return false;
}

// ─── State ────────────────────────────────────────────────────────────────────
function refreshState() {
  api('/api/state').then(function (data) {
    state = data;
    state.markers = state.markers || {};
    renderFileList();
    renderSegmentList();
  }).catch(function (e) { console.error('refreshState', e); });
}

// ─── Upload ───────────────────────────────────────────────────────────────────
function handleFileUpload(fileList) {
  if (!fileList || !fileList.length) return;
  var form = new FormData();
  for (var i = 0; i < fileList.length; i++) form.append('files', fileList[i]);
  var drop = document.getElementById('drop-area');
  drop.innerHTML = '<span class="spinner"></span>';
  api('/api/upload', 'POST', form).then(function (res) {
    return refreshStatePromise().then(function () {
      if (res.added && res.added.length) selectFile(res.added[0]);
    });
  }).catch(function (e) {
    alert('Upload error: ' + e.message);
  }).finally(function () {
    drop.innerHTML = '<div class="drop-label">WAV / MP3 をドロップ<br>またはクリックして選択</div><input type="file" id="file-input" multiple accept=".wav,.mp3" hidden>';
    document.getElementById('file-input').addEventListener('change', function (e) { handleFileUpload(e.target.files); });
  });
}

// ─── File list ────────────────────────────────────────────────────────────────
function importDatasetFolder(fileList) {
  if (!fileList || !fileList.length) return;
  if (!confirm('現在のデータ準備状態をクリアして、選択したプロジェクトを読み込みます。続行しますか？')) return;
  var form = new FormData();
  for (var i = 0; i < fileList.length; i++) {
    var file = fileList[i];
    form.append('files', file, file.webkitRelativePath || file.name);
  }
  var btn = document.getElementById('btn-load-project-state');
  withSpinner(btn, function () {
    return api('/api/import_dataset_folder', 'POST', form).then(function (res) {
      document.getElementById('job-name').value = res.name || '';
      document.getElementById('speaker-name').value = res.speaker || '';
      return refreshStatePromise().then(function () {
        markDatasetDirty();
        if (res.first_file_id) selectFile(res.first_file_id);
        setStatus('Loaded dataset folder: ' + (res.name || '') + ' (' + res.segment_count + ' segments)', false);
      });
    });
  }).finally(function () {
    document.getElementById('dataset-folder-input').value = '';
  });
}

function clearDataPrep() {
  if (!confirm('現在のデータ準備状態をクリアします。続行しますか？')) return;
  api('/api/clear_data_prep', 'POST', {}).then(function () {
    currentFileId = null;
    splitMarkers = [];
    clearSelection();
    if (wavesurfer) { wavesurfer.destroy(); wavesurfer = null; }
    document.getElementById('waveform-filename').textContent = '';
    document.getElementById('waveform-time').textContent = 'ファイルを選択してください';
    document.getElementById('waveform-controls').classList.add('hidden');
    document.getElementById('file-info-bar').classList.add('hidden');
    resetDatasetBuildForm();
    return refreshStatePromise();
  }).then(function () {
    clearDatasetDirty();
    setStatus('Data preparation cleared', false);
  }).catch(function (e) {
    setStatus('Clear error: ' + e.message, true);
  });
}

function renderFileList() {
  var el = document.getElementById('file-list');
  el.innerHTML = '';
  var files = Object.values(state.files);
  if (!files.length) {
    el.innerHTML = '<div class="dim" style="font-size:12px;padding:8px">No files</div>';
    return;
  }
  files.forEach(function (f) {
    var segCount = state.segments.filter(function (s) { return s.file_id === f.id; }).length;
    var div = document.createElement('div');
    div.className = 'file-item' + (f.id === currentFileId ? ' active' : '');
    div.dataset.id = f.id;
    div.innerHTML =
      '<span class="file-name" title="' + esc(f.name) + '">' + esc(f.name) + '</span>' +
      '<span class="file-dur">' + fmtDur(f.duration) + '</span>' +
      (segCount ? '<span class="seg-count-badge">' + segCount + '</span>' : '') +
      '<button class="del-btn" data-id="' + f.id + '" title="Delete">&#10005;</button>';
    div.addEventListener('click', function (e) {
      if (!e.target.classList.contains('del-btn')) selectFile(f.id);
    });
    div.querySelector('.del-btn').addEventListener('click', function (e) {
      e.stopPropagation(); deleteFile(f.id);
    });
    el.appendChild(div);
  });
}

// ─── Select file → load waveform ──────────────────────────────────────────────
function selectFile(fileId) {
  currentFileId = fileId;
  restoreMarkersForFile(fileId);
  clearSelection();
  renderFileList();

  var f = state.files[fileId];
  document.getElementById('waveform-controls').classList.remove('hidden');
  document.getElementById('waveform-filename').textContent = f ? f.name : '';
  document.getElementById('waveform-time').textContent = 'Loading...';
  document.getElementById('waveform-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  document.getElementById('file-info-bar').classList.add('hidden');

  if (wavesurfer) { wavesurfer.destroy(); wavesurfer = null; }

  wavesurfer = WaveSurfer.create({
    container: '#waveform-inner',
    interact: false,  // we handle all interaction via #wf-interact overlay
    waveColor: '#4a9eff',
    progressColor: '#1a6fc4',
    cursorColor: '#ffffff',
    barWidth: 2,
    barGap: 1,
    height: 80,
    normalize: true,
  });

  wavesurfer.setVolume(parseFloat(document.getElementById('volume-slider').value));

  wavesurfer.on('ready', function () {
    var dur = wavesurfer.getDuration();
    document.getElementById('waveform-time').textContent = '0.00 / ' + dur.toFixed(2) + ' s';
    // Show file info
    showFileInfo(fileId, f);
    // Re-render markers after waveform geometry is settled
    renderMarkers();
    renderMarkerOverlays();
  });
  wavesurfer.on('timeupdate', function (t) {
    var dur = wavesurfer.getDuration();
    document.getElementById('time-display').textContent = t.toFixed(2) + ' / ' + dur.toFixed(2) + ' s';
    document.getElementById('waveform-time').textContent = t.toFixed(2) + ' / ' + dur.toFixed(2) + ' s';
  });
  wavesurfer.on('play', function () { document.getElementById('btn-play-pause').textContent = '⏸'; });
  wavesurfer.on('pause', function () { document.getElementById('btn-play-pause').textContent = '▶'; });
  wavesurfer.on('finish', function () {
    document.getElementById('btn-play-pause').textContent = '▶';
    if (loopEnabled) wavesurfer.play();
  });
  wavesurfer.on('error', function (e) {
    document.getElementById('waveform-time').textContent = 'Error loading audio';
    console.error('WaveSurfer error', e);
  });

  wavesurfer.load('/audio/upload/' + encodeURIComponent(fileId) + '?v=' + Date.now());
}

function showFileInfo(fileId, f) {
  var bar = document.getElementById('file-info-bar');
  var txt = document.getElementById('file-info-text');
  var parts = [];
  if (f && f.samplerate) parts.push(f.samplerate + ' Hz');
  if (f && f.channels)   parts.push(f.channels === 1 ? 'Mono' : f.channels === 2 ? 'Stereo' : f.channels + 'ch');
  if (f && f.subtype)    parts.push(f.subtype);
  if (f && f.duration)   parts.push(f.duration.toFixed(2) + ' s');
  txt.textContent = parts.join('  ·  ');
  bar.classList.remove('hidden');
}

// ─── SR fix ───────────────────────────────────────────────────────────────────
// ─── File operations ──────────────────────────────────────────────────────────
function deleteFile(fileId) {
  if (!confirm('Delete this file and its segments?')) return;
  api('/api/file/' + encodeURIComponent(fileId), 'DELETE').then(function () {
    if (currentFileId === fileId) {
      currentFileId = null;
      if (wavesurfer) { wavesurfer.destroy(); wavesurfer = null; }
      document.getElementById('waveform-controls').classList.add('hidden');
      document.getElementById('file-info-bar').classList.add('hidden');
      document.getElementById('waveform-filename').textContent = '';
      clearSelection();
    }
    refreshState();
  }).catch(function (e) { alert('Delete error: ' + e.message); });
}

function addWholeFileAsSegment() {
  if (!currentFileId) return;
  withSpinner(document.getElementById('btn-add-whole'), function () {
    return api('/api/file/' + encodeURIComponent(currentFileId) + '/as_segment', 'POST').then(function () { refreshState(); });
  });
}

// ─── Auto-markers ─────────────────────────────────────────────────────────────
function detectAutoMarkers() {
  if (!currentFileId) {
    setStatus('Please select a file first', true);
    return Promise.resolve();
  }
  var params = {
    min_silence_ms: parseInt(document.getElementById('min-silence').value, 10),
    silence_thresh_db: parseFloat(document.getElementById('silence-thresh').value),
  };
  return api('/api/file/' + encodeURIComponent(currentFileId) + '/auto_markers', 'POST', params)
    .then(function (res) {
      var added = 0;
      (res.markers || []).forEach(function (t) {
        if (splitMarkers.indexOf(t) === -1) { splitMarkers.push(t); added++; }
      });
      splitMarkers.sort(function (a, b) { return a - b; });
      renderMarkers();
      renderMarkerOverlays();
      persistMarkers();
      if (res.markers.length === 0) {
        setStatus('No silence gaps detected. Try a lower threshold (e.g. -50 dB).', true);
      } else {
        setStatus('Added ' + added + ' marker(s). Total: ' + splitMarkers.length, false);
      }
    });
}

// ─── Marker management ────────────────────────────────────────────────────────
function addMarkerAtCurrent() {
  if (!wavesurfer) return;
  var t = parseFloat(wavesurfer.getCurrentTime().toFixed(3));
  if (splitMarkers.indexOf(t) === -1) {
    splitMarkers.push(t);
    splitMarkers.sort(function (a, b) { return a - b; });
    renderMarkers();
    renderMarkerOverlays();
    persistMarkers();
  }
}

function clearMarkers(options) {
  splitMarkers = [];
  renderMarkers();
  renderMarkerOverlays();
  if (!(options && options.persist === false)) persistMarkers();
}

function normaliseMarkers(values) {
  var seen = {};
  var out = [];
  (values || []).forEach(function (value) {
    var t = parseFloat(value);
    if (!isFinite(t) || t <= 0) return;
    t = Math.round(t * 1000) / 1000;
    var key = String(t);
    if (!seen[key]) {
      seen[key] = true;
      out.push(t);
    }
  });
  out.sort(function (a, b) { return a - b; });
  return out;
}

function restoreMarkersForFile(fileId) {
  state.markers = state.markers || {};
  splitMarkers = normaliseMarkers(state.markers[fileId] || []);
  renderMarkers();
  renderMarkerOverlays();
}

function persistMarkers() {
  if (!currentFileId) return Promise.resolve();
  state.markers = state.markers || {};
  splitMarkers = normaliseMarkers(splitMarkers);
  if (splitMarkers.length) state.markers[currentFileId] = splitMarkers.slice();
  else delete state.markers[currentFileId];
  return api('/api/file/' + encodeURIComponent(currentFileId) + '/markers', 'PUT', { markers: splitMarkers })
    .then(function (res) {
      if (res.markers && res.markers.length) state.markers[currentFileId] = res.markers;
      else delete state.markers[currentFileId];
      return res;
    }).catch(function (e) {
      setStatus('Marker save error: ' + e.message, true);
    });
}

// Chip list below controls
function renderMarkers() {
  var el = document.getElementById('markers-list');
  el.innerHTML = '';
  if (wavesurfer) {
    var startChip = document.createElement('div');
    startChip.className = 'marker-chip virtual-marker';
    startChip.title = 'Click to seek to the beginning';
    startChip.textContent = '0.00s start';
    startChip.addEventListener('click', function () {
      if (wavesurfer) wavesurfer.setTime(0);
    });
    el.appendChild(startChip);
  }
  if (!splitMarkers.length) return;
  splitMarkers.forEach(function (t, i) {
    var chip = document.createElement('div');
    chip.className = 'marker-chip';
    chip.title = 'Click to seek / × to delete';
    chip.innerHTML = t.toFixed(2) + 's <span class="remove-marker">×</span>';

    // Click chip body → seek to marker position
    chip.addEventListener('click', (function (time) {
      return function (e) {
        if (e.target.classList.contains('remove-marker')) return;
        if (wavesurfer) wavesurfer.setTime(time);
      };
    })(t));

    // Click × → delete marker
    chip.querySelector('.remove-marker').addEventListener('click', (function (idx) {
      return function (e) {
        e.stopPropagation();
        splitMarkers.splice(idx, 1);
        renderMarkers();
        renderMarkerOverlays();
        persistMarkers();
      };
    })(i));

    el.appendChild(chip);
  });
}

// Visual marker lines on the waveform overlay
function renderMarkerOverlays() {
  var container = document.getElementById('wf-markers');
  container.innerHTML = '';
  if (!wavesurfer) return;
  var dur = wavesurfer.getDuration();
  if (!dur) return;
  var startLine = document.createElement('div');
  startLine.className = 'wf-marker-line virtual';
  startLine.style.left = '0%';
  var startLabel = document.createElement('div');
  startLabel.className = 'wf-marker-label';
  startLabel.textContent = '0.00';
  startLine.appendChild(startLabel);
  container.appendChild(startLine);
  splitMarkers.forEach(function (t, i) {
    var pct = (t / dur * 100).toFixed(3);
    var line = document.createElement('div');
    line.className = 'wf-marker-line';
    line.style.left = pct + '%';
    var label = document.createElement('div');
    label.className = 'wf-marker-label';
    label.textContent = t.toFixed(2);
    line.appendChild(label);
    container.appendChild(line);
  });
}

// ─── Apply split ──────────────────────────────────────────────────────────────
function applyManualSplit() {
  if (!currentFileId || !splitMarkers.length) {
    alert('Add markers first');
    return Promise.resolve();
  }
  var segs = state.segments.filter(function (s) { return s.file_id === currentFileId; });
  if (!segs.length) {
    return api('/api/file/' + encodeURIComponent(currentFileId) + '/as_segment', 'POST').then(function () {
      return refreshStatePromise().then(function () {
        var s = state.segments.find(function (s) { return s.file_id === currentFileId; });
        if (!s) return;
        return doSplit(s.id);
      });
    });
  }
  return doSplit(segs[0].id);
}

function doSplit(segId) {
  return api('/api/segment/' + encodeURIComponent(segId) + '/split', 'POST', { positions: splitMarkers }).then(function () {
    clearMarkers();
    refreshState();
    document.querySelector('.tab[data-tab="segments"]').click();
  }).catch(function (e) { alert('Split error: ' + e.message); });
}

function refreshStatePromise() {
  return api('/api/state').then(function (data) { state = data; state.markers = state.markers || {}; renderFileList(); renderSegmentList(); });
}

// ─── Segment list ─────────────────────────────────────────────────────────────
function renderSegmentList() {
  var el = document.getElementById('segment-list');
  document.getElementById('seg-count').textContent = state.segments.length;
  el.innerHTML = '';
  state.segments.forEach(function (seg) {
    var file = state.files[seg.file_id];
    var card = document.createElement('div');
    card.className = 'seg-card' + (seg.transcribed ? '' : ' untranscribed');
    card.dataset.id = seg.id;
    card.innerHTML =
      '<div class="seg-header">' +
        '<span class="seg-info">' + esc(file ? file.name : '?') + ' ' +
          seg.start.toFixed(2) + '–' + seg.end.toFixed(2) + 's (' + fmtDur(seg.duration) + ')</span>' +
        '<div class="seg-actions">' +
          '<button class="icon-btn seg-play" title="Play">▶</button>' +
          '<button class="icon-btn seg-transcribe" title="Transcribe">🎙</button>' +
          '<button class="icon-btn seg-delete" title="Delete">✕</button>' +
        '</div>' +
      '</div>' +
      '<textarea class="seg-text" rows="2" placeholder="Text (or auto-transcribe)">' + esc(seg.text) + '</textarea>';

    var textarea = card.querySelector('.seg-text');
    var saveTimeout = null;
    textarea.addEventListener('input', function () {
      textarea.classList.add('unsaved');
      clearTimeout(saveTimeout);
      saveTimeout = setTimeout(function () { saveSegText(seg.id, textarea); }, 800);
    });
    textarea.addEventListener('blur', function () { clearTimeout(saveTimeout); saveSegText(seg.id, textarea); });

    card.querySelector('.seg-play').addEventListener('click', function () { playSegment(seg.id); });
    card.querySelector('.seg-transcribe').addEventListener('click', (function (sid, c) {
      return function () { transcribeOne(sid, c); };
    })(seg.id, card));
    card.querySelector('.seg-delete').addEventListener('click', (function (sid) {
      return function () { deleteSegment(sid); };
    })(seg.id));
    el.appendChild(card);
  });
}

function saveSegText(segId, textarea) {
  api('/api/segment/' + encodeURIComponent(segId) + '/text', 'PUT', { text: textarea.value })
    .then(function () {
      textarea.classList.remove('unsaved');
      var seg = state.segments.find(function (s) { return s.id === segId; });
      if (seg) { seg.text = textarea.value; seg.transcribed = !!textarea.value.trim(); }
      markDatasetDirty();
    }).catch(function () {});
}

function playSegment(segId) {
  var audio = document.getElementById('mini-audio');
  document.getElementById('mini-player').classList.remove('hidden');
  audio.src = '/audio/segment/' + encodeURIComponent(segId);
  audio.play();
}

function transcribeOne(segId, card) {
  var btn = card.querySelector('.seg-transcribe');
  withSpinner(btn, function () {
    return api('/api/segment/' + encodeURIComponent(segId) + '/transcribe', 'POST').then(function (res) {
      var seg = state.segments.find(function (s) { return s.id === segId; });
      if (seg) { seg.text = res.text; seg.transcribed = true; }
      card.querySelector('.seg-text').value = res.text;
      card.classList.remove('untranscribed');
    }).catch(function (e) { alert('Transcription error: ' + e.message); });
  });
}

function deleteSegment(segId) {
  if (!confirm('Delete this segment?')) return;
  api('/api/segment/' + encodeURIComponent(segId), 'DELETE').then(function () {
    state.segments = state.segments.filter(function (s) { return s.id !== segId; });
    renderSegmentList(); renderFileList();
    markDatasetDirty();
  }).catch(function (e) { alert('Delete error: ' + e.message); });
}

// ─── Transcribe all ───────────────────────────────────────────────────────────
function transcribeAll() {
  var btn = document.getElementById('btn-transcribe-all');
  var prog = document.getElementById('transcribe-progress');
  var failed = 0;
  var origHTML = btn.innerHTML;

  btn.disabled = true;
  prog.textContent = '';

  function setBtnProgress(text) {
    btn.innerHTML = '<span class="spinner"></span> ' + text;
  }

  setBtnProgress('...');

  fetchSSE('/api/transcribe_all', 'POST', null, function (event) {
    if (event.total !== undefined) setBtnProgress('0 / ' + event.total + ' 完了');
    if (event.progress !== undefined) {
      if (event.error) {
        failed += 1;
        setStatus('Transcription error: ' + event.error, true);
      } else {
        var seg = state.segments.find(function (s) { return s.id === event.id; });
        if (seg && event.text !== undefined) {
          seg.text = event.text; seg.transcribed = true;
          var card = document.querySelector('.seg-card[data-id="' + event.id + '"]');
          if (card) { card.querySelector('.seg-text').value = event.text; card.classList.remove('untranscribed'); }
        }
      }
      setBtnProgress(event.progress + ' / ' + event.total + (failed ? ' (' + failed + ' failed)' : ' 完了'));
    }
    if (event.done) prog.textContent = failed ? 'Done (' + failed + ' failed)' : 'Done';
  }).catch(function (e) {
    prog.textContent = 'Error: ' + e.message;
  }).finally(function () {
    btn.disabled = false;
    btn.innerHTML = origHTML;
  });
}

// ─── Dataset dirty tracking ───────────────────────────────────────────────────
var datasetDirty = false;

function markDatasetDirty() {
  datasetDirty = true;
  document.getElementById('btn-build-dataset').classList.add('dirty');
  document.getElementById('dataset-dirty-msg').classList.remove('hidden');
}

function clearDatasetDirty() {
  datasetDirty = false;
  document.getElementById('btn-build-dataset').classList.remove('dirty');
  document.getElementById('dataset-dirty-msg').classList.add('hidden');
}

function resetDatasetBuildForm() {
  document.getElementById('speaker-name').value = '';
  document.getElementById('job-name').value = '';
  document.getElementById('build-result').textContent = '';
  var paths = document.getElementById('build-paths');
  paths.innerHTML = '';
  paths.classList.add('hidden');
}

// ─── Training ─────────────────────────────────────────────────────────────────
function getDatasetNames() {
  var speakerName = document.getElementById('speaker-name').value.trim();
  var jobName = document.getElementById('job-name').value.trim() || speakerName;
  return {
    jobName: jobName,
    speakerName: speakerName || jobName,
  };
}

function buildDataset(overwrite) {
  if (!ensureIrodoriRoot()) return Promise.resolve();
  var result = document.getElementById('build-result');
  var paths = document.getElementById('build-paths');
  result.textContent = '';
  paths.classList.add('hidden');
  var names = getDatasetNames();
  if (!names.jobName) {
    result.textContent = 'キャラクター名または出力先データセット名を入力してください';
    result.style.color = 'var(--danger)';
    return Promise.resolve();
  }
  document.getElementById('job-name').value = names.jobName;
  var payload = {
    job_name: names.jobName,
    speaker_name: names.speakerName,
    overwrite: !!overwrite,
    device: document.getElementById('train-device').value,
    precision: document.getElementById('precision').value,
    max_steps: parseInt(document.getElementById('max-steps').value, 10),
    batch_size: parseInt(document.getElementById('batch-size').value, 10),
    grad_accum: parseInt(document.getElementById('grad-accum').value, 10),
    tokens: parseInt(document.getElementById('tokens').value, 10),
    learning_rate: parseFloat(document.getElementById('lr').value),
    save_every: parseInt(document.getElementById('save-every').value, 10),
    normalize_db: document.getElementById('normalize-db').value,
    init_embedding: document.getElementById('init-embedding').value.trim(),
    irodori_root: settings.irodori_root,
    uv_exe: settings.uv_exe,
  };
  return api('/api/build_dataset', 'POST', payload).then(function (res) {
    result.textContent = res.count + ' samples';
    result.style.color = 'var(--success)';
    paths.innerHTML =
      '📁 ' + res.dataset_dir + '<br>' +
      '📄 source.jsonl  ·  train.bat  ·  train.sh';
    paths.classList.remove('hidden');
    clearDatasetDirty();
    refreshDatasets();
    refreshDataDatasetProjects();
  }).catch(function (e) {
    // 409 = dataset already exists → ask to overwrite
    if (e.message.indexOf('409') === 0) {
      if (confirm('Dataset "' + payload.job_name + '" already exists.\nOverwrite WAVs and scripts?')) {
        return buildDataset(true);
      }
      return;
    }
    result.textContent = 'Error: ' + e.message;
    result.style.color = 'var(--danger)';
    throw e;
  });
}

// ─── Dataset list & load ──────────────────────────────────────────────────────
function refreshDatasets() {
  api('/api/datasets').then(function (data) {
    var el = document.getElementById('dataset-list');
    if (!el) return;
    if (!data.datasets.length) {
      el.innerHTML = '<span class="dim" style="font-size:12px">データセットなし</span>';
      return;
    }
    el.innerHTML = '';
    data.datasets.forEach(function (ds) {
      var row = document.createElement('div');
      row.className = 'dataset-row';
      row.innerHTML =
        '<span class="ds-name">' + esc(ds.name) + '</span>' +
        '<span class="ds-badge ' + (ds.has_source ? 'ok' : 'dim') + '">source</span>' +
        '<span class="ds-badge ' + (ds.has_manifest ? 'ok' : 'dim') + '">manifest</span>' +
        '<span class="ds-badge ' + (ds.has_embedding ? 'ok' : 'dim') + '">embedding</span>' +
        '<span class="dim" style="font-size:11px">' + ds.segment_count + '件</span>' +
        '<button class="load-ds-btn" data-name="' + esc(ds.name) + '">Load</button>';
      row.querySelector('.load-ds-btn').addEventListener('click', (function (name) {
        return function () { loadDatasetProject(name); };
      })(ds.name));
      el.appendChild(row);
    });
  }).catch(function () {});
}

function refreshDataDatasetProjects() {
  var el = document.getElementById('dataset-project-list');
  if (!el) return;
  el.innerHTML = '<span class="dim" style="font-size:12px">Select a dataset folder to restore data preparation.</span>';
}

function loadDatasetProject(name) {
  var btn = document.querySelector('.load-ds-btn[data-name="' + esc(name) + '"]');
  var doLoad = function () {
    return api('/api/datasets/' + encodeURIComponent(name) + '/load', 'POST').then(function (res) {
      document.getElementById('job-name').value = name;
      return refreshStatePromise().then(function () {
        if (res.first_file_id) selectFile(res.first_file_id);
      });
    }).then(function () {
      clearDatasetDirty();
      refreshDataDatasetProjects();
      setStatus('Loaded project: ' + name, false);
      document.querySelector('.tab[data-tab="data"]').click();
    });
  };
  if (btn) {
    withSpinner(btn, doLoad);
  } else {
    doLoad().catch(function (e) { setStatus('Load error: ' + e.message, true); });
  }
}

function prepareManifest() {
  if (!ensureIrodoriRoot()) return Promise.resolve();
  var names = getDatasetNames();
  if (!names.jobName) { setStatus('キャラクター名または出力先データセット名を入力してください', true); return Promise.resolve(); }
  document.getElementById('job-name').value = names.jobName;
  var log = document.getElementById('train-log');
  var result = document.getElementById('manifest-result');
  log.textContent = '';
  result.textContent = '';
  result.style.color = 'var(--dim)';
  appendLog(log, 'Starting manifest preparation...\n', '');
  return fetchSSE('/api/prepare_manifest', 'POST', {
    job_name: names.jobName,
    device: document.getElementById('prep-device').value,
    normalize_db: document.getElementById('normalize-db').value,
    max_seconds: parseFloat(document.getElementById('max-seconds').value || '0'),
    irodori_root: settings.irodori_root,
    uv_exe: settings.uv_exe,
  }, function (ev) {
    handleLogEvent(ev, log);
    if (ev.done) {
      if (ev.rc === 0) {
        result.textContent = 'Manifest 準備完了';
        result.style.color = 'var(--success)';
      } else {
        result.textContent = 'Manifest 準備失敗';
        result.style.color = 'var(--danger)';
      }
    }
  }).catch(function (e) {
    result.textContent = 'Manifest 準備失敗';
    result.style.color = 'var(--danger)';
    appendLog(log, 'Error: ' + e.message + '\n', 'log-error');
  });
}

function startTraining() {
  if (!ensureIrodoriRoot()) return;
  var names = getDatasetNames();
  if (!names.jobName) { setStatus('キャラクター名または出力先データセット名を入力してください', true); return; }
  document.getElementById('job-name').value = names.jobName;
  var log = document.getElementById('train-log');
  log.textContent = '';
  appendLog(log, 'Starting training...\n', '');
  var btn = document.getElementById('btn-train');
  var stop = document.getElementById('btn-stop-train');
  btn.classList.add('hidden');
  stop.classList.remove('hidden');
  fetchSSE('/api/train', 'POST', {
    job_name: names.jobName,
    device: document.getElementById('train-device').value,
    precision: document.getElementById('precision').value,
    max_steps: parseInt(document.getElementById('max-steps').value, 10),
    batch_size: parseInt(document.getElementById('batch-size').value, 10),
    grad_accum: parseInt(document.getElementById('grad-accum').value, 10),
    tokens: parseInt(document.getElementById('tokens').value, 10),
    learning_rate: parseFloat(document.getElementById('lr').value),
    save_every: parseInt(document.getElementById('save-every').value, 10),
    init_embedding: document.getElementById('init-embedding').value.trim(),
    checkpoint_path: settings.checkpoint_path,
    irodori_root: settings.irodori_root,
    uv_exe: settings.uv_exe,
  }, function (ev) { handleLogEvent(ev, log); }).then(function () {
    refreshRuns();
  }).catch(function (e) {
    appendLog(log, 'Error: ' + e.message + '\n', 'log-error');
  }).finally(function () {
    btn.classList.remove('hidden');
    stop.classList.add('hidden');
  });
}

// ─── Test ─────────────────────────────────────────────────────────────────────
function refreshRuns() {
  api('/api/runs').then(function (data) {
    runsData = data.runs || [];
    var sel = document.getElementById('embed-select');
    var cur = sel.value;
    sel.innerHTML = '<option value="">-- Select --</option>';
    runsData.forEach(function (run) {
      if (!run.has_embedding) return;
      var opt = document.createElement('option');
      opt.value = run.embedding_path; opt.textContent = run.name;
      sel.appendChild(opt);
    });
    if (cur) sel.value = cur;
    updateStepEmbedList();
  }).catch(function () {});
}

function updateStepEmbedList() {
  var sel = document.getElementById('embed-select');
  var stepSel = document.getElementById('step-embed-select');
  var run = runsData.find(function (r) { return r.embedding_path === sel.value; });
  stepSel.innerHTML = '';
  if (!run) {
    var opt = document.createElement('option');
    opt.value = ''; opt.textContent = '-- run を選択すると表示されます --';
    opt.disabled = true;
    stepSel.appendChild(opt);
    return;
  }
  var finalOpt = document.createElement('option');
  finalOpt.value = run.embedding_path;
  finalOpt.textContent = 'checkpoint_final.speaker.safetensors';
  finalOpt.selected = true;
  stepSel.appendChild(finalOpt);
  (run.step_embeddings || []).forEach(function (e) {
    var opt = document.createElement('option');
    opt.value = e.path; opt.textContent = e.name;
    stepSel.appendChild(opt);
  });
}

function generateAudio() {
  if (!ensureIrodoriRoot()) return Promise.resolve();
  var log = document.getElementById('gen-log');
  var outputSection = document.getElementById('output-section');
  log.textContent = ''; outputSection.classList.add('hidden');
  var embPath = document.getElementById('embed-path-custom').value.trim()
    || document.getElementById('step-embed-select').value
    || document.getElementById('embed-select').value;
  if (!embPath) { alert('Select a speaker embedding'); return Promise.resolve(); }
  var text = document.getElementById('test-text').value.trim();
  if (!text) { alert('Enter text to synthesize'); return Promise.resolve(); }
  appendLog(log, 'Starting generation...\n', '');
  return fetchSSE('/api/generate', 'POST', {
    text: text, embedding_path: embPath, checkpoint_path: settings.checkpoint_path,
    output_name: 'output_' + Date.now(),
    num_steps: parseInt(document.getElementById('gen-steps').value, 10),
    seed: parseInt(document.getElementById('gen-seed').value, 10),
    irodori_root: settings.irodori_root, uv_exe: settings.uv_exe,
  }, function (ev) {
    handleLogEvent(ev, log);
    if (ev.audio_url) { var a = document.getElementById('output-audio'); a.src = ev.audio_url; outputSection.classList.remove('hidden'); a.play(); }
  }).catch(function (e) { appendLog(log, 'Error: ' + e.message + '\n', 'log-error'); });
}

// ─── Log ──────────────────────────────────────────────────────────────────────
function handleLogEvent(event, logEl) {
  if (event.log !== undefined) appendLog(logEl, event.log + '\n', '');
  if (event.done) appendLog(logEl, '\n[Done rc=' + event.rc + ']\n', event.rc === 0 ? 'log-done' : 'log-error');
}
function appendLog(el, text, cls) {
  var span = document.createElement('span');
  span.textContent = text;
  if (cls) span.className = cls;
  el.appendChild(span);
  el.scrollTop = el.scrollHeight;
}

// ─── SSE fetch ────────────────────────────────────────────────────────────────
function fetchSSE(url, method, body, onEvent) {
  var opts = { method: method || 'GET', headers: {} };
  if (body && !(body instanceof FormData)) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  else if (body instanceof FormData) { opts.body = body; }
  return fetch(url, opts).then(function (res) {
    if (!res.ok) return res.text().then(function (t) { throw new Error(res.status + ': ' + t); });
    var reader = res.body.getReader(), decoder = new TextDecoder(), buf = '';
    function pump() {
      return reader.read().then(function (r) {
        if (r.done) return;
        buf += decoder.decode(r.value, { stream: true });
        var lines = buf.split('\n'); buf = lines.pop();
        lines.forEach(function (line) {
          if (line.indexOf('data: ') === 0) { try { onEvent(JSON.parse(line.slice(6))); } catch (e) {} }
        });
        return pump();
      });
    }
    return pump();
  });
}

// ─── API ──────────────────────────────────────────────────────────────────────
function api(url, method, body) {
  var opts = { method: method || 'GET', headers: {} };
  if (body && !(body instanceof FormData)) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  else if (body instanceof FormData) { opts.body = body; }
  return fetch(url, opts).then(function (res) {
    if (!res.ok) return res.text().then(function (t) { throw new Error(res.status + ': ' + t); });
    return res.json();
  });
}

// ─── Utils ────────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function fmtDur(sec) {
  if (!sec || sec < 0) return '0s';
  if (sec < 60) return sec.toFixed(1) + 's';
  return Math.floor(sec / 60) + 'm' + (sec % 60).toFixed(0) + 's';
}
