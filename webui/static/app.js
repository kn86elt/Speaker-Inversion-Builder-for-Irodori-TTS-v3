// Speaker Inversion Builder – frontend
// WaveSurfer is loaded as a global via wavesurfer.min.js

// ─── State ───────────────────────────────────────────────────────────────────
var state = { files: {}, segments: [] };
var currentFileId = null;
var wavesurfer = null;
var splitMarkers = [];
var loopEnabled = false;
var settings = {
  irodori_root: 'C:/usr/sd/Irodori-TTS-v3',
  python_exe: '',
  checkpoint_path: '',
};

// ─── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
  bindUI();          // bind all handlers first – never block UI
  loadSettings();
  refreshState();
  refreshRuns();
});

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

  // Drop zone
  var dropArea = document.getElementById('drop-area');
  var fileInput = document.getElementById('file-input');
  dropArea.addEventListener('click', function () { fileInput.click(); });
  dropArea.addEventListener('dragover', function (e) { e.preventDefault(); dropArea.classList.add('dragover'); });
  dropArea.addEventListener('dragleave', function () { dropArea.classList.remove('dragover'); });
  dropArea.addEventListener('drop', function (e) {
    e.preventDefault();
    dropArea.classList.remove('dragover');
    handleFileUpload(e.dataTransfer.files);
  });
  fileInput.addEventListener('change', function (e) { handleFileUpload(e.target.files); });

  // Waveform controls
  document.getElementById('btn-play-pause').addEventListener('click', function () { if (wavesurfer) wavesurfer.playPause(); });
  document.getElementById('btn-loop').addEventListener('click', function () {
    loopEnabled = !loopEnabled;
    document.getElementById('btn-loop').style.color = loopEnabled ? 'var(--accent)' : '';
  });
  document.getElementById('btn-add-whole').addEventListener('click', addWholeFileAsSegment);
  document.getElementById('btn-split-silence').addEventListener('click', splitBySilence);
  document.getElementById('btn-add-marker').addEventListener('click', addMarkerAtCurrent);
  document.getElementById('btn-clear-markers').addEventListener('click', clearMarkers);
  document.getElementById('btn-apply-split').addEventListener('click', applyManualSplit);

  // Segments tab
  document.getElementById('btn-transcribe-all').addEventListener('click', transcribeAll);

  // Training tab
  document.getElementById('btn-build-dataset').addEventListener('click', buildDataset);
  document.getElementById('btn-prepare-manifest').addEventListener('click', prepareManifest);
  document.getElementById('btn-train').addEventListener('click', startTraining);

  // Test tab
  document.getElementById('btn-refresh-runs').addEventListener('click', refreshRuns);
  document.getElementById('btn-generate').addEventListener('click', generateAudio);

  // Settings
  document.getElementById('btn-save-settings').addEventListener('click', saveSettings);
}

// ─── Settings ─────────────────────────────────────────────────────────────────
function loadSettings() {
  api('/api/settings').then(function (data) {
    settings.irodori_root = data.irodori_root || settings.irodori_root;
    settings.python_exe = '';
    settings.checkpoint_path = data.default_checkpoint || '';
    document.getElementById('set-irodori-root').value = settings.irodori_root;
    document.getElementById('set-python-exe').value = '';
    document.getElementById('set-checkpoint').value = settings.checkpoint_path;
  }).catch(function () {});
}

function saveSettings() {
  settings.irodori_root = document.getElementById('set-irodori-root').value.trim();
  settings.python_exe = document.getElementById('set-python-exe').value.trim();
  settings.checkpoint_path = document.getElementById('set-checkpoint').value.trim();
  var st = document.getElementById('settings-status');
  st.textContent = 'Saved';
  setTimeout(function () { st.textContent = ''; }, 2000);
}

// ─── State refresh ─────────────────────────────────────────────────────────────
function refreshState() {
  api('/api/state').then(function (data) {
    state = data;
    renderFileList();
    renderSegmentList();
  }).catch(function (e) { console.error('refreshState error', e); });
}

// ─── Upload ───────────────────────────────────────────────────────────────────
function handleFileUpload(fileList) {
  if (!fileList || !fileList.length) return;
  var form = new FormData();
  for (var i = 0; i < fileList.length; i++) form.append('files', fileList[i]);
  api('/api/upload', 'POST', form).then(function () {
    refreshState();
  }).catch(function (e) {
    alert('Upload error: ' + e.message);
  });
}

// ─── File list rendering ──────────────────────────────────────────────────────
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
      e.stopPropagation();
      deleteFile(f.id);
    });
    el.appendChild(div);
  });
}

// ─── Select file → load waveform ──────────────────────────────────────────────
function selectFile(fileId) {
  currentFileId = fileId;
  clearMarkers();
  renderFileList();

  var f = state.files[fileId];
  document.getElementById('waveform-controls').classList.remove('hidden');
  document.getElementById('waveform-filename').textContent = f ? f.name : '';
  document.getElementById('waveform-time').textContent = 'Loading...';

  if (wavesurfer) { wavesurfer.destroy(); wavesurfer = null; }

  wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: '#4a9eff',
    progressColor: '#1a6fc4',
    cursorColor: '#ffffff',
    barWidth: 2,
    barGap: 1,
    height: 80,
    normalize: true,
  });

  wavesurfer.on('timeupdate', function (t) {
    var dur = wavesurfer.getDuration();
    var txt = t.toFixed(2) + ' / ' + dur.toFixed(2) + ' s';
    document.getElementById('time-display').textContent = txt;
    document.getElementById('waveform-time').textContent = txt;
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

  wavesurfer.load('/audio/upload/' + fileId);
}

// ─── File operations ──────────────────────────────────────────────────────────
function deleteFile(fileId) {
  if (!confirm('Delete this file and its segments?')) return;
  api('/api/file/' + fileId, 'DELETE').then(function () {
    if (currentFileId === fileId) {
      currentFileId = null;
      if (wavesurfer) { wavesurfer.destroy(); wavesurfer = null; }
      document.getElementById('waveform-controls').classList.add('hidden');
      document.getElementById('waveform-filename').textContent = '';
    }
    refreshState();
  }).catch(function (e) { alert('Delete error: ' + e.message); });
}

function addWholeFileAsSegment() {
  if (!currentFileId) return;
  api('/api/file/' + currentFileId + '/as_segment', 'POST').then(function () {
    refreshState();
  }).catch(function (e) { alert('Error: ' + e.message); });
}

// ─── Silence split ────────────────────────────────────────────────────────────
function splitBySilence() {
  if (!currentFileId) return;
  var btn = document.getElementById('btn-split-silence');
  btn.disabled = true;
  btn.textContent = 'Processing...';
  var params = {
    min_silence_ms: parseInt(document.getElementById('min-silence').value, 10),
    silence_thresh_db: parseFloat(document.getElementById('silence-thresh').value),
    keep_silence_ms: parseInt(document.getElementById('keep-silence').value, 10),
  };
  api('/api/file/' + currentFileId + '/split_silence', 'POST', params).then(function (result) {
    refreshState();
    alert(result.segments.length + ' segments created');
  }).catch(function (e) {
    alert('Split error: ' + e.message);
  }).finally(function () {
    btn.disabled = false;
    btn.textContent = 'Auto-split';
  });
}

// ─── Manual split markers ─────────────────────────────────────────────────────
function addMarkerAtCurrent() {
  if (!wavesurfer) return;
  var t = parseFloat(wavesurfer.getCurrentTime().toFixed(3));
  if (splitMarkers.indexOf(t) === -1) {
    splitMarkers.push(t);
    splitMarkers.sort(function (a, b) { return a - b; });
    renderMarkers();
  }
}

function clearMarkers() {
  splitMarkers = [];
  renderMarkers();
}

function renderMarkers() {
  var el = document.getElementById('markers-list');
  el.innerHTML = '';
  splitMarkers.forEach(function (t, i) {
    var chip = document.createElement('div');
    chip.className = 'marker-chip';
    chip.innerHTML = t.toFixed(2) + 's <span class="remove-marker" data-i="' + i + '">×</span>';
    chip.querySelector('.remove-marker').addEventListener('click', (function (idx) {
      return function () { splitMarkers.splice(idx, 1); renderMarkers(); };
    })(i));
    el.appendChild(chip);
  });
}

function applyManualSplit() {
  if (!currentFileId || !splitMarkers.length) {
    alert('Add markers first');
    return;
  }
  var segs = state.segments.filter(function (s) { return s.file_id === currentFileId; });
  if (!segs.length) {
    // Create whole-file segment first, then split
    api('/api/file/' + currentFileId + '/as_segment', 'POST').then(function (res) {
      refreshState();
      return doSplit(res.segment.id);
    }).catch(function (e) { alert('Error: ' + e.message); });
    return;
  }
  doSplit(segs[0].id);
}

function doSplit(segId) {
  api('/api/segment/' + segId + '/split', 'POST', { positions: splitMarkers }).then(function () {
    clearMarkers();
    refreshState();
  }).catch(function (e) { alert('Split error: ' + e.message); });
}

// ─── Segment list rendering ───────────────────────────────────────────────────
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
    textarea.addEventListener('blur', function () {
      clearTimeout(saveTimeout);
      saveSegText(seg.id, textarea);
    });

    card.querySelector('.seg-play').addEventListener('click', function () { playSegment(seg.id); });
    card.querySelector('.seg-transcribe').addEventListener('click', (function (s, c) {
      return function () { transcribeOne(s, c); };
    })(seg.id, card));
    card.querySelector('.seg-delete').addEventListener('click', (function (s) {
      return function () { deleteSegment(s); };
    })(seg.id));

    el.appendChild(card);
  });
}

function saveSegText(segId, textarea) {
  var text = textarea.value;
  api('/api/segment/' + segId + '/text', 'PUT', { text: text }).then(function () {
    textarea.classList.remove('unsaved');
    var seg = state.segments.find(function (s) { return s.id === segId; });
    if (seg) { seg.text = text; seg.transcribed = !!text.trim(); }
  }).catch(function () {});
}

function playSegment(segId) {
  var audio = document.getElementById('mini-audio');
  var player = document.getElementById('mini-player');
  audio.src = '/audio/segment/' + segId;
  player.classList.remove('hidden');
  audio.play();
}

function transcribeOne(segId, card) {
  var btn = card.querySelector('.seg-transcribe');
  btn.disabled = true;
  btn.textContent = '...';
  api('/api/segment/' + segId + '/transcribe', 'POST').then(function (result) {
    var seg = state.segments.find(function (s) { return s.id === segId; });
    if (seg) { seg.text = result.text; seg.transcribed = true; }
    card.querySelector('.seg-text').value = result.text;
    card.classList.remove('untranscribed');
  }).catch(function (e) {
    alert('Transcription error: ' + e.message);
  }).finally(function () {
    btn.disabled = false;
    btn.textContent = '🎙';
  });
}

function deleteSegment(segId) {
  if (!confirm('Delete this segment?')) return;
  api('/api/segment/' + segId, 'DELETE').then(function () {
    state.segments = state.segments.filter(function (s) { return s.id !== segId; });
    renderSegmentList();
    renderFileList();
  }).catch(function (e) { alert('Delete error: ' + e.message); });
}

// ─── Transcribe all (SSE) ─────────────────────────────────────────────────────
function transcribeAll() {
  var btn = document.getElementById('btn-transcribe-all');
  var prog = document.getElementById('transcribe-progress');
  btn.disabled = true;
  prog.textContent = 'Starting...';

  fetchSSE('/api/transcribe_all', 'POST', null, function (event) {
    if (event.total !== undefined) prog.textContent = '0 / ' + event.total;
    if (event.progress !== undefined) {
      prog.textContent = event.progress + ' / ' + event.total;
      var seg = state.segments.find(function (s) { return s.id === event.id; });
      if (seg && event.text) {
        seg.text = event.text;
        seg.transcribed = true;
        var card = document.querySelector('.seg-card[data-id="' + event.id + '"]');
        if (card) {
          card.querySelector('.seg-text').value = event.text;
          card.classList.remove('untranscribed');
        }
      }
    }
    if (event.done) { prog.textContent = 'Done'; btn.disabled = false; }
  }).catch(function (e) {
    prog.textContent = 'Error: ' + e.message;
    btn.disabled = false;
  });
}

// ─── Training tab ─────────────────────────────────────────────────────────────
function buildDataset() {
  var btn = document.getElementById('btn-build-dataset');
  var result = document.getElementById('build-result');
  btn.disabled = true;
  result.textContent = '';
  api('/api/build_dataset', 'POST', {
    job_name: document.getElementById('job-name').value.trim(),
    speaker_name: document.getElementById('speaker-name').value.trim(),
  }).then(function (res) {
    result.textContent = res.count + ' samples saved';
    result.style.color = 'var(--success)';
  }).catch(function (e) {
    result.textContent = 'Error: ' + e.message;
    result.style.color = 'var(--danger)';
  }).finally(function () { btn.disabled = false; });
}

function prepareManifest() {
  var log = document.getElementById('train-log');
  log.textContent = '';
  appendLog(log, 'Starting manifest preparation...\n', '');
  fetchSSE('/api/prepare_manifest', 'POST', {
    job_name: document.getElementById('job-name').value.trim(),
    device: document.getElementById('prep-device').value,
    normalize_db: document.getElementById('normalize-db').value,
    max_seconds: parseFloat(document.getElementById('max-seconds').value || '0'),
    irodori_root: settings.irodori_root,
    python_exe: settings.python_exe,
  }, function (event) { handleLogEvent(event, log); }).catch(function (e) {
    appendLog(log, 'Error: ' + e.message + '\n', 'log-error');
  });
}

function startTraining() {
  var log = document.getElementById('train-log');
  log.textContent = '';
  appendLog(log, 'Starting training...\n', '');
  var btn = document.getElementById('btn-train');
  var stop = document.getElementById('btn-stop-train');
  btn.classList.add('hidden');
  stop.classList.remove('hidden');

  fetchSSE('/api/train', 'POST', {
    job_name: document.getElementById('job-name').value.trim(),
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
    python_exe: settings.python_exe,
  }, function (event) { handleLogEvent(event, log); }).then(function () {
    refreshRuns();
  }).catch(function (e) {
    appendLog(log, 'Error: ' + e.message + '\n', 'log-error');
  }).finally(function () {
    btn.classList.remove('hidden');
    stop.classList.add('hidden');
  });
}

// ─── Test tab ─────────────────────────────────────────────────────────────────
function refreshRuns() {
  api('/api/runs').then(function (data) {
    var sel = document.getElementById('embed-select');
    var cur = sel.value;
    sel.innerHTML = '<option value="">-- Select --</option>';
    data.runs.forEach(function (run) {
      if (!run.has_embedding) return;
      var opt = document.createElement('option');
      opt.value = run.embedding_path;
      opt.textContent = run.name;
      sel.appendChild(opt);
    });
    if (cur) sel.value = cur;
  }).catch(function () {});
}

function generateAudio() {
  var log = document.getElementById('gen-log');
  var outputSection = document.getElementById('output-section');
  log.textContent = '';
  outputSection.classList.add('hidden');

  var embPath = document.getElementById('embed-select').value ||
    document.getElementById('embed-path-custom').value.trim();
  if (!embPath) { alert('Select a speaker embedding'); return; }
  var text = document.getElementById('test-text').value.trim();
  if (!text) { alert('Enter text to synthesize'); return; }

  appendLog(log, 'Starting generation...\n', '');

  fetchSSE('/api/generate', 'POST', {
    text: text,
    embedding_path: embPath,
    checkpoint_path: settings.checkpoint_path,
    output_name: 'output_' + Date.now(),
    num_steps: parseInt(document.getElementById('gen-steps').value, 10),
    seed: parseInt(document.getElementById('gen-seed').value, 10),
    irodori_root: settings.irodori_root,
    python_exe: settings.python_exe,
  }, function (event) {
    handleLogEvent(event, log);
    if (event.audio_url) {
      document.getElementById('output-audio').src = event.audio_url;
      outputSection.classList.remove('hidden');
    }
  }).catch(function (e) {
    appendLog(log, 'Error: ' + e.message + '\n', 'log-error');
  });
}

// ─── Log helpers ──────────────────────────────────────────────────────────────
function handleLogEvent(event, logEl) {
  if (event.log !== undefined) appendLog(logEl, event.log + '\n', '');
  if (event.done) {
    appendLog(logEl, '\n[Done rc=' + event.rc + ']\n', event.rc === 0 ? 'log-done' : 'log-error');
  }
}

function appendLog(el, text, cls) {
  var span = document.createElement('span');
  span.textContent = text;
  if (cls) span.className = cls;
  el.appendChild(span);
  el.scrollTop = el.scrollHeight;
}

// ─── SSE via fetch (supports POST) ───────────────────────────────────────────
function fetchSSE(url, method, body, onEvent) {
  var opts = { method: method || 'GET', headers: {} };
  if (body && !(body instanceof FormData)) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  } else if (body instanceof FormData) {
    opts.body = body;
  }

  return fetch(url, opts).then(function (res) {
    if (!res.ok) {
      return res.text().then(function (txt) {
        throw new Error(res.status + ': ' + txt);
      });
    }
    var reader = res.body.getReader();
    var decoder = new TextDecoder();
    var buf = '';

    function pump() {
      return reader.read().then(function (result) {
        if (result.done) return;
        buf += decoder.decode(result.value, { stream: true });
        var lines = buf.split('\n');
        buf = lines.pop();
        lines.forEach(function (line) {
          if (line.indexOf('data: ') === 0) {
            try { onEvent(JSON.parse(line.slice(6))); } catch (e) { console.warn('SSE parse error', e); }
          }
        });
        return pump();
      });
    }
    return pump();
  });
}

// ─── API helpers ──────────────────────────────────────────────────────────────
function api(url, method, body) {
  var opts = { method: method || 'GET', headers: {} };
  if (body && !(body instanceof FormData)) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  } else if (body instanceof FormData) {
    opts.body = body;
  }
  return fetch(url, opts).then(function (res) {
    if (!res.ok) {
      return res.text().then(function (txt) {
        throw new Error(res.status + ': ' + txt);
      });
    }
    return res.json();
  });
}

// ─── Utilities ────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function fmtDur(sec) {
  if (!sec || sec < 0) return '0s';
  if (sec < 60) return sec.toFixed(1) + 's';
  return Math.floor(sec / 60) + 'm' + (sec % 60).toFixed(0) + 's';
}
