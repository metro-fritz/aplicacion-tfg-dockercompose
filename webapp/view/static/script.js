// ── State ──────────────────────────────────────────────────
let selectedModel = document.querySelector('.model-card.active')?.dataset.model || 'vgg';
let selectedFile = null;

// ── Model selection ────────────────────────────────────────
document.querySelectorAll('.model-card').forEach(card => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    selectedModel = card.dataset.model;
    document.getElementById('model-info').textContent = card.dataset.desc;
  });
});

// ── Drop zone ──────────────────────────────────────────────
const dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const preview     = document.getElementById('drop-preview');
const placeholder = document.getElementById('drop-placeholder');
const btnClassify = document.getElementById('btn-classify');
const errorMsg    = document.getElementById('error-msg');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  hideError();
  if (!file.type.startsWith('image/')) { showError('El archivo no es una imagen válida.'); return; }
  if (file.size > 10 * 1024 * 1024)   { showError('La imagen supera el límite de 10 MB.'); return; }

  selectedFile = file;
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.style.display = 'block';
  placeholder.style.display = 'none';
  btnClassify.disabled = false;
}

// ── Classification ─────────────────────────────────────────
btnClassify.addEventListener('click', async () => {
  if (!selectedFile) return;

  setLoading(true);
  hideError();

  const fd = new FormData();
  fd.append('image', selectedFile);
  fd.append('model_id', selectedModel);

  try {
    const res  = await fetch('/classify', { method: 'POST', body: fd });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || 'Error desconocido.');
    } else {
      renderResults(data);
    }
  } catch (err) {
    showError('Error de red. ¿Está el servidor corriendo?');
  } finally {
    setLoading(false);
  }
});

function setLoading(on) {
  const spinner = document.getElementById('btn-spinner');
  const text    = document.getElementById('btn-text');
  btnClassify.disabled  = on;
  spinner.style.display = on ? 'block' : 'none';
  text.textContent      = on ? 'PROCESANDO…' : 'CLASIFICAR IMAGEN';
}

// ── Render results ─────────────────────────────────────────
function renderResults(data) {
  document.getElementById('empty-state').style.display = 'none';
  const results = document.getElementById('results');
  results.style.display = 'flex';

  document.getElementById('res-model-tag').textContent = data.model_id;

  // Predicciones rankeadas
  const chart = document.getElementById('bar-chart');
  chart.innerHTML = '';
  const maxProb = data.predictions[0].probability;

  data.predictions.forEach((p, i) => {
    const row  = document.createElement('div');
    row.className = 'bar-row';
    row.style.animationDelay = `${i * 0.07}s`;

    const fill = document.createElement('div');
    fill.className = 'bar-fill';

    row.innerHTML = `
      <div class="bar-rank">#${p.rank}</div>
      <div class="bar-label ${i === 0 ? 'top' : ''}">${capitalize(p.label)}</div>
      <div class="bar-track"></div>
      <div class="bar-prob">${p.probability.toFixed(2)}%</div>
    `;
    row.querySelector('.bar-track').appendChild(fill);
    chart.appendChild(row);

    // Animate bar after paint
    requestAnimationFrame(() => {
      setTimeout(() => {
        fill.style.width = `${(p.probability / maxProb) * 100}%`;
      }, 50 + i * 70);
    });
  });

  
}

// ── Helpers ────────────────────────────────────────────────
function capitalize(str) {
  return str.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function showError(msg) {
  errorMsg.textContent  = '⚠ ' + msg;
  errorMsg.style.display = 'block';
}

function hideError() {
  errorMsg.style.display = 'none';
}