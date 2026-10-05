document.addEventListener('DOMContentLoaded', () => {
  const textInput = document.getElementById('text-input');
  const analyzeBtn = document.getElementById('analyze-btn');
  const clearBtn = document.getElementById('clear-btn');
  const resultArea = document.getElementById('result-area');
  const resultPlaceholder = document.getElementById('result-placeholder');
  const resultContent = document.getElementById('result-content');
  
  const verdictBadge = document.getElementById('verdict-badge');
  const confidenceText = document.getElementById('confidence-text');
  const meterFill = document.getElementById('meter-fill');
  const probNormal = document.getElementById('prob-normal');
  const probThreat = document.getElementById('prob-threat');
  const explanationText = document.getElementById('explanation-text');
  const latencyText = document.getElementById('latency-text');
  const rawJsonOutput = document.getElementById('raw-json-output');
  const jsonToggle = document.getElementById('json-toggle');
  const jsonContainer = document.getElementById('json-container');

  // Health check on startup
  fetchHealth();

  // Handle Analysis
  async function performAnalysis() {
    const text = textInput.value.trim();
    if (!text) {
      alert('Please enter a social-media post to analyze.');
      textInput.focus();
      return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span>⚡ Analyzing...</span>';

    try {
      const startTime = performance.now();
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      const data = await response.json();
      const endTime = performance.now();
      const clientLatency = Math.round(endTime - startTime);

      if (response.ok) {
        displayResult(data, clientLatency);
      } else {
        alert('Error: ' + (data.detail || data.error || 'Failed to obtain prediction.'));
      }
    } catch (err) {
      console.error('Inference error:', err);
      alert('Could not connect to the backend prediction service. Ensure the server is running.');
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = '<span>🔍 Analyze Threat</span>';
    }
  }

  function displayResult(data, clientLatency) {
    resultPlaceholder.style.display = 'none';
    resultContent.style.display = 'block';

    const isThreat = data.label === 'Threat';
    const conf = data.confidence !== undefined ? data.confidence : (data.confidence_raw * 100);

    // Update Badge
    if (isThreat) {
      verdictBadge.className = 'verdict-badge badge-threat';
      verdictBadge.innerHTML = '🚨 THREAT DETECTED';
      meterFill.className = 'meter-fill threat';
    } else {
      verdictBadge.className = 'verdict-badge badge-normal';
      verdictBadge.innerHTML = '✅ NORMAL (SAFE)';
      meterFill.className = 'meter-fill normal';
    }

    // Update Confidence & Meter
    confidenceText.textContent = `${conf.toFixed(2)}%`;
    meterFill.style.width = `${Math.max(conf, 5)}%`;

    // Update Probabilities
    if (data.probabilities) {
      probNormal.textContent = `${data.probabilities.Normal.toFixed(2)}%`;
      probThreat.textContent = `${data.probabilities.Threat.toFixed(2)}%`;
    }

    // Explanation & Latency
    explanationText.textContent = data.explanation || 'Analyzed via fine-tuned BERT transformer model.';
    latencyText.textContent = `${data.inference_time_ms ? data.inference_time_ms.toFixed(1) : clientLatency} ms`;

    // JSON response
    rawJsonOutput.textContent = JSON.stringify(data, null, 2);
  }

  // Clear Input
  clearBtn.addEventListener('click', () => {
    textInput.value = '';
    resultPlaceholder.style.display = 'flex';
    resultContent.style.display = 'none';
    textInput.focus();
  });

  analyzeBtn.addEventListener('click', performAnalysis);

  // Allow Ctrl+Enter or Cmd+Enter to submit
  textInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      performAnalysis();
    }
  });

  // Sample Chips Click Handler
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const sampleText = chip.getAttribute('data-sample');
      if (sampleText) {
        textInput.value = sampleText;
        performAnalysis();
      }
    });
  });

  // Toggle JSON Output
  if (jsonToggle && jsonContainer) {
    jsonToggle.addEventListener('click', () => {
      if (jsonContainer.style.display === 'none' || !jsonContainer.style.display) {
        jsonContainer.style.display = 'block';
        jsonToggle.textContent = 'Hide Raw JSON Response ▴';
      } else {
        jsonContainer.style.display = 'none';
        jsonToggle.textContent = 'View Raw JSON Response ▾';
      }
    });
  }

  // Check Health Endpoint
  async function fetchHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        const statusEl = document.getElementById('system-status');
        if (statusEl) {
          statusEl.innerHTML = `<span class="pulse-dot"></span> System Online | ${data.device} | BERT Active`;
        }
      }
    } catch (e) {
      console.warn('Health check failed:', e);
    }
  }

  // Modal Image Preview Setup
  const modal = document.getElementById('img-modal');
  const modalImg = document.getElementById('modal-img');
  const modalClose = document.getElementById('modal-close');

  document.querySelectorAll('.previewable-plot').forEach(img => {
    img.addEventListener('click', () => {
      modalImg.src = img.src;
      modal.classList.add('active');
    });
  });

  if (modalClose) {
    modalClose.addEventListener('click', () => {
      modal.classList.remove('active');
    });
  }

  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.classList.remove('active');
      }
    });
  }
});
