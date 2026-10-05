document.addEventListener('DOMContentLoaded', () => {
  const postInput = document.getElementById('post-input');
  const analyzeBtn = document.getElementById('analyze-btn');
  const clearBtn = document.getElementById('clear-btn');
  const charCounter = document.getElementById('char-counter');
  const validationMsg = document.getElementById('validation-msg');
  const btnSpinner = document.getElementById('btn-spinner');
  const btnText = document.getElementById('btn-text');

  const resultWrapper = document.getElementById('result-wrapper');
  const predictionBadge = document.getElementById('prediction-badge');
  const verdictIcon = document.getElementById('verdict-icon');
  const verdictText = document.getElementById('verdict-text');
  const confidenceValue = document.getElementById('confidence-value');
  const meterBar = document.getElementById('meter-bar');
  const probNormal = document.getElementById('prob-normal');
  const probThreat = document.getElementById('prob-threat');
  const resultLatency = document.getElementById('result-latency');
  const explanationText = document.getElementById('explanation-text');

  // Initial health check
  fetchHealthStatus();

  // Character counter listener
  postInput.addEventListener('input', updateCharCount);

  function updateCharCount() {
    const count = postInput.value.length;
    charCounter.textContent = `${count} character${count === 1 ? '' : 's'}`;
    if (postInput.value.trim().length > 0) {
      validationMsg.style.display = 'none';
    }
  }

  // Handle Analysis
  async function performAnalysis() {
    const text = postInput.value.trim();

    if (!text) {
      validationMsg.style.display = 'block';
      postInput.focus();
      return;
    }

    validationMsg.style.display = 'none';
    setLoadingState(true);

    try {
      const startTime = performance.now();
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      const data = await response.json();
      const endTime = performance.now();
      const roundtripMs = Math.round(endTime - startTime);

      if (response.ok) {
        displayResult(data, roundtripMs);
      } else {
        alert('Error: ' + (data.detail || data.error || 'Failed to analyze post.'));
      }
    } catch (err) {
      console.error('Inference error:', err);
      alert('Could not connect to the backend prediction service. Ensure the server is running.');
    } finally {
      setLoadingState(false);
    }
  }

  function setLoadingState(isLoading) {
    analyzeBtn.disabled = isLoading;
    if (isLoading) {
      btnSpinner.style.display = 'inline-block';
      btnText.textContent = 'Analyzing...';
    } else {
      btnSpinner.style.display = 'none';
      btnText.textContent = 'Analyze Post';
    }
  }

  function displayResult(data, roundtripMs) {
    resultWrapper.style.display = 'flex';

    const isThreat = data.label === 'Threat';
    const conf = data.confidence !== undefined ? data.confidence : (data.confidence_raw * 100);
    const latency = data.inference_time_ms ? `${data.inference_time_ms.toFixed(1)} ms` : `${roundtripMs} ms`;

    // Latency pill
    resultLatency.textContent = `⚡ ${latency}`;

    // Verdict Badge & Colors
    if (isThreat) {
      predictionBadge.className = 'verdict-badge verdict-threat';
      verdictIcon.textContent = '⚠️';
      verdictText.textContent = 'THREAT';
      meterBar.className = 'meter-bar meter-threat';
    } else {
      predictionBadge.className = 'verdict-badge verdict-normal';
      verdictIcon.textContent = '🛡️';
      verdictText.textContent = 'NORMAL';
      meterBar.className = 'meter-bar meter-normal';
    }

    // Confidence & Meter
    confidenceValue.textContent = `${conf.toFixed(2)}%`;
    meterBar.style.width = `${Math.max(conf, 5)}%`;

    // Probabilities
    if (data.probabilities) {
      probNormal.textContent = `${data.probabilities.Normal.toFixed(2)}%`;
      probThreat.textContent = `${data.probabilities.Threat.toFixed(2)}%`;
    }

    // Explanation
    explanationText.textContent = data.explanation || 
      (isThreat 
        ? 'The model classifies this post as potentially threatening based on learned malicious, scam, or hostile patterns.' 
        : 'The model classifies this post as normal, safe social media communication without threat patterns.');

    // Smooth scroll to result
    resultWrapper.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Clear Button
  clearBtn.addEventListener('click', () => {
    postInput.value = '';
    updateCharCount();
    validationMsg.style.display = 'none';
    resultWrapper.style.display = 'none';
    postInput.focus();
  });

  // Analyze Button click
  analyzeBtn.addEventListener('click', performAnalysis);

  // Keyboard shortcut: Ctrl+Enter / Cmd+Enter
  postInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      performAnalysis();
    }
  });

  // Sample Chips
  document.querySelectorAll('.sample-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const sample = chip.getAttribute('data-sample');
      if (sample) {
        postInput.value = sample;
        updateCharCount();
        validationMsg.style.display = 'none';
        performAnalysis();
      }
    });
  });

  // Health Status check
  async function fetchHealthStatus() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        const statusEl = document.getElementById('system-status');
        if (statusEl) {
          const deviceLabel = data.device.includes('ONNX') ? 'ONNX INT8 Active' : data.device;
          statusEl.innerHTML = `<span class="pulse-dot"></span><span>${deviceLabel}</span>`;
        }
      }
    } catch (e) {
      console.warn('Backend health check note:', e);
    }
  }
});
