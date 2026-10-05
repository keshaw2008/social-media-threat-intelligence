document.addEventListener('DOMContentLoaded', () => {
  const textInput = document.getElementById('text-input');
  const analyzeBtn = document.getElementById('analyze-btn');
  const clearBtn = document.getElementById('clear-btn');
  const validationMsg = document.getElementById('validation-msg');
  
  const resultContainer = document.getElementById('result-container');
  const resultPrediction = document.getElementById('result-prediction');
  const resultConfidence = document.getElementById('result-confidence');
  const confidenceBar = document.getElementById('confidence-bar');
  const resultExplanation = document.getElementById('result-explanation');

  // Check health on page load
  checkHealth();

  // Perform Analysis
  async function performAnalysis() {
    const text = textInput.value.trim();

    if (!text) {
      validationMsg.style.display = 'block';
      textInput.focus();
      return;
    }

    validationMsg.style.display = 'none';
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      const data = await response.json();

      if (response.ok) {
        displayResult(data);
      } else {
        alert('Error: ' + (data.detail || data.error || 'Failed to process post.'));
      }
    } catch (err) {
      console.error('Inference error:', err);
      alert('Could not connect to the backend prediction service. Ensure the server is online.');
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = 'Analyze Post';
    }
  }

  // Display Prediction
  function displayResult(data) {
    resultContainer.style.display = 'flex';

    const isThreat = data.label === 'Threat';
    const conf = data.confidence !== undefined ? data.confidence : (data.confidence_raw * 100);

    // Update Prediction Badge
    if (isThreat) {
      resultPrediction.className = 'stat-value prediction-badge threat';
      resultPrediction.textContent = 'THREAT';
      confidenceBar.className = 'confidence-bar-fill threat';
    } else {
      resultPrediction.className = 'stat-value prediction-badge normal';
      resultPrediction.textContent = 'NORMAL';
      confidenceBar.className = 'confidence-bar-fill normal';
    }

    // Update Confidence Value & Bar
    const confFormatted = `${conf.toFixed(2)}%`;
    resultConfidence.textContent = confFormatted;
    confidenceBar.style.width = `${Math.max(conf, 5)}%`;

    // Update Explanation
    resultExplanation.textContent = data.explanation || 
      (isThreat 
        ? 'The model flags this post as potentially threatening based on learned malicious, scam, or hostile patterns.' 
        : 'The model classifies this post as normal, safe social media communication without threat patterns.');
  }

  // Clear Input & Results
  clearBtn.addEventListener('click', () => {
    textInput.value = '';
    validationMsg.style.display = 'none';
    resultContainer.style.display = 'none';
    textInput.focus();
  });

  // Hide validation message on typing
  textInput.addEventListener('input', () => {
    if (textInput.value.trim()) {
      validationMsg.style.display = 'none';
    }
  });

  // Click handler for Analyze Post
  analyzeBtn.addEventListener('click', performAnalysis);

  // Keyboard shortcut: Ctrl+Enter or Cmd+Enter
  textInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      performAnalysis();
    }
  });

  // Quick Sample Chips
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const sample = chip.getAttribute('data-sample');
      if (sample) {
        textInput.value = sample;
        validationMsg.style.display = 'none';
        performAnalysis();
      }
    });
  });

  // Server Health Check
  async function checkHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        const statusEl = document.getElementById('system-status');
        if (statusEl) {
          statusEl.innerHTML = `<span class="status-dot"></span> Online`;
        }
      }
    } catch (e) {
      console.warn('Backend health check warning:', e);
    }
  }
});
