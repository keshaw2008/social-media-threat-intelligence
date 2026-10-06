/* ==========================================================================
   Social Media Threat Intelligence System — Academic Project Client Script
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  const postInput = document.getElementById('post-input');
  const analyzeBtn = document.getElementById('analyze-btn');
  const clearBtn = document.getElementById('clear-btn');
  const btnSpinner = document.getElementById('btn-spinner');
  const btnText = document.getElementById('btn-text');
  const charCounter = document.getElementById('char-counter');
  const validationMsg = document.getElementById('validation-msg');

  // Example Buttons
  const example1Btn = document.getElementById('example-1-btn');
  const example2Btn = document.getElementById('example-2-btn');
  const example3Btn = document.getElementById('example-3-btn');
  const example4Btn = document.getElementById('example-4-btn');

  // Result Elements
  const resultContainer = document.getElementById('result-container');
  const predictionBadge = document.getElementById('prediction-badge');
  const confidenceText = document.getElementById('confidence-text');
  const progressBarFill = document.getElementById('progress-bar-fill');
  const probNormalVal = document.getElementById('prob-normal-val');
  const probThreatVal = document.getElementById('prob-threat-val');
  const inferenceTime = document.getElementById('inference-time');

  // Character Counter & Live Input Handler
  postInput.addEventListener('input', () => {
    const len = postInput.value.length;
    charCounter.textContent = `${len} character${len === 1 ? '' : 's'}`;
    if (postInput.value.trim().length > 0) {
      validationMsg.style.display = 'none';
    }
  });

  // Example Button Handlers (Populates textarea without auto-running)
  function setExampleText(text) {
    postInput.value = text;
    postInput.dispatchEvent(new Event('input'));
    validationMsg.style.display = 'none';
    postInput.focus();
  }

  if (example1Btn) {
    example1Btn.addEventListener('click', () => {
      setExampleText('Had a great day with my friends today.');
    });
  }

  if (example2Btn) {
    example2Btn.addEventListener('click', () => {
      setExampleText('Congratulations! You won a free phone. Click this link immediately!');
    });
  }

  if (example3Btn) {
    example3Btn.addEventListener('click', () => {
      setExampleText('We are going to totally destroy the enemy squad in Valorant tonight!');
    });
  }

  if (example4Btn) {
    example4Btn.addEventListener('click', () => {
      setExampleText('We know where your children go to school in Chicago. Keep your mouth shut.');
    });
  }

  // Clear Button
  clearBtn.addEventListener('click', () => {
    postInput.value = '';
    postInput.dispatchEvent(new Event('input'));
    validationMsg.style.display = 'none';
    resultContainer.style.display = 'none';
    postInput.focus();
  });

  // Analyze Button & Keyboard Shortcut (Ctrl+Enter / Cmd+Enter)
  analyzeBtn.addEventListener('click', performAnalysis);

  postInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      performAnalysis();
    }
  });

  async function performAnalysis() {
    const text = postInput.value.trim();

    if (!text) {
      validationMsg.style.display = 'inline';
      postInput.focus();
      return;
    }

    validationMsg.style.display = 'none';
    setLoading(true);

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
        displayPrediction(data, roundtripMs);
      } else {
        alert('Analysis Error: ' + (data.detail || data.error || 'Failed to process post.'));
      }
    } catch (err) {
      console.error('Inference error:', err);
      alert('Could not connect to the backend server. Please verify that the server is active.');
    } finally {
      setLoading(false);
    }
  }

  function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading;
    if (isLoading) {
      btnSpinner.style.display = 'inline-block';
      btnText.textContent = 'Analyzing...';
    } else {
      btnSpinner.style.display = 'none';
      btnText.textContent = 'Analyze Post';
    }
  }

  function displayPrediction(data, roundtripMs) {
    resultContainer.style.display = 'block';

    const isThreat = data.label === 'Threat';
    const conf = data.confidence !== undefined ? Number(data.confidence) : (Number(data.confidence_raw) * 100);
    const latency = data.inference_time_ms ? `${data.inference_time_ms.toFixed(1)} ms` : `${roundtripMs} ms`;

    // Latency
    inferenceTime.textContent = `Inference: ${latency}`;

    // Prediction Badge & Progress Bar Styling
    if (isThreat) {
      predictionBadge.className = 'badge-prediction badge-threat';
      predictionBadge.textContent = 'Threat';
      progressBarFill.className = 'progress-bar-fill bar-threat';
    } else {
      predictionBadge.className = 'badge-prediction badge-normal';
      predictionBadge.textContent = 'Normal';
      progressBarFill.className = 'progress-bar-fill bar-normal';
    }

    // Calibrated Confidence
    confidenceText.textContent = `${conf.toFixed(2)}%`;
    progressBarFill.style.width = `${Math.min(100, Math.max(0, conf))}%`;

    // Class Probabilities (API returns values as percentages, format directly)
    if (data.probabilities) {
      const pNormal = Number(data.probabilities.Normal);
      const pThreat = Number(data.probabilities.Threat);

      probNormalVal.textContent = `${pNormal.toFixed(2)}%`;
      probThreatVal.textContent = `${pThreat.toFixed(2)}%`;
    }
  }
});
