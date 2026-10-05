/* ==========================================================================
   Social Media Threat Intelligence — Academic Project Client JavaScript
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  const postInput = document.getElementById('post-input');
  const analyzeBtn = document.getElementById('analyze-btn');
  const clearBtn = document.getElementById('clear-btn');
  const btnSpinner = document.getElementById('btn-spinner');
  const btnText = document.getElementById('btn-text');
  const charCounter = document.getElementById('char-counter');
  const validationMsg = document.getElementById('validation-msg');

  const exampleNormalBtn = document.getElementById('example-normal-btn');
  const exampleThreatBtn = document.getElementById('example-threat-btn');

  const resultContainer = document.getElementById('result-container');
  const predictionBadge = document.getElementById('prediction-badge');
  const confidenceText = document.getElementById('confidence-text');
  const progressBarFill = document.getElementById('progress-bar-fill');
  const probNormalVal = document.getElementById('prob-normal-val');
  const probThreatVal = document.getElementById('prob-threat-val');
  const inferenceTime = document.getElementById('inference-time');

  // Character counter
  postInput.addEventListener('input', () => {
    const len = postInput.value.length;
    charCounter.textContent = `${len} character${len === 1 ? '' : 's'}`;
    if (postInput.value.trim().length > 0) {
      validationMsg.style.display = 'none';
    }
  });

  // Example Buttons
  exampleNormalBtn.addEventListener('click', () => {
    postInput.value = 'Had a great day with my friends today.';
    postInput.dispatchEvent(new Event('input'));
    postInput.focus();
  });

  exampleThreatBtn.addEventListener('click', () => {
    postInput.value = 'Congratulations! You won a free phone. Click this link immediately!';
    postInput.dispatchEvent(new Event('input'));
    postInput.focus();
  });

  // Clear Button
  clearBtn.addEventListener('click', () => {
    postInput.value = '';
    postInput.dispatchEvent(new Event('input'));
    validationMsg.style.display = 'none';
    resultContainer.style.display = 'none';
    postInput.focus();
  });

  // Analyze Button & Keyboard Shortcut (Ctrl+Enter)
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
      alert('Could not connect to the backend server. Please check if the server is running.');
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
    const conf = data.confidence !== undefined ? data.confidence : (data.confidence_raw * 100);
    const latency = data.inference_time_ms ? `${data.inference_time_ms.toFixed(1)} ms` : `${roundtripMs} ms`;

    // Latency
    inferenceTime.textContent = `Inference: ${latency}`;

    // Prediction Badge
    if (isThreat) {
      predictionBadge.className = 'badge-prediction badge-threat';
      predictionBadge.textContent = 'THREAT';
      progressBarFill.className = 'progress-bar-fill bar-threat';
    } else {
      predictionBadge.className = 'badge-prediction badge-normal';
      predictionBadge.textContent = 'NORMAL';
      progressBarFill.className = 'progress-bar-fill bar-normal';
    }

    // Confidence
    confidenceText.textContent = `${conf.toFixed(2)}%`;
    progressBarFill.style.width = `${Math.min(100, Math.max(0, conf))}%`;

    // Class Probabilities
    if (data.probabilities) {
      const pNormal = (data.probabilities.Normal * 100).toFixed(2);
      const pThreat = (data.probabilities.Threat * 100).toFixed(2);
      probNormalVal.textContent = `${pNormal}%`;
      probThreatVal.textContent = `${pThreat}%`;
    }
  }
});
