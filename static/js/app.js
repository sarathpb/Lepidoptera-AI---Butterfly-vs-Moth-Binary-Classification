/**
 * Lepidoptera AI - Client Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const samplesGrid = document.getElementById("samplesGrid");
  const resultsPlaceholder = document.getElementById("resultsPlaceholder");
  const resultsContainer = document.getElementById("resultsContainer");

  // Results elements
  const verdictCard = document.getElementById("verdictCard");
  const predictedClassTitle = document.getElementById("predictedClassTitle");
  const predictedLatinName = document.getElementById("predictedLatinName");
  const confidenceValue = document.getElementById("confidenceValue");
  const gaugeProgress = document.getElementById("gaugeProgress");

  const butterflyProbText = document.getElementById("butterflyProbText");
  const butterflyBar = document.getElementById("butterflyBar");
  const mothProbText = document.getElementById("mothProbText");
  const mothBar = document.getElementById("mothBar");

  const mainSpecimenImg = document.getElementById("mainSpecimenImg");
  const splitOriginalImg = document.getElementById("splitOriginalImg");
  const splitHeatmapImg = document.getElementById("splitHeatmapImg");
  const singleView = document.getElementById("singleView");
  const splitView = document.getElementById("splitView");

  const viewOriginalBtn = document.getElementById("viewOriginalBtn");
  const viewHeatmapBtn = document.getElementById("viewHeatmapBtn");
  const viewSplitBtn = document.getElementById("viewSplitBtn");

  const diagVibrancy = document.getElementById("diagVibrancy");
  const diagCamo = document.getElementById("diagCamo");
  const diagPosture = document.getElementById("diagPosture");
  const diagEntropy = document.getElementById("diagEntropy");
  const diagnosticPointsList = document.getElementById("diagnosticPointsList");

  const cnnProbVal = document.getElementById("cnnProbVal");
  const rfProbVal = document.getElementById("rfProbVal");

  // Modal elements
  const openMetricsBtn = document.getElementById("openMetricsBtn");
  const closeMetricsBtn = document.getElementById("closeMetricsBtn");
  const metricsModal = document.getElementById("metricsModal");

  // State
  let currentOriginalUrl = "";
  let currentHeatmapUrl = "";
  let activeSampleCard = null;

  // Initialize
  loadSamples();
  setupEventListeners();

  function setupEventListeners() {
    // Drop Zone Click
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        fileInput.click();
      }
    });

    // File Input Change
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        uploadFile(e.target.files[0]);
      }
    });

    // Drag & Drop
    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        uploadFile(e.dataTransfer.files[0]);
      }
    });

    // Visual View Switchers
    viewOriginalBtn.addEventListener("click", () => setViewerMode("original"));
    viewHeatmapBtn.addEventListener("click", () => setViewerMode("heatmap"));
    viewSplitBtn.addEventListener("click", () => setViewerMode("split"));

    // Modal
    openMetricsBtn.addEventListener("click", () => {
      metricsModal.classList.remove("hidden");
      loadMetrics();
    });

    closeMetricsBtn.addEventListener("click", () => {
      metricsModal.classList.add("hidden");
    });

    metricsModal.addEventListener("click", (e) => {
      if (e.target === metricsModal) {
        metricsModal.classList.add("hidden");
      }
    });
  }

  // Fetch and Render Curated Samples
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      if (!data.samples || data.samples.length === 0) return;

      samplesGrid.innerHTML = "";
      data.samples.forEach((sample, idx) => {
        const card = document.createElement("div");
        card.className = "sample-card";
        card.tabIndex = 0;
        card.setAttribute("role", "button");
        card.setAttribute("aria-label", `Classify sample ${sample.variety}`);

        const tagClass = sample.label.toLowerCase() === "butterfly" ? "tag-butterfly" : "tag-moth";

        card.innerHTML = `
          <div class="sample-thumb-wrap">
            <img src="${sample.url}" alt="${sample.variety}" loading="lazy">
            <span class="sample-tag-pill ${tagClass}">${sample.label}</span>
          </div>
          <div class="sample-info">
            <span class="sample-name">${sample.variety}</span>
            <span class="sample-latin">${sample.latin}</span>
            <span class="sample-feature-tag">${sample.badge}</span>
          </div>
        `;

        card.addEventListener("click", () => selectSample(sample, card));
        card.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            selectSample(sample, card);
          }
        });

        samplesGrid.appendChild(card);

        // Auto-select the first sample on initial page load for instant WOW experience
        if (idx === 0) {
          selectSample(sample, card);
        }
      });
    } catch (err) {
      console.error("Failed to load curated samples:", err);
    }
  }

  // Select Sample from Grid
  async function selectSample(sample, cardEl) {
    if (activeSampleCard) {
      activeSampleCard.classList.remove("active");
    }
    cardEl.classList.add("active");
    activeSampleCard = cardEl;

    currentOriginalUrl = sample.url;
    showLoading();

    const formData = new FormData();
    formData.append("sample_name", sample.name);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      renderPrediction(data, sample.url);
    } catch (err) {
      console.error("Prediction error:", err);
      alert("Failed to classify sample. Check server logs.");
    }
  }

  // Upload Local File
  async function uploadFile(file) {
    if (activeSampleCard) {
      activeSampleCard.classList.remove("active");
      activeSampleCard = null;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      currentOriginalUrl = e.target.result;
    };
    reader.readAsDataURL(file);

    showLoading();

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      renderPrediction(data, currentOriginalUrl);
    } catch (err) {
      console.error("Prediction upload error:", err);
      alert("Failed to analyze image.");
    }
  }

  function showLoading() {
    resultsPlaceholder.classList.add("hidden");
    resultsContainer.classList.remove("hidden");
    predictedClassTitle.textContent = "Analyzing Specimen...";
    confidenceValue.textContent = "--%";
  }

  // Render Prediction Results to UI
  function renderPrediction(data, originalImgSrc) {
    resultsPlaceholder.classList.add("hidden");
    resultsContainer.classList.remove("hidden");

    const isButterfly = data.predicted_class === "butterfly";
    currentOriginalUrl = originalImgSrc;
    currentHeatmapUrl = data.heatmap_overlay || originalImgSrc;

    // 1. Verdict Card
    verdictCard.className = `verdict-card ${isButterfly ? "verdict-butterfly" : "verdict-moth"}`;
    predictedClassTitle.textContent = isButterfly ? "Butterfly" : "Moth";
    predictedLatinName.textContent = data.taxonomy_info.title;

    // 2. Circular Gauge
    const conf = data.confidence;
    confidenceValue.textContent = `${conf.toFixed(1)}%`;
    const circumference = 2 * Math.PI * 42; // ~264
    const offset = circumference - (conf / 100) * circumference;
    gaugeProgress.style.strokeDashoffset = offset;

    // 3. Probability Bars
    const bProb = data.probabilities.butterfly;
    const mProb = data.probabilities.moth;
    butterflyProbText.textContent = `${bProb.toFixed(1)}%`;
    butterflyBar.style.width = `${bProb}%`;
    mothProbText.textContent = `${mProb.toFixed(1)}%`;
    mothBar.style.width = `${mProb}%`;

    // 4. Specimen Image & Heatmap
    mainSpecimenImg.src = currentOriginalUrl;
    splitOriginalImg.src = currentOriginalUrl;
    splitHeatmapImg.src = currentHeatmapUrl;
    setViewerMode("original");

    // 5. Morphological Diagnostics
    const m = data.morphological_metrics;
    diagVibrancy.textContent = `${(m.vibrant_ratio * 100).toFixed(0)}% (Sat: ${(m.mean_saturation * 100).toFixed(0)}%)`;
    diagCamo.textContent = `${(m.edge_density * 100).toFixed(0)}% density`;
    diagPosture.textContent = m.posture_tendency;
    diagEntropy.textContent = m.hue_entropy.toFixed(2);

    // Diagnostic bullet points
    diagnosticPointsList.innerHTML = "";
    (data.diagnostic_points || []).forEach((pt) => {
      const li = document.createElement("li");
      li.textContent = pt;
      diagnosticPointsList.appendChild(li);
    });

    // 6. Dual-Engine Consensus
    if (data.models.cnn_available && data.models.cnn_probabilities.butterfly !== null) {
      const cnnB = data.models.cnn_probabilities.butterfly;
      cnnProbVal.textContent = `${cnnB.toFixed(1)}% Butterfly / ${(100 - cnnB).toFixed(1)}% Moth`;
    } else {
      cnnProbVal.textContent = "Standby (PyTorch Model Ready)";
    }

    const rfB = data.models.rf_probabilities.butterfly;
    rfProbVal.textContent = `${rfB.toFixed(1)}% Butterfly / ${(100 - rfB).toFixed(1)}% Moth`;
  }

  // Switcher between Original, Heatmap, and Side-by-Side Split
  function setViewerMode(mode) {
    [viewOriginalBtn, viewHeatmapBtn, viewSplitBtn].forEach((b) => b.classList.remove("active"));

    if (mode === "original") {
      viewOriginalBtn.classList.add("active");
      singleView.classList.remove("hidden");
      splitView.classList.add("hidden");
      mainSpecimenImg.src = currentOriginalUrl;
    } else if (mode === "heatmap") {
      viewHeatmapBtn.classList.add("active");
      singleView.classList.remove("hidden");
      splitView.classList.add("hidden");
      mainSpecimenImg.src = currentHeatmapUrl;
    } else if (mode === "split") {
      viewSplitBtn.classList.add("active");
      singleView.classList.add("hidden");
      splitView.classList.remove("hidden");
    }
  }

  // Fetch Metrics from /api/metrics
  async function loadMetrics() {
    try {
      const res = await fetch("/api/metrics");
      const d = await res.json();
      document.getElementById("mAccuracy").textContent = `${d.accuracy}%`;
      document.getElementById("mPrecision").textContent = `${d.precision}%`;
      document.getElementById("mRecall").textContent = `${d.recall}%`;
      document.getElementById("mF1").textContent = `${d.f1_score}%`;

      if (d.confusion_matrix) {
        const [[tp, fn], [fp, tn]] = d.confusion_matrix;
        document.getElementById("cmTP").textContent = `${tp} (True Pos)`;
        document.getElementById("cmFN").textContent = `${fn} (False Neg)`;
        document.getElementById("cmFP").textContent = `${fp} (False Pos)`;
        document.getElementById("cmTN").textContent = `${tn} (True Neg)`;
      }
    } catch (e) {
      console.warn("Metrics load error:", e);
    }
  }
});
