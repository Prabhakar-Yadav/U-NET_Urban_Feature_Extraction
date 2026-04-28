const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const selectedFile = document.getElementById("selected-file");
const uploadForm = document.getElementById("upload-form");
const modelMode = document.getElementById("model-mode");
const modelId = document.getElementById("model-id");
const submitButton = document.getElementById("submit-button");
const statusCard = document.getElementById("status-card");
const resultsSection = document.getElementById("results-section");

function setStatus(message, isError = false) {
  statusCard.textContent = message;
  statusCard.classList.remove("hidden", "error");
  if (isError) {
    statusCard.classList.add("error");
  }
}

function clearStatus() {
  statusCard.classList.add("hidden");
  statusCard.classList.remove("error");
  statusCard.textContent = "";
}

function updateSelectedFile() {
  const file = fileInput.files?.[0];
  selectedFile.textContent = file ? file.name : "No file selected yet";
}

function toggleManualModel() {
  const manual = modelMode.value === "manual";
  modelId.disabled = !manual;
  if (!manual) {
    modelId.value = "";
  }
}

function createClassBreakdown(items) {
  const container = document.getElementById("class-breakdown");
  container.innerHTML = "";

  items.forEach((item) => {
    const block = document.createElement("div");
    block.className = "class-item";

    const row = document.createElement("div");
    row.className = "class-row";
    row.innerHTML = `
      <span class="class-name">${item.class_name}</span>
      <span class="class-percentage">${item.percentage.toFixed(2)}%</span>
    `;

    const barTrack = document.createElement("div");
    barTrack.className = "bar-track";

    const barFill = document.createElement("div");
    barFill.className = "bar-fill";
    barFill.style.width = `${Math.max(2, item.percentage)}%`;
    barFill.style.background = `linear-gradient(90deg, rgba(${item.color_rgb.join(",")}, 0.95), rgba(${item.color_rgb.join(",")}, 0.65))`;
    barTrack.appendChild(barFill);

    block.appendChild(row);
    block.appendChild(barTrack);
    container.appendChild(block);
  });
}

function createLegend(items) {
  const container = document.getElementById("legend-row");
  container.innerHTML = "";

  items.forEach((item) => {
    const block = document.createElement("div");
    block.className = "legend-item";
    block.innerHTML = `
      <span class="legend-swatch" style="background: rgb(${item.color_rgb.join(",")})"></span>
      <span>${item.class_name}</span>
    `;
    container.appendChild(block);
  });
}

function createCandidateRanking(items) {
  const container = document.getElementById("candidate-ranking");
  container.innerHTML = "";

  items.forEach((item, index) => {
    const block = document.createElement("div");
    block.className = "candidate-item";
    block.innerHTML = `
      <div class="candidate-top">
        <span class="candidate-name">${index === 0 ? "Selected" : "Alternative"}: ${item.display_name}</span>
        <span class="candidate-score">Score ${item.score.toFixed(4)}</span>
      </div>
      <div class="candidate-details">
        <span>${item.dataset_name}</span>
        <span>${item.model_type.replaceAll("_", " ")}</span>
        <span>Dominant class: ${item.dominant_class}</span>
        <span>${item.primary_metric_label}: ${(item.validation_accuracy * 100).toFixed(2)}%</span>
        ${item.secondary_metric_label && item.secondary_metric_value !== null ? `<span>${item.secondary_metric_label}: ${(item.secondary_metric_value * 100).toFixed(2)}%</span>` : ""}
        <span>Confidence mean: ${(item.score_breakdown.confidence_mean * 100).toFixed(1)}%</span>
        <span>Domain similarity: ${(item.score_breakdown.domain_similarity * 100).toFixed(1)}%</span>
      </div>
    `;
    container.appendChild(block);
  });
}

function createDownloadLinks(artifacts) {
  const labels = {
    marked_preview_url: "Marked preview image",
    segmentation_preview_url: "Segmentation preview image",
    marked_full_url: "Marked full-size image",
    segmentation_full_url: "Segmentation full-size image",
    labels_npy_url: "Patch label grid (.npy)",
    geotiff_url: "GeoTIFF labels",
    class_shapefiles_zip_url: "Class shapefiles (.zip)",
    summary_json_url: "Prediction summary (.json)",
    summary_text_url: "Simple summary (.txt)",
    patch_predictions_csv_url: "Patch predictions (.csv)",
    class_percentages_url: "Class percentages (.json)",
  };

  const container = document.getElementById("download-links");
  container.innerHTML = "";

  Object.entries(labels).forEach(([key, label]) => {
    const url = artifacts[key];
    if (!url) {
      return;
    }

    const block = document.createElement("div");
    block.className = "download-item";
    block.innerHTML = `
      <span>${label}</span>
      <a class="download-link" href="${url}" target="_blank" rel="noopener">Open</a>
    `;
    container.appendChild(block);
  });
}

function renderResults(result) {
  const legendMap = new Map(result.legend.map((item) => [item.class_id, item.color_rgb]));
  const classesForBars = result.class_percentages.map((item) => ({
    ...item,
    color_rgb: legendMap.get(item.class_id) || [180, 180, 180],
  }));

  document.getElementById("selected-model-name").textContent = result.selected_model.display_name;
  document.getElementById("selected-model-dataset").textContent = `${result.selected_model.dataset_name} • ${result.selected_model.model_type.replaceAll("_", " ")}`;
  document.getElementById("dominant-class-name").textContent = result.dominant_class;
  document.getElementById("fit-score").textContent = `Model fit score: ${result.score_breakdown.final_score.toFixed(4)}`;
  document.getElementById("input-shape").textContent = `${result.image.width} x ${result.image.height}`;
  document.getElementById("input-loader").textContent = `Loaded with ${result.image.loader}`;
  document.getElementById("result-note").textContent = result.note;

  document.getElementById("input-preview").src = result.artifacts.input_preview_url;
  document.getElementById("input-preview-link").href = result.artifacts.input_preview_url;
  document.getElementById("marked-preview").src = result.artifacts.marked_preview_url;
  document.getElementById("marked-link").href = result.artifacts.marked_full_url || result.artifacts.marked_preview_url;
  document.getElementById("segmentation-preview").src = result.artifacts.segmentation_preview_url;
  document.getElementById("segmentation-link").href = result.artifacts.segmentation_full_url || result.artifacts.segmentation_preview_url;

  createClassBreakdown(classesForBars);
  createLegend(result.legend);
  createCandidateRanking(result.candidate_rankings);
  createDownloadLinks(result.artifacts);

  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const files = event.dataTransfer?.files;
  if (files && files.length) {
    const transfer = new DataTransfer();
    Array.from(files).forEach((file) => transfer.items.add(file));
    fileInput.files = transfer.files;
    updateSelectedFile();
  }
});

fileInput.addEventListener("change", updateSelectedFile);
modelMode.addEventListener("change", toggleManualModel);

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearStatus();

  const file = fileInput.files?.[0];
  if (!file) {
    setStatus("Choose an image file before running prediction.", true);
    return;
  }

  if (modelMode.value === "manual" && !modelId.value) {
    setStatus("Choose a manual model or switch back to Auto mode.", true);
    return;
  }

  submitButton.disabled = true;
  setStatus("Processing image with the trained models. Large scenes can take a little longer.");

  const formData = new FormData(uploadForm);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    setStatus("Prediction finished. Scroll down to review the selected model and result maps.");
    renderResults(data);
  } catch (error) {
    setStatus(error.message || "Prediction failed.", true);
  } finally {
    submitButton.disabled = false;
  }
});

toggleManualModel();
