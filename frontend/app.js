import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const state = {
  scans: [],
  selectedScan: null,
  selectedLevel: "2",
  volume: null,
  sliceShape: null,
  sliceLevel: "0",
  detections: [],
  selectedDetection: null,
  selectedMolecule: null,
  labelsHidden: false,
  expandedMolecules: new Set(),
  currentSlices: { x: 0, y: 0, z: 0 },
  activeSliceAxis: "z",
  autoRotate: false,
  overlayColorMode: "type",
  gvi: null,
  crowdingSummary: null,
  neighborMap: new Map(),
  datasetAnalysis: null,
  minOcclusionFilter: 0
};

function getMoleculeGroups() {
  const map = new Map();
  state.detections.forEach((det) => {
    const key = det.molecule || det.type;
    if (!map.has(key)) {
      map.set(key, { molecule: key, label: det.type, picks: [] });
    }
    map.get(key).picks.push(det);
  });
  return [...map.values()];
}

function detectionKey(detection) {
  return `${detection?.molecule || detection?.type || "unknown"}:${detection?.id || ""}`;
}

function refreshAnnotationNumbers() {
  annotationNumbers = new Map();
  getMoleculeGroups().forEach((group) => {
    group.picks.forEach((detection, index) => {
      annotationNumbers.set(detectionKey(detection), index + 1);
    });
  });
}

function annotationLabel(detection) {
  if (state.selectedDetection) {
    return detection.type;
  }
  return String(annotationNumbers.get(detectionKey(detection)) || "?");
}

const highDetailSliceLevel = "0";
const highDetailSliceMaxSize = "512";
const interactiveSliceMaxSize = "420";
const slicePrecacheConcurrency = Math.max(3, Math.min(8, (navigator.hardwareConcurrency || 6) - 1));
const slicePrecacheBatchSize = slicePrecacheConcurrency * 2;
const sliceAxes = ["x", "y", "z"];
const sliceCache = new Map();
const sliceTimers = { x: null, y: null, z: null };
const prefetchTimers = { x: null, y: null, z: null };
const annotationLimit = 36;
let precacheRunId = 0;
let annotationItems = [];
let annotationNumbers = new Map();
let analysisRequestScan = null;

const elements = {
  viewer: document.querySelector("#volume-viewer"),
  message: document.querySelector("#viewer-message"),
  annotations: document.querySelector("#viewer-annotations"),
  scanSelect: document.querySelector("#scan-select"),
  levelSelect: document.querySelector("#level-select"),
  strideSelect: document.querySelector("#stride-select"),
  pointLimit: document.querySelector("#point-limit"),
  pointSize: document.querySelector("#point-size"),
  upload: document.querySelector("#zarr-upload"),
  labelsUpload: document.querySelector("#labels-upload"),
  localZarrPath: document.querySelector("#local-zarr-path"),
  openLocalZarr: document.querySelector("#open-local-zarr"),
  loadStatus: document.querySelector("#load-status"),
  progress: document.querySelector("#load-progress"),
  detectionCount: document.querySelector("#detection-count"),
  analysisStatus: document.querySelector("#analysis-status"),
  analysisSummary: document.querySelector("#analysis-summary"),
  analysisReport: document.querySelector("#analysis-report"),
  crowdingSummary: document.querySelector("#crowding-summary"),
  showAllBtn: document.querySelector("#show-all-molecules"),
  detectionList: document.querySelector("#detection-list"),
  selectedId: document.querySelector("#selected-id"),
  selectedType: document.querySelector("#selected-type"),
  selectedConfidence: document.querySelector("#selected-confidence"),
  selectedPosition: document.querySelector("#selected-position"),
  selectedNotes: document.querySelector("#selected-notes"),
  sliceCanvases: {
    x: document.querySelector("#x-slice-canvas"),
    y: document.querySelector("#y-slice-canvas"),
    z: document.querySelector("#z-slice-canvas")
  },
  sliceLabels: {
    x: document.querySelector("#x-slice-label"),
    y: document.querySelector("#y-slice-label"),
    z: document.querySelector("#z-slice-label")
  },
  sliceSliders: {
    x: document.querySelector("#x-slice-slider"),
    y: document.querySelector("#y-slice-slider"),
    z: document.querySelector("#z-slice-slider")
  },
  sliceSliderValues: {
    x: document.querySelector("#x-slice-value"),
    y: document.querySelector("#y-slice-value"),
    z: document.querySelector("#z-slice-value")
  },
  sliceSliderControls: {
    x: document.querySelector("#x-slice-value")?.closest(".slice-slider-control"),
    y: document.querySelector("#y-slice-value")?.closest(".slice-slider-control"),
    z: document.querySelector("#z-slice-value")?.closest(".slice-slider-control")
  },
  runInference: document.querySelector("#run-inference"),
  inferenceProgress: document.querySelector("#inference-progress"),
  inferenceProgressFill: document.querySelector("#inference-progress-fill"),
  inferenceStatus: document.querySelector("#inference-status"),
  runCrowding: document.querySelector("#run-crowding"),
  crowdingProgress: document.querySelector("#crowding-progress"),
  crowdingProgressFill: document.querySelector("#crowding-progress-fill"),
  crowdingStatus: document.querySelector("#crowding-status"),
  qaStatus: document.querySelector("#qa-status"),
  qaDetail: document.querySelector("#qa-detail"),
  overlayColorMode: document.querySelector("#overlay-color-mode"),
  selectedExposure: document.querySelector("#selected-exposure"),
  selectedCleanExtraction: document.querySelector("#selected-clean-extraction"),
  selectedAnisotropy: document.querySelector("#selected-anisotropy"),
  selectedOpenDirection: document.querySelector("#selected-open-direction"),
  selectedGnnExposure: document.querySelector("#selected-gnn-exposure"),
  selectedNeighbors: document.querySelector("#selected-neighbors"),
  graphPanel: document.querySelector("#graph-panel"),
  graphViewer: document.querySelector("#graph-viewer"),
  graphCaption: document.querySelector("#graph-caption"),
  graphPickLabel: document.querySelector("#graph-pick-label"),
  resetGraphCamera: document.querySelector("#reset-graph-camera"),
  viewerRow: document.querySelector(".viewer-row"),
  occlusionFilterPanel: document.querySelector("#occlusion-filter-panel"),
  occlusionFilterSlider: document.querySelector("#occlusion-filter-slider"),
  occlusionFilterValue: document.querySelector("#occlusion-filter-value"),
  occlusionFilterStats: document.querySelector("#occlusion-filter-stats"),
  exportCleanPicks: document.querySelector("#export-clean-picks")
};

const scene = new THREE.Scene();
scene.background = null;

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(1.55, 1.25, 1.65);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x000000, 0);
elements.viewer.appendChild(renderer.domElement);

const mainSliceCanvas = document.createElement("canvas");
mainSliceCanvas.className = "main-slice-canvas";
mainSliceCanvas.width = 1;
mainSliceCanvas.height = 1;
mainSliceCanvas.hidden = true;
elements.viewer.appendChild(mainSliceCanvas);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

const volumeGroup = new THREE.Group();
scene.add(volumeGroup);

const picksGroup = new THREE.Group();
picksGroup.name = "pick-overlays";
volumeGroup.add(picksGroup);

const slicePlaneObjects = { x: null, y: null, z: null };
const sliceRequestIds = { x: 0, y: 0, z: 0 };

scene.add(new THREE.AmbientLight(0xffffff, 1.05));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.35);
keyLight.position.set(2.2, 2.4, 3);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x8ddbf0, 0.7);
rimLight.position.set(-2.5, -1.8, -1.5);
scene.add(rimLight);

const GRAPH_SCALE = 0.0025;
const graphScene = new THREE.Scene();
graphScene.background = new THREE.Color(0xf4f7f8);

const graphCamera = new THREE.PerspectiveCamera(50, 1, 0.001, 100);
graphCamera.position.set(0.85, 0.65, 1.05);

const graphRenderer = new THREE.WebGLRenderer({ antialias: true });
graphRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
elements.graphViewer.appendChild(graphRenderer.domElement);

const graphControls = new OrbitControls(graphCamera, graphRenderer.domElement);
graphControls.enableDamping = true;
graphControls.dampingFactor = 0.08;
graphControls.target.set(0, 0, 0);

const graphGroup = new THREE.Group();
const hemisphereGroup = new THREE.Group();
graphGroup.add(hemisphereGroup);
graphScene.add(graphGroup);
graphScene.add(new THREE.AmbientLight(0xffffff, 1.15));
const graphKeyLight = new THREE.DirectionalLight(0xffffff, 1.1);
graphKeyLight.position.set(1.5, 2, 1.2);
graphScene.add(graphKeyLight);

function setStatus(text, active = false) {
  elements.loadStatus.textContent = text;
  elements.loadStatus.classList.toggle("active", active);
}

function setProgress(percent) {
  if (elements.progress) {
    elements.progress.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  }
}

function showLoadingScreen(title, detail = "") {
  elements.message.hidden = false;
  elements.message.classList.add("is-cache-status");
  elements.message.innerHTML = detail
    ? `<strong>${title}</strong><small>${detail}</small>`
    : `<strong>${title}</strong>`;
}

function hideLoadingScreen() {
  elements.message.classList.remove("is-cache-status");
  elements.message.hidden = true;
  elements.message.textContent = "";
}

function currentTomoId() {
  if (!state.selectedScan) {
    return "scan";
  }
  const parts = state.selectedScan.split("/");
  const base = parts[parts.length - 1] || "scan";
  return base.replace(/\.zarr$/i, "");
}

function gviStatusLabel(status) {
  const labels = {
    pass: "QA pass",
    warn: "QA warn",
    fail_static: "QA fail (static)",
    fail_blob: "QA fail (blob)",
    insufficient: "QA insufficient"
  };
  return labels[status] || "QA";
}

function renderGvi(gvi) {
  state.gvi = gvi || null;
  if (!gvi) {
    elements.qaStatus.hidden = true;
    elements.qaDetail.hidden = true;
    return;
  }

  elements.qaStatus.hidden = false;
  elements.qaDetail.hidden = false;
  elements.qaStatus.textContent = gviStatusLabel(gvi.status);
  elements.qaStatus.className = `status-pill qa-pill qa-${gvi.status === "fail_static" || gvi.status === "fail_blob" ? "fail" : gvi.status}`;

  const hopkins = Number.isFinite(gvi.hopkins_h) ? gvi.hopkins_h.toFixed(3) : "—";
  const knn = Number.isFinite(gvi.mean_knn_dist) ? `${Math.round(gvi.mean_knn_dist)} Å` : "—";
  elements.qaDetail.textContent = `Hopkins H ${hopkins} · ${gvi.n_particles} particles · mean kNN ${knn}. ${gvi.message || ""}`;
}

function shouldBlockInferenceForGvi() {
  const status = state.gvi?.status;
  return status === "fail_static" || status === "fail_blob";
}

async function runGviAfterPicks() {
  if (!state.detections.length) {
    renderGvi(null);
    elements.runCrowding.disabled = true;
    return;
  }

  elements.runCrowding.disabled = false;

  try {
    const payload = await fetchJson("/api/graph/gvi", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tomoId: currentTomoId(),
        detections: state.detections
      })
    });
    renderGvi(payload.gvi);
  } catch (error) {
    console.warn("GVI assessment failed:", error);
    renderGvi(null);
  }
}

function applyCrowdingResult(result) {
  if (!result?.picks?.length) {
    return;
  }

  const byId = new Map(result.picks.map((pick) => [pick.id, pick]));
  state.neighborMap.clear();
  state.detections = state.detections.map((det) => {
    const enriched = byId.get(det.id);
    if (!enriched) {
      return det;
    }
    if (Array.isArray(enriched.neighbors)) {
      state.neighborMap.set(det.id, enriched.neighbors);
    }
    return {
      ...det,
      stericExposure: enriched.stericExposure,
      gnnExposure: enriched.gnnExposure,
      neighborCount: enriched.neighborCount,
      anisotropyIndex: enriched.anisotropyIndex,
      cleanExtractionScore: enriched.cleanExtractionScore,
      openDirection: enriched.openDirection,
      cleanConeHalfAngleDeg: enriched.cleanConeHalfAngleDeg,
      nOpenComponents: enriched.nOpenComponents
    };
  });

  state.crowdingSummary = result.summary || null;
  if (result.gvi) {
    renderGvi(result.gvi);
  }

  if (state.selectedDetection) {
    const refreshed = state.detections.find((d) => d.id === state.selectedDetection.id);
    if (refreshed) {
      state.selectedDetection = refreshed;
      showPickInfo(refreshed);
      renderGraphPanel(refreshed);
    }
  } else {
    renderGraphPanel(null);
  }

  renderDetectionList();
  renderExposureSummary(result.summary);
  updateOcclusionFilterUI();
  syncPickOverlays();
  refreshSlicePreviews();
  syncViewerAnnotations();
  updateGraphPanelVisibility();
}

function renderExposureSummary(summary) {
  if (!elements.crowdingSummary) {
    return;
  }

  if (!summary) {
    elements.crowdingSummary.hidden = true;
    elements.crowdingSummary.replaceChildren();
    updateGraphPanelVisibility();
    return;
  }

  elements.crowdingSummary.hidden = false;
  const byType = summary.byType || {};
  const typeRows = Object.entries(byType)
    .sort((a, b) => (b[1].count || 0) - (a[1].count || 0))
    .map(
      ([type, stats]) => `
        <tr>
          <td>${type}</td>
          <td>${stats.count ?? 0}</td>
          <td>${stats.meanCleanExtraction?.toFixed(3) ?? stats.meanExposure?.toFixed(3) ?? "—"}</td>
          <td>${stats.meanExposure?.toFixed(3) ?? "—"}</td>
        </tr>`
    )
    .join("");

  elements.crowdingSummary.innerHTML = `
    <p class="crowding-heading">Crowding analyzed (${summary.nRays ?? 2000} rays)</p>
    <div class="crowding-stats-compact">
      <span><strong>${summary.particleCount ?? 0}</strong> particles</span>
      <span>clean ${summary.meanCleanExtraction?.toFixed(3) ?? "—"}</span>
      <span>P10 ${summary.p10CleanExtraction?.toFixed(3) ?? "—"}</span>
      <span>exp ${summary.meanExposure?.toFixed(3) ?? "—"}</span>
    </div>
    ${
      typeRows
        ? `<table class="crowding-type-table">
            <thead><tr><th>Type</th><th>Count</th><th>Mean clean</th><th>Mean exp.</th></tr></thead>
            <tbody>${typeRows}</tbody>
          </table>`
        : ""
    }
  `;

  updateGraphPanelVisibility();
}

function getDetectionPhysical(detection) {
  if (detection?.physical && Number.isFinite(detection.physical.x)) {
    return detection.physical;
  }
  const voxel = detection?.voxel;
  if (!voxel) {
    return null;
  }
  const spacing = 10;
  return {
    x: voxel.x * spacing,
    y: voxel.y * spacing,
    z: voxel.z * spacing
  };
}

function cryoDirectionToGraphVector(dx, dy, dz) {
  const v = new THREE.Vector3(dx, dz, -dy);
  return v.lengthSq() > 0 ? v.normalize() : v;
}

function graphPositionFromPhysical(physical, origin) {
  const delta = {
    x: (physical.x - origin.x) * GRAPH_SCALE,
    y: (physical.y - origin.y) * GRAPH_SCALE,
    z: (physical.z - origin.z) * GRAPH_SCALE
  };
  return cryoDirectionToGraphVector(delta.x, delta.y, delta.z).multiplyScalar(
    Math.hypot(delta.x, delta.y, delta.z)
  );
}

function graphNodeRadius(detection) {
  const radiusAngstrom = detection?.radiusAngstrom || moleculeRadiusAngstrom(detection) || 60;
  return Math.max(0.016, radiusAngstrom * GRAPH_SCALE * 0.85);
}

function graphMaterialForDetection(detection, { highlight = false } = {}) {
  const hex = detection?.color || "#0b7f83";
  const color = new THREE.Color(hex);
  return new THREE.MeshStandardMaterial({
    color,
    emissive: highlight ? color.clone().multiplyScalar(0.35) : new THREE.Color(0x000000),
    emissiveIntensity: highlight ? 0.28 : 0,
    roughness: 0.45,
    metalness: 0.05
  });
}

function updateGraphPanelVisibility() {
  const showPanel = Boolean(state.crowdingSummary);
  elements.graphPanel.hidden = !showPanel;
  elements.viewerRow?.classList.toggle("has-graph", showPanel);
  if (showPanel) {
    resizeGraphViewer();
  }
}

function setGraphCaption(text) {
  elements.graphCaption.textContent = text;
}

function frameGraphCamera(maxDistance) {
  const distance = Math.max(0.35, maxDistance * 1.75);
  graphCamera.position.set(distance * 0.85, distance * 0.62, distance * 1.05);
  graphControls.target.set(0, 0, 0);
  graphControls.update();
}

function renderGraphPanel(detection) {
  clearGroup(graphGroup);
  graphGroup.add(hemisphereGroup);

  if (!state.crowdingSummary) {
    updateGraphPanelVisibility();
    return;
  }

  updateGraphPanelVisibility();

  if (!detection) {
    elements.graphPickLabel.textContent = "Neighborhood";
    setGraphCaption("Select a pick to view its local particle graph.");
    return;
  }

  const neighbors = state.neighborMap.get(detection.id) || [];
  const centerPhys = getDetectionPhysical(detection);
  if (!centerPhys) {
    elements.graphPickLabel.textContent = detection.id;
    setGraphCaption("Pick coordinates unavailable for graph rendering.");
    return;
  }

  elements.graphPickLabel.textContent = detection.id;
  if (!neighbors.length) {
    setGraphCaption("No neighbors within cutoff for this pick.");
    return;
  }

  const cleanScore = detectionStaScore(detection);
  const openDir = detection.openDirection;
  setGraphCaption(
    `${detection.type || detection.molecule} · ${neighbors.length} neighbors${
      cleanScore !== null ? ` · clean ${cleanScore.toFixed(2)}` : ""
    }${openDir ? ` · open ${formatOpenDirectionLabel(openDir)}` : ""}`
  );

  const centerMesh = new THREE.Mesh(
    new THREE.SphereGeometry(graphNodeRadius(detection), 20, 20),
    graphMaterialForDetection(detection, { highlight: true })
  );
  graphGroup.add(centerMesh);

  const linePositions = [];
  let maxDistance = 0.2;

  neighbors.forEach((neighbor) => {
    const neighborDet = state.detections.find((d) => d.id === neighbor.id);
    const phys = neighborDet ? getDetectionPhysical(neighborDet) : neighbor.physical;
    if (!phys) {
      return;
    }

    const pos = graphPositionFromPhysical(phys, centerPhys);
    maxDistance = Math.max(maxDistance, pos.length());

    const node = new THREE.Mesh(
      new THREE.SphereGeometry(graphNodeRadius(neighborDet || detection), 16, 16),
      graphMaterialForDetection(neighborDet || { color: detection.color, radiusAngstrom: detection.radiusAngstrom })
    );
    node.position.copy(pos);
    graphGroup.add(node);

    linePositions.push(0, 0, 0, pos.x, pos.y, pos.z);
  });

  if (linePositions.length) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
    graphGroup.add(
      new THREE.LineSegments(
        geometry,
        new THREE.LineBasicMaterial({ color: 0x6a858c, transparent: true, opacity: 0.5 })
      )
    );
  }

  frameGraphCamera(maxDistance);
  resizeGraphViewer();

  const shellRadius = Math.max(0.14, maxDistance * 0.55);
  fetchHemisphereForPick(detection).then((hemisphereData) => {
    if (state.selectedDetection?.id !== detection.id) {
      return;
    }
    renderHemisphereHeatmap(hemisphereData, shellRadius);
  });
}

function resizeGraphViewer() {
  if (elements.graphPanel.hidden) {
    return;
  }

  const rect = elements.graphViewer.getBoundingClientRect();
  graphCamera.aspect = rect.width / Math.max(1, rect.height);
  graphCamera.updateProjectionMatrix();
  graphRenderer.setSize(rect.width, rect.height, false);
}

function physicalToWorld(physical) {
  const shape = state.sliceShape || state.volume?.levelShapes?.["0"] || state.volume?.shape;
  if (!shape || !physical) {
    return new THREE.Vector3();
  }

  const spacing = 10;
  const dimensions = volumeDimensions();
  const voxel = {
    x: Number(physical.x) / spacing,
    y: Number(physical.y) / spacing,
    z: Number(physical.z) / spacing
  };
  return new THREE.Vector3(
    ((voxel.x / Math.max(1, shape.x - 1)) - 0.5) * dimensions.x,
    -((voxel.y / Math.max(1, shape.y - 1)) - 0.5) * dimensions.y,
    ((voxel.z / Math.max(1, shape.z - 1)) - 0.5) * dimensions.z
  );
}

function sortedPicks(picks) {
  return [...picks].sort((a, b) => String(a.id).localeCompare(String(b.id)));
}

function resizeViewer() {
  const rect = elements.viewer.getBoundingClientRect();
  camera.aspect = rect.width / Math.max(1, rect.height);
  camera.updateProjectionMatrix();
  renderer.setSize(rect.width, rect.height, false);
  updateAnnotationPositions();
}

function detectionExposureValue(detection) {
  const value = detection?.gnnExposure ?? detection?.stericExposure;
  return Number.isFinite(value) ? value : null;
}

function detectionStaScore(detection) {
  const value = detection?.cleanExtractionScore;
  return Number.isFinite(value) ? value : null;
}

function hasExposureData() {
  return state.detections.some((det) => detectionStaScore(det) !== null);
}

function passesOcclusionFilter(detection) {
  if (state.minOcclusionFilter <= 0) {
    return true;
  }
  const score = detectionStaScore(detection);
  if (score === null) {
    return false;
  }
  return score >= state.minOcclusionFilter;
}

function visibleDetections() {
  return state.detections.filter(passesOcclusionFilter);
}

function occlusionFilterThresholdFromSlider() {
  return Number(elements.occlusionFilterSlider?.value || 0) / 100;
}

function updateOcclusionFilterUI() {
  if (!elements.occlusionFilterPanel) {
    return;
  }

  const enabled = hasExposureData();
  elements.occlusionFilterPanel.hidden = !enabled;
  if (!enabled) {
    return;
  }

  const threshold = state.minOcclusionFilter;
  if (elements.occlusionFilterValue) {
    elements.occlusionFilterValue.textContent = threshold.toFixed(2);
  }
  if (elements.occlusionFilterSlider) {
    elements.occlusionFilterSlider.value = String(Math.round(threshold * 100));
    elements.occlusionFilterSlider.setAttribute("aria-valuenow", threshold.toFixed(2));
  }

  const visibleCount = visibleDetections().length;
  const total = state.detections.length;
  if (elements.occlusionFilterStats) {
    elements.occlusionFilterStats.textContent =
      threshold > 0
        ? `${visibleCount} / ${total} picks pass filter (≥ ${threshold.toFixed(2)} clean extraction)`
        : `${total} picks visible · slide to prune low clean-extraction particles for STA`;
  }
  if (elements.exportCleanPicks) {
    elements.exportCleanPicks.disabled = visibleCount === 0;
  }
}

function applyOcclusionFilter() {
  state.minOcclusionFilter = occlusionFilterThresholdFromSlider();
  updateOcclusionFilterUI();
  renderDetectionList();
  syncPickOverlays();
  syncViewerAnnotations();
  refreshSlicePreviews();
  if (state.selectedDetection && !passesOcclusionFilter(state.selectedDetection)) {
    elements.selectedNotes.textContent = `${state.selectedDetection.notes} Filtered out by clean extraction gate (score below ${state.minOcclusionFilter.toFixed(2)}).`;
  }
}

function syncPickOverlays() {
  clearGroup(picksGroup);
  if (!state.detections.length) {
    return;
  }

  state.detections.forEach((detection) => {
    const position = detectionWorldPosition(detection);
    if (!position) {
      return;
    }

    const visible = passesOcclusionFilter(detection);
    const radius = Math.max(0.008, moleculeRadiusWorld(detection) * 0.55);
    const [r, g, b] = detectionColorArray(detection);
    const color = new THREE.Color(r / 255, g / 255, b / 255);
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 14, 14),
      new THREE.MeshStandardMaterial({
        color,
        emissive: color.clone().multiplyScalar(visible ? 0.22 : 0),
        emissiveIntensity: visible ? 0.35 : 0,
        transparent: true,
        opacity: visible ? 0.62 : 0.05,
        depthWrite: false
      })
    );
    mesh.position.copy(position);
    mesh.userData.detectionId = detection.id;
    picksGroup.add(mesh);
  });
}

function exportCleanCopickPicks() {
  const visible = visibleDetections();
  if (!visible.length) {
    showError(new Error("No picks pass the current clean extraction filter."));
    return;
  }

  const byMolecule = new Map();
  visible.forEach((detection) => {
    const molecule = detection.molecule || detection.type || "unknown";
    if (!byMolecule.has(molecule)) {
      byMolecule.set(molecule, []);
    }
    byMolecule.get(molecule).push(detection);
  });

  const files = [...byMolecule.entries()].map(([molecule, picks]) => ({
    pickable_object_name: molecule,
    user_id: "qbi",
    session_id: "0",
    run_name: currentTomoId(),
    voxel_spacing: null,
    unit: picks[0]?.physical?.unit || "angstrom",
    trust_orientation: true,
    points: picks.map((detection) => {
      const physical = getDetectionPhysical(detection);
      const openDir = detection.openDirection;
      const transform = openDir && Number.isFinite(openDir.x)
        ? rotationMatrixFromOpenDirection(openDir)
        : [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
          ];
      return {
        location: {
          x: physical?.x ?? 0,
          y: physical?.y ?? 0,
          z: physical?.z ?? 0
        },
        transformation_: transform,
        instance_id: 0,
        score: detectionStaScore(detection) ?? 1,
        open_direction: openDir || null,
        anisotropy_index: detection.anisotropyIndex ?? null,
        steric_exposure: detection.stericExposure ?? null
      };
    })
  }));

  const payload = {
    export_format: "copick_bundle",
    exported_at: new Date().toISOString(),
    min_clean_extraction_filter: state.minOcclusionFilter,
    tomo_id: currentTomoId(),
    particle_count: visible.length,
    files
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${currentTomoId()}_clean_picks_${state.minOcclusionFilter.toFixed(2)}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function rotationMatrixFromOpenDirection(openDir) {
  const zAxis = openDirectionGraphVector({ openDirection: openDir });
  const up = Math.abs(zAxis.y) < 0.99 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const xAxis = new THREE.Vector3().crossVectors(up, zAxis).normalize();
  const yAxis = new THREE.Vector3().crossVectors(zAxis, xAxis).normalize();
  return [
    [xAxis.x, yAxis.x, zAxis.x, 0],
    [xAxis.y, yAxis.y, zAxis.y, 0],
    [xAxis.z, yAxis.z, zAxis.z, 0],
    [0, 0, 0, 1]
  ];
}

function openDirectionGraphVector(detection) {
  const d = detection?.openDirection;
  if (!d || !Number.isFinite(d.x)) {
    return new THREE.Vector3(0, 1, 0);
  }
  return new THREE.Vector3(d.x, d.y, d.z).normalize();
}

function formatOpenDirectionLabel(openDir) {
  if (!openDir || !Number.isFinite(openDir.x)) {
    return "—";
  }
  return `(${openDir.x.toFixed(2)}, ${openDir.y.toFixed(2)}, ${openDir.z.toFixed(2)})`;
}

let hemisphereFetchId = 0;

async function fetchHemisphereForPick(detection) {
  if (!detection?.id || !state.detections.length) {
    return null;
  }
  const reqId = ++hemisphereFetchId;
  try {
    const payload = await fetchJson("/api/graph/hemisphere", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pickId: detection.id,
        tomoId: currentTomoId(),
        detections: state.detections
      })
    });
    if (reqId !== hemisphereFetchId) {
      return null;
    }
    return payload;
  } catch (error) {
    console.warn("Hemisphere fetch failed:", error);
    return null;
  }
}

function renderHemisphereHeatmap(hemisphereData, shellRadius) {
  clearGroup(hemisphereGroup);
  if (!hemisphereData?.directions?.length) {
    return;
  }

  const positions = [];
  const colors = [];
  hemisphereData.directions.forEach((dir, index) => {
    const open = !hemisphereData.blocked[index];
    const graphDir = cryoDirectionToGraphVector(dir[0], dir[1], dir[2]);
    positions.push(graphDir.x * shellRadius, graphDir.y * shellRadius, graphDir.z * shellRadius);
    if (open) {
      colors.push(0.35, 0.88, 0.62);
    } else {
      colors.push(0.92, 0.35, 0.35);
    }
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  hemisphereGroup.add(
    new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        size: shellRadius * 0.08,
        vertexColors: true,
        transparent: true,
        opacity: 0.85,
        depthWrite: false
      })
    )
  );
}

function exposureColorRgb(exposure) {
  const t = Math.max(0, Math.min(1, exposure ?? 0));
  // purple (buried) -> teal (exposed)
  const r = Math.round(110 + t * 60);
  const g = Math.round(70 + t * 120);
  const b = Math.round(180 - t * 70);
  return [r, g, b];
}

function detectionColorArray(detection) {
  if (state.overlayColorMode === "clean") {
    const score = detectionStaScore(detection);
    if (score !== null) {
      return exposureColorRgb(score);
    }
  }
  if (state.overlayColorMode === "exposure") {
    const exposure = detectionExposureValue(detection);
    if (exposure !== null) {
      return exposureColorRgb(exposure);
    }
  }

  const fallback = [11, 127, 131];
  const hex = String(detection.color || "").replace("#", "");
  if (!/^[0-9a-f]{6}$/i.test(hex)) {
    return fallback;
  }
  return [
    Number.parseInt(hex.slice(0, 2), 16),
    Number.parseInt(hex.slice(2, 4), 16),
    Number.parseInt(hex.slice(4, 6), 16)
  ];
}

function detectionColorNumber(detection) {
  const [r, g, b] = detectionColorArray(detection);
  return (r << 16) + (g << 8) + b;
}

function detectionColorCss(detection) {
  return `rgb(${detectionColorArray(detection).join(", ")})`;
}

function moleculeRadiusVoxels(detection) {
  if (Number.isFinite(detection?.radiusVoxel) && detection.radiusVoxel > 0) {
    return detection.radiusVoxel;
  }

  const molecule = detection?.molecule || "";
  const radii = {
    "apo-ferritin": 6,
    "beta-amylase": 7,
    "beta-galactosidase": 9,
    "ribosome": 15,
    "thyroglobulin": 13,
    "virus-like-particle": 14
  };
  return radii[molecule] || 8;
}

function moleculeRadiusAngstrom(detection) {
  if (Number.isFinite(detection?.radiusAngstrom) && detection.radiusAngstrom > 0) {
    return detection.radiusAngstrom;
  }
  return moleculeRadiusVoxels(detection) * 10;
}

function moleculeRadiusWorld(detection) {
  const dimensions = volumeDimensions();
  const shape = state.sliceShape || state.volume?.levelShapes?.["0"] || state.volume?.shape || { x: 1, y: 1, z: 1 };
  const longestAxis = Math.max(1, shape.x, shape.y, shape.z);
  const normalizedRadius = moleculeRadiusVoxels(detection) / longestAxis;
  return normalizedRadius * Math.max(dimensions.x, dimensions.y, dimensions.z);
}

function setSliceSliderValue(axis) {
  const valueElement = elements.sliceSliderValues[axis];
  if (valueElement) {
    valueElement.textContent = `${axis.toUpperCase()} ${state.currentSlices[axis]}`;
  }
}

function detectionWorldPosition(detection) {
  const shape = state.sliceShape || state.volume?.levelShapes?.["0"] || state.volume?.shape;
  const voxel = detection?.voxel;
  if (!shape || !voxel) {
    return null;
  }

  const dimensions = volumeDimensions();
  return new THREE.Vector3(
    ((voxel.x / Math.max(1, shape.x - 1)) - 0.5) * dimensions.x,
    -((voxel.y / Math.max(1, shape.y - 1)) - 0.5) * dimensions.y,
    ((voxel.z / Math.max(1, shape.z - 1)) - 0.5) * dimensions.z
  );
}

function detectionSliceWorldPosition(detection, axis) {
  const shape = state.sliceShape || state.volume?.levelShapes?.["0"] || state.volume?.shape;
  const voxel = detection?.voxel;
  if (!shape || !voxel) {
    return null;
  }

  const dimensions = volumeDimensions();
  const x = ((voxel.x / Math.max(1, shape.x - 1)) - 0.5) * dimensions.x;
  const y = (0.5 - (voxel.y / Math.max(1, shape.y - 1))) * dimensions.y;
  const z = ((voxel.z / Math.max(1, shape.z - 1)) - 0.5) * dimensions.z;

  if (axis === "x") {
    return new THREE.Vector3(sliceWorldPosition("x"), y, z);
  }
  if (axis === "y") {
    return new THREE.Vector3(x, -sliceWorldPosition("y"), z);
  }
  return new THREE.Vector3(x, y, sliceWorldPosition("z"));
}

function visibleSliceAxesForDetection(detection) {
  const voxel = detection?.voxel;
  if (!voxel) {
    return [];
  }

  return sliceAxes
    .map((axis) => ({
      axis,
      distance: Math.abs(voxel[axis] - state.currentSlices[axis])
    }))
    .filter((item) => item.distance <= moleculeRadiusVoxels(detection))
    .sort((a, b) => {
      if (a.axis === state.activeSliceAxis) {
        return -1;
      }
      if (b.axis === state.activeSliceAxis) {
        return 1;
      }
      return a.distance - b.distance;
    })
    .map((item) => item.axis);
}

function annotationWorldPosition(detection) {
  const visibleAxes = visibleSliceAxesForDetection(detection);
  if (visibleAxes.length > 0) {
    return detectionSliceWorldPosition(detection, visibleAxes[0]);
  }
  return state.selectedDetection ? detectionWorldPosition(detection) : null;
  return detectionWorldPosition(detection);
}

function projectedViewerPosition(detection) {
  const worldPosition = annotationWorldPosition(detection);
  if (!worldPosition) {
    return null;
  }

  volumeGroup.updateMatrixWorld();
  worldPosition.applyMatrix4(volumeGroup.matrixWorld);
  const projected = worldPosition.project(camera);
  if (projected.z < -1 || projected.z > 1) {
    return null;
  }

  const rect = elements.viewer.getBoundingClientRect();
  return {
    x: (projected.x * 0.5 + 0.5) * rect.width,
    y: (-projected.y * 0.5 + 0.5) * rect.height
  };
}

function selectedAnnotationDetections() {
  if (state.labelsHidden) {
    return [];
  }
  if (state.selectedDetection) {
    return [state.selectedDetection];
  }
  if (!state.selectedMolecule) {
    return [];
  }

  return state.detections
    .filter((detection) => passesOcclusionFilter(detection))
    .filter((detection) => (detection.molecule || detection.type) === state.selectedMolecule)
    .filter((detection) => visibleSliceAxesForDetection(detection).length > 0)
    .slice(0, annotationLimit);
}

function focusSlicesOnDetection(detection) {
  const voxel = detection?.voxel;
  const shape = state.sliceShape || state.volume?.shape;
  if (!voxel || !shape) {
    return;
  }

  sliceAxes.forEach((axis) => {
    const nextSlice = Math.max(0, Math.min(shape[axis] - 1, Math.round(voxel[axis])));
    state.currentSlices[axis] = nextSlice;
    if (elements.sliceSliders[axis]) {
      elements.sliceSliders[axis].value = String(nextSlice);
    }
    setSliceSliderValue(axis);
    scheduleInteractiveSlice(axis);
  });

  setActiveSliceAxis("z");
  updateSliceSeams();
  refreshSlicePreviews();
}

function syncViewerAnnotations() {
  if (!elements.annotations) {
    return;
  }

  const detections = selectedAnnotationDetections();
  elements.annotations.replaceChildren();
  annotationItems = detections.map((detection) => {
    const annotation = document.createElement("button");
    annotation.type = "button";
    annotation.className = `viewer-annotation${state.selectedDetection ? " is-single" : ""}`;
    annotation.style.setProperty("--annotation-color", detectionColorCss(detection));
    annotation.dataset.detectionId = detection.id;
    const dot = document.createElement("span");
    dot.className = "viewer-annotation-dot";
    dot.setAttribute("aria-hidden", "true");
    const text = document.createElement("span");
    text.className = "viewer-annotation-text";
    text.textContent = annotationLabel(detection);
    annotation.append(dot, text);
    annotation.addEventListener("click", () => selectDetection(detection.id));
    elements.annotations.append(annotation);
    return { detection, element: annotation };
  });

  updateAnnotationPositions();
}

function updateAnnotationPositions() {
  if (!annotationItems.length) {
    return;
  }

  annotationItems.forEach(({ detection, element }) => {
    const position = projectedViewerPosition(detection);
    if (!position) {
      element.hidden = true;
      return;
    }

    element.hidden = false;
    element.style.transform = `translate(${position.x}px, ${position.y}px)`;
  });
}

function downloadCanvasImage(canvas, filename) {
  const link = document.createElement("a");
  link.href = canvas.toDataURL("image/png");
  link.download = filename;
  link.click();
}

async function copyOrDownloadSlicePreview(axis) {
  const canvas = elements.sliceCanvases[axis];
  if (!canvas || canvas.width <= 1 || canvas.height <= 1) {
    return;
  }

  const filename = `qbi-${axis}-slice-${state.currentSlices[axis]}.png`;
  if (navigator.clipboard && window.ClipboardItem) {
    try {
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      if (blob) {
        await navigator.clipboard.write([
          new ClipboardItem({ [blob.type]: blob })
        ]);
        setStatus("Slice copied", false);
        return;
      }
    } catch (error) {
      console.warn("Copying slice preview failed; downloading instead.", error);
    }
  }

  downloadCanvasImage(canvas, filename);
  setStatus("Slice downloaded", false);
}

function setupSlicePreviewInteractions() {
  sliceAxes.forEach((axis) => {
    const canvas = elements.sliceCanvases[axis];
    const tile = canvas?.closest(".slice-preview-tile");
    if (!canvas || !tile) {
      return;
    }

    canvas.addEventListener("click", () => {
      const wasExpanded = tile.classList.contains("is-expanded");
      document.querySelectorAll(".slice-preview-tile.is-expanded").forEach((expandedTile) => {
        if (expandedTile !== tile) {
          expandedTile.classList.remove("is-expanded");
        }
      });
      tile.classList.toggle("is-expanded", !wasExpanded);
    });

    canvas.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      copyOrDownloadSlicePreview(axis).catch(showError);
    });
  });
}

function colorizedScanRgb(value) {
  const intensity = Math.max(0, Math.min(255, value)) / 255;
  return [
    Math.round(36 + intensity * 219),
    Math.round(32 + intensity * 197),
    Math.round(42 + intensity * 142)
  ];
}

function sliceSourceSize(axis) {
  const shape = state.sliceShape || state.volume?.levelShapes?.["0"] || state.volume?.shape || { x: 1, y: 1, z: 1 };
  if (axis === "x") {
    return { width: shape.y, height: shape.z, depth: shape.x };
  }
  if (axis === "y") {
    return { width: shape.x, height: shape.z, depth: shape.y };
  }
  return { width: shape.x, height: shape.y, depth: shape.z };
}

function detectionSliceProjection(detection, axis, width, height) {
  const shape = state.sliceShape || state.volume?.levelShapes?.["0"] || state.volume?.shape;
  const voxel = detection.voxel;
  if (!shape || !voxel) {
    return null;
  }

  const source = sliceSourceSize(axis);
  const position = {
    x: axis === "x" ? voxel.y : voxel.x,
    y: axis === "z" ? voxel.y : voxel.z,
    axis: voxel[axis]
  };

  return {
    x: (position.x / Math.max(1, source.width - 1)) * Math.max(0, width - 1),
    y: (position.y / Math.max(1, source.height - 1)) * Math.max(0, height - 1),
    axisPosition: position.axis,
    radius: moleculeRadiusVoxels(detection) * (width / Math.max(1, source.width)),
    axisRadius: moleculeRadiusVoxels(detection)
  };
}

function createColorizedSliceData(axis, bytes, width, height) {
  const textureData = new Uint8Array(width * height * 4);
  for (let index = 0; index < bytes.length; index += 1) {
    const [r, g, b] = colorizedScanRgb(bytes[index]);
    textureData[index * 4] = r;
    textureData[index * 4 + 1] = g;
    textureData[index * 4 + 2] = b;
    textureData[index * 4 + 3] = 255;
  }

  const sliceIndex = state.currentSlices[axis];
  state.detections.forEach((detection) => {
    if (!passesOcclusionFilter(detection)) {
      return;
    }

    const projection = detectionSliceProjection(detection, axis, width, height);
    if (!projection) {
      return;
    }

    const axisDistance = Math.abs(projection.axisPosition - sliceIndex);
    if (axisDistance > projection.axisRadius) {
      return;
    }

    const sliceRadius = Math.max(
      2,
      projection.radius * Math.sqrt(Math.max(0, 1 - (axisDistance / projection.axisRadius) ** 2))
    );
    const [overlayR, overlayG, overlayB] = detectionColorArray(detection);
    const alpha = 0.24 + 0.38 * (1 - axisDistance / projection.axisRadius);
    const minX = Math.max(0, Math.floor(projection.x - sliceRadius - 2));
    const maxX = Math.min(width - 1, Math.ceil(projection.x + sliceRadius + 2));
    const minY = Math.max(0, Math.floor(projection.y - sliceRadius - 2));
    const maxY = Math.min(height - 1, Math.ceil(projection.y + sliceRadius + 2));

    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        const distance = Math.hypot(x - projection.x, y - projection.y);
        if (distance > sliceRadius) {
          continue;
        }

        const edgeFade = 1 - Math.min(1, distance / sliceRadius);
        const mixAmount = alpha * (0.45 + edgeFade * 0.55);
        const offset = (y * width + x) * 4;
        textureData[offset] = Math.round(textureData[offset] * (1 - mixAmount) + overlayR * mixAmount);
        textureData[offset + 1] = Math.round(textureData[offset + 1] * (1 - mixAmount) + overlayG * mixAmount);
        textureData[offset + 2] = Math.round(textureData[offset + 2] * (1 - mixAmount) + overlayB * mixAmount);
      }
    }
  });

  return textureData;
}

function clearGroup(group) {
  while (group.children.length > 0) {
    const child = group.children.pop();
    disposeObject(child);
  }
}

function disposeObject(object) {
  object.children?.forEach(disposeObject);
  object.geometry?.dispose();
  if (Array.isArray(object.material)) {
    object.material.forEach(disposeMaterial);
  } else {
    disposeMaterial(object.material);
  }
}

function disposeMaterial(material) {
  if (!material) {
    return;
  }
  material.map?.dispose();
  material.uniforms?.uTexture?.value?.dispose();
  material.dispose();
}

function pointSizePixels() {
  return 5 + Number(elements.pointSize.value) * 2.7;
}

function createVolumePointMaterial() {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    vertexColors: true,
    blending: THREE.NormalBlending,
    uniforms: {
      uPointSize: { value: pointSizePixels() },
      uCameraNear: { value: camera.near },
      uCameraFar: { value: camera.far }
    },
    vertexShader: `
      attribute float strength;
      attribute float density;
      varying vec3 vColor;
      varying float vStrength;
      varying float vDensity;
      varying float vDepth;

      void main() {
        vColor = color;
        vStrength = strength;
        vDensity = density;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        float perspectiveScale = clamp(1.6 / max(0.45, -mvPosition.z), 0.58, 2.35);
        gl_PointSize = uPointSize * (0.34 + strength * 0.52) * perspectiveScale;
        vDepth = clamp((-mvPosition.z - uCameraNear) / (uCameraFar - uCameraNear), 0.0, 1.0);
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vStrength;
      varying float vDensity;
      varying float vDepth;

      void main() {
        vec2 center = gl_PointCoord - vec2(0.5);
        float radius = length(center);
        float core = 1.0 - smoothstep(0.12, 0.48, radius);
        float halo = 1.0 - smoothstep(0.2, 0.5, radius);
        if (halo <= 0.01) {
          discard;
        }

        vec3 luminous = vColor * (0.58 + vStrength * 0.95);
        luminous += vec3(0.72, 0.9, 0.96) * core * 0.18;
        luminous += vec3(1.0, 0.76, 0.46) * max(0.0, vDensity - 0.74) * 0.22;

        float alpha = (0.035 + vStrength * 0.2) * halo + core * 0.1;
        alpha *= 1.0 - smoothstep(0.76, 1.0, vDepth) * 0.28;
        gl_FragColor = vec4(luminous, clamp(alpha, 0.025, 0.38));
      }
    `
  });
}

function densityColor(strength, density) {
  const cold = new THREE.Color(0x3555a6);
  const glass = new THREE.Color(0x75d8d2);
  const warm = new THREE.Color(0xf0b45d);
  const highlight = new THREE.Color(0xfff3bf);
  const base = density < 0.52
    ? cold.clone().lerp(glass, density / 0.52)
    : glass.clone().lerp(warm, (density - 0.52) / 0.48);
  return base.lerp(highlight, Math.max(0, strength - 0.82) * 0.38);
}

function normalizedSlicePosition(axis) {
  if (!state.volume) {
    return 0;
  }
  const shape = state.sliceShape || state.volume.shape;
  return (state.currentSlices[axis] / Math.max(1, shape[axis] - 1)) - 0.5;
}

function volumeDimensions() {
  if (!state.volume?.shape) {
    return { x: 1, y: 1, z: 1 };
  }

  const { x, y, z } = state.sliceShape || state.volume.shape;
  const longestAxis = Math.max(1, x, y, z);
  return {
    x: x / longestAxis,
    y: y / longestAxis,
    z: z / longestAxis
  };
}

function sliceWorldPosition(axis) {
  const dimensions = volumeDimensions();
  return normalizedSlicePosition(axis) * dimensions[axis];
}

function createSliceGeometry(axis) {
  const dimensions = volumeDimensions();
  const geometry = new THREE.BufferGeometry();
  const halfX = dimensions.x / 2;
  const halfY = dimensions.y / 2;
  const halfZ = dimensions.z / 2;
  const uvs = [
    0, 1,
    1, 1,
    1, 0,
    0, 0
  ];
  let positions;

  if (axis === "x") {
    // Row 0 of the texture (z=0) must land at world z=-halfZ to match volume orientation.
    positions = [
      0, halfY, -halfZ,
      0, -halfY, -halfZ,
      0, -halfY, halfZ,
      0, halfY, halfZ
    ];
  } else if (axis === "y") {
    // Same z-direction fix for the y-slice.
    positions = [
      -halfX, 0, -halfZ,
      halfX, 0, -halfZ,
      halfX, 0, halfZ,
      -halfX, 0, halfZ
    ];
  } else {
    positions = [
      -halfX, halfY, 0,
      halfX, halfY, 0,
      halfX, -halfY, 0,
      -halfX, -halfY, 0
    ];
  }

  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  geometry.computeVertexNormals();
  return geometry;
}

function positionSlicePlane(axis, plane) {
  if (axis === "x") {
    plane.position.set(sliceWorldPosition("x"), 0, 0);
    return;
  }
  if (axis === "y") {
    plane.position.set(0, -sliceWorldPosition("y"), 0);
    return;
  }
  plane.position.set(0, 0, sliceWorldPosition("z"));
}

function setActiveSliceAxis(axis) {
  state.activeSliceAxis = axis;
  sliceAxes.forEach((sliceAxis) => {
    elements.sliceSliderControls[sliceAxis]?.classList.toggle("is-active", sliceAxis === axis);
  });
}

function createSliceMaterial(axis, texture) {
  const uniforms = {
    uTexture: { value: texture },
    uAxis: { value: axis === "x" ? 0 : axis === "y" ? 1 : 2 },
    uX: { value: sliceWorldPosition("x") },
    uY: { value: -sliceWorldPosition("y") },
    uZ: { value: sliceWorldPosition("z") },
    uGap: { value: 0.0035 },
    uBrightness: { value: 1 },
    uGamma: { value: 1 },
    uLineWidth: { value: 0.0045 }
  };

  return new THREE.ShaderMaterial({
    uniforms,
    transparent: false,
    depthTest: true,
    depthWrite: true,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: axis === "z" ? 0 : axis === "y" ? 1 : 2,
    polygonOffsetUnits: axis === "z" ? 0 : axis === "y" ? 1 : 2,
    vertexShader: `
      varying vec2 vUv;
      varying vec3 vWorldPosition;

      void main() {
        vUv = uv;
        vec4 worldPosition = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPosition.xyz;
        gl_Position = projectionMatrix * viewMatrix * worldPosition;
      }
    `,
    fragmentShader: `
      uniform sampler2D uTexture;
      uniform int uAxis;
      uniform float uX;
      uniform float uY;
      uniform float uZ;
      uniform float uGap;
      uniform float uBrightness;
      uniform float uGamma;
      uniform float uLineWidth;
      varying vec2 vUv;
      varying vec3 vWorldPosition;

      void main() {
        bool cut = false;
        vec3 lineColor = vec3(1.0);
        float lineMask = 0.0;
        if (uAxis == 0) {
          float yLine = 1.0 - smoothstep(uLineWidth, uLineWidth * 1.8, abs(vWorldPosition.y - uY));
          float zLine = 1.0 - smoothstep(uLineWidth, uLineWidth * 1.8, abs(vWorldPosition.z - uZ));
          cut = abs(vWorldPosition.y - uY) < uGap && abs(vWorldPosition.z - uZ) < uGap;
          lineMask = max(yLine, zLine);
          lineColor = yLine >= zLine ? vec3(0.15, 0.78, 0.22) : vec3(0.1, 0.38, 1.0);
        } else if (uAxis == 1) {
          float xLine = 1.0 - smoothstep(uLineWidth, uLineWidth * 1.8, abs(vWorldPosition.x - uX));
          float zLine = 1.0 - smoothstep(uLineWidth, uLineWidth * 1.8, abs(vWorldPosition.z - uZ));
          cut = abs(vWorldPosition.x - uX) < uGap && abs(vWorldPosition.z - uZ) < uGap;
          lineMask = max(xLine, zLine);
          lineColor = xLine >= zLine ? vec3(1.0, 0.1, 0.08) : vec3(0.1, 0.38, 1.0);
        } else {
          float xLine = 1.0 - smoothstep(uLineWidth, uLineWidth * 1.8, abs(vWorldPosition.x - uX));
          float yLine = 1.0 - smoothstep(uLineWidth, uLineWidth * 1.8, abs(vWorldPosition.y - uY));
          cut = abs(vWorldPosition.x - uX) < uGap && abs(vWorldPosition.y - uY) < uGap;
          lineMask = max(xLine, yLine);
          lineColor = xLine >= yLine ? vec3(1.0, 0.1, 0.08) : vec3(0.15, 0.78, 0.22);
        }
        if (cut) {
          discard;
        }

        vec4 color = texture2D(uTexture, vUv);
        color.rgb = pow(color.rgb, vec3(uGamma)) * uBrightness;
        color.rgb = mix(color.rgb, lineColor, clamp(lineMask * 0.85, 0.0, 1.0));
        gl_FragColor = vec4(clamp(color.rgb, 0.0, 1.0), color.a);
      }
    `
  });
}

function updateSliceSeams() {
  const seamValues = {
    x: sliceWorldPosition("x"),
    y: -sliceWorldPosition("y"),
    z: sliceWorldPosition("z")
  };

  sliceAxes.forEach((axis) => {
    const material = slicePlaneObjects[axis]?.material;
    if (!material?.uniforms) {
      return;
    }
    material.uniforms.uX.value = seamValues.x;
    material.uniforms.uY.value = seamValues.y;
    material.uniforms.uZ.value = seamValues.z;
  });
}

function updateVolumeSlicePlane(axis, bytes, width, height) {
  const textureData = createColorizedSliceData(axis, bytes, width, height);
  const texture = new THREE.DataTexture(textureData, width, height, THREE.RGBAFormat);
  texture.needsUpdate = true;
  texture.colorSpace = THREE.NoColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.flipY = true;

  if (!slicePlaneObjects[axis]) {
    const geometry = createSliceGeometry(axis);
    const material = createSliceMaterial(axis, texture);
    slicePlaneObjects[axis] = new THREE.Mesh(geometry, material);
    slicePlaneObjects[axis].name = `${axis}-tomogram-slice`;
    slicePlaneObjects[axis].renderOrder = 1;
    volumeGroup.add(slicePlaneObjects[axis]);
  } else {
    slicePlaneObjects[axis].material.uniforms.uTexture.value?.dispose();
    slicePlaneObjects[axis].material.uniforms.uTexture.value = texture;
    slicePlaneObjects[axis].material.needsUpdate = true;
  }

  positionSlicePlane(axis, slicePlaneObjects[axis]);
  updateSliceSeams();
  setActiveSliceAxis(state.activeSliceAxis);
}

function drawColorizedCanvas(axis, canvas, bytes, width, height) {
  if (!canvas) {
    return;
  }

  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const image = context.createImageData(width, height);
  image.data.set(createColorizedSliceData(axis, bytes, width, height));
  context.putImageData(image, 0, 0);
}

function drawMainSlice(axis, bytes, width, height, renderedLevel, renderedIndex) {
  drawColorizedCanvas(axis, mainSliceCanvas, bytes, width, height);
  elements.message.hidden = true;
}

function drawSlicePreview(axis, bytes, width, height, renderedLevel, renderedIndex) {
  const canvas = elements.sliceCanvases[axis];
  if (!canvas) {
    return;
  }

  drawColorizedCanvas(axis, canvas, bytes, width, height);
  if (elements.sliceLabels[axis]) {
    elements.sliceLabels[axis].textContent = `${axis.toUpperCase()} ${state.currentSlices[axis]} | L${renderedLevel} ${renderedIndex}`;
  }
}

function refreshSlicePreviews() {
  sliceAxes.forEach((axis) => {
    const cached = getCachedSlice(axis, state.currentSlices[axis], highDetailSliceMaxSize)
      || getCachedSlice(axis, state.currentSlices[axis], interactiveSliceMaxSize);
    if (!cached) {
      return;
    }

    drawSlicePreview(axis, cached.bytes, cached.width, cached.height, cached.renderedLevel, cached.renderedIndex);
  });
}

function renderVolumeScene(payload) {
  clearGroup(volumeGroup);
  sliceAxes.forEach((axis) => {
    slicePlaneObjects[axis] = null;
  });

  const dimensions = volumeDimensions();
  const box = new THREE.BoxGeometry(dimensions.x, dimensions.y, dimensions.z);
  const edges = new THREE.EdgesGeometry(box);
  const frame = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({ color: 0x71818a, transparent: true, opacity: 0.34 })
  );
  frame.name = "volume-bounds";
  volumeGroup.add(frame);

  elements.message.hidden = true;
}

function renderDetectionList() {
  elements.detectionList.replaceChildren();
  const groups = getMoleculeGroups();
  const totalPicks = state.detections.length;
  const visibleCount = visibleDetections().length;
  const filterActive = state.minOcclusionFilter > 0 && hasExposureData();
  elements.detectionCount.textContent = filterActive
    ? `${groups.length} type${groups.length !== 1 ? "s" : ""} · ${visibleCount}/${totalPicks} picks`
    : `${groups.length} type${groups.length !== 1 ? "s" : ""} · ${totalPicks} picks`;
  elements.showAllBtn.hidden = state.selectedMolecule === null;
  updateOcclusionFilterUI();
  elements.showAllBtn.textContent = state.labelsHidden ? "Show all" : "Hide all";

  groups.forEach((group) => {
    const isSelected = state.selectedMolecule === group.molecule;
    const isExpanded = state.expandedMolecules.has(group.molecule);
    const groupColor = detectionColorCss(group.picks[0] || { color: "#0b7f83" });

    const groupEl = document.createElement("div");
    groupEl.className = "molecule-group";

    const header = document.createElement("button");
    header.className = `molecule-group-header${isSelected ? " is-selected" : ""}`;
    header.type = "button";
    header.innerHTML = `
      <span class="dot" style="background:${groupColor}"></span>
      <span class="molecule-name">${group.label}</span>
      <span class="pick-badge">${group.picks.length}</span>
      <span class="expand-icon">${isExpanded ? "▾" : "▸"}</span>
    `;
    header.addEventListener("click", () => {
      const alreadySelected = state.selectedMolecule === group.molecule;
      selectMoleculeGroup(group.molecule);
      if (alreadySelected) {
        if (state.expandedMolecules.has(group.molecule)) {
          state.expandedMolecules.delete(group.molecule);
        } else {
          state.expandedMolecules.add(group.molecule);
        }
      } else {
        state.expandedMolecules.add(group.molecule);
      }
      renderDetectionList();
    });
    groupEl.append(header);

    if (isExpanded) {
      const pickList = document.createElement("div");
      pickList.className = "molecule-pick-list";
      sortedPicks(group.picks).forEach((det) => {
        const row = document.createElement("button");
        const filteredOut = !passesOcclusionFilter(det);
        row.className = `molecule-pick-row${det.id === state.selectedDetection?.id ? " is-selected" : ""}${filteredOut ? " is-filtered" : ""}`;
        row.type = "button";
        row.dataset.id = det.id;
        const dot = document.createElement("span");
        dot.className = "dot molecule-pick-dot";
        dot.style.background = detectionColorCss(det);
        const label = document.createElement("span");
        label.textContent = det.id;
        row.append(dot, label);
        const staScore = detectionStaScore(det);
        if (staScore !== null) {
          const badge = document.createElement("span");
          badge.className = `exposure-badge${filteredOut ? " is-below-threshold" : ""}`;
          badge.textContent = staScore.toFixed(2);
          row.append(badge);
        }
        row.addEventListener("click", (e) => {
          e.stopPropagation();
          selectDetection(det.id);
        });
        pickList.append(row);
      });
      groupEl.append(pickList);
    }

    elements.detectionList.append(groupEl);
  });
}

function hasAnalysisContent(analysis) {
  if (!analysis) {
    return false;
  }
  const structured = analysis.structured;
  const items = analysis.aggregation?.items;
  const report = analysis.report || analysis.localSummary;
  return Boolean(
    structured ||
    (Array.isArray(items) && items.length > 0) ||
    report ||
    analysis.reportError ||
    analysis.reportErrorBody
  );
}

function renderAnalysis(analysis) {
  if (hasAnalysisContent(analysis)) {
    state.datasetAnalysis = analysis;
  }

  const payload = state.datasetAnalysis;
  if (!hasAnalysisContent(payload)) {
    elements.analysisStatus.textContent = "Waiting for labels";
    elements.analysisSummary.replaceChildren();
    elements.analysisReport.textContent = "Upload a Picks labels folder to generate molecule-level insights and a structured interpretation.";
    return;
  }

  elements.analysisSummary.replaceChildren();

  const structured = payload.structured || null;
  const fallbackItems = payload.aggregation?.items || [];
  const cards = Array.isArray(structured?.insights) && structured.insights.length > 0
    ? structured.insights
    : fallbackItems.map((item) => ({
        tone: item.difficulty === "easy" ? "positive" : item.difficulty.includes("hard") ? "warning" : "analytical",
        label: `${item.label}: ${item.count} picks · ${item.frequencyPercent.toFixed(1)}%`,
        text: `${item.clusterCount} clusters, ${item.singletonClusters} singletons, largest cluster: ${item.largestClusterSize}.`,
        evidence: [
          `${item.count} picks`,
          `${item.frequencyPercent.toFixed(1)}%`,
          `${item.clusterCount} clusters`
        ]
      }));

  if (!cards.length && !structured && !(payload.report || payload.localSummary) && !payload.reportError) {
    return;
  }

  elements.analysisStatus.textContent = payload.reportStatus || analysis?.reportStatus || "Ready";
  elements.analysisStatus.textContent = analysis.reportStatus || "Ready";
  elements.analysisStatus.classList.remove("active");

  // Keywords pills
  const keywords = structured?.keywords || [];
  if (keywords.length > 0) {
    const kwRow = document.createElement("div");
    kwRow.className = "analysis-keywords";
    keywords.forEach((word) => {
      const tag = document.createElement("span");
      tag.className = "keyword-tag";
      tag.textContent = word;
      kwRow.append(tag);
    });
    elements.analysisSummary.append(kwRow);
  }

  // Insight cards
  cards.forEach((item) => {
    const card = document.createElement("div");
    const tone = ["positive", "analytical", "warning", "serious"].includes(item.tone) ? item.tone : "analytical";
    card.className = `analysis-card tone-${tone}`;
    const evidence = Array.isArray(item.evidence) ? item.evidence : [];
    card.innerHTML = `
      <div class="analysis-card-head">
        <strong>${item.label || "Insight"}</strong>
      </div>
      <p>${item.text || ""}</p>
      ${evidence.length > 0 ? `<small>${evidence.join(" · ")}</small>` : ""}
    `;
    elements.analysisSummary.append(card);
  });

  // Spatial distribution section
  if (structured?.spatialAnalysis) {
    const section = document.createElement("div");
    section.className = "analysis-section";
    const heading = document.createElement("h4");
    heading.className = "analysis-section-heading";
    heading.textContent = "Spatial Distribution";
    const body = document.createElement("p");
    body.textContent = structured.spatialAnalysis;
    section.append(heading, body);
    elements.analysisSummary.append(section);
  }

  // Biological context section
  if (structured?.biologicalContext) {
    const section = document.createElement("div");
    section.className = "analysis-section";
    const heading = document.createElement("h4");
    heading.className = "analysis-section-heading";
    heading.textContent = "Biological Context";
    const body = document.createElement("p");
    body.textContent = structured.biologicalContext;
    section.append(heading, body);
    elements.analysisSummary.append(section);
  }

  const report = payload.report || payload.localSummary || analysis?.report || analysis?.localSummary || "";
  // Report panel
  const report = analysis.report || analysis.localSummary || "";
  elements.analysisReport.replaceChildren();
  if (structured?.title) {
    const heading = document.createElement("h3");
    heading.textContent = structured.title;
    elements.analysisReport.append(heading);
  }
  if (structured?.headline) {
    const paragraph = document.createElement("p");
    paragraph.textContent = structured.headline;
    elements.analysisReport.append(paragraph);
  }
  if (structured?.datasetSummary) {
    const heading = document.createElement("h3");
    heading.textContent = "Dataset Composition";
    const paragraph = document.createElement("p");
    paragraph.textContent = structured.datasetSummary;
    elements.analysisReport.append(heading, paragraph);
  }
  if (structured?.caveats?.length) {
    const heading = document.createElement("h3");
    heading.textContent = "Caveats";
    elements.analysisReport.append(heading);
    structured.caveats.forEach((cav) => {
      const p = document.createElement("p");
      p.textContent = cav;
      elements.analysisReport.append(p);
    });
  }
  if (structured?.nextSteps?.length) {
    const heading = document.createElement("h3");
    heading.textContent = "Next Validation Steps";
    elements.analysisReport.append(heading);
    structured.nextSteps.forEach((step) => {
      const p = document.createElement("p");
      p.textContent = step;
      elements.analysisReport.append(p);
    });
  }
  if (!structured && report) {
    report.split(/\n{2,}/).forEach((block) => {
      const text = block.trim();
      if (!text) {
        return;
      }
      if (/^#{1,3}\s+/.test(text)) {
        const heading = document.createElement("h3");
        heading.textContent = text.replace(/^#{1,3}\s+/, "");
        elements.analysisReport.append(heading);
        return;
      }
      const paragraph = document.createElement("p");
      paragraph.textContent = text.replace(/\*\*/g, "");
      elements.analysisReport.append(paragraph);
    });
  }
  const reportWarning = payload.reportWarning || analysis?.reportWarning;
  if (reportWarning) {
    const warningEl = document.createElement("p");
    warningEl.className = "analysis-warning";
    warningEl.textContent = `⚠ ${reportWarning}`;
    elements.analysisReport.append(warningEl);
  }

  const reportError = payload.reportError || analysis?.reportError;
  const reportErrorBody = payload.reportErrorBody || analysis?.reportErrorBody;
  if (reportError || reportErrorBody) {
  if (analysis.reportWarning) {
    const warningEl = document.createElement("p");
    warningEl.className = "analysis-warning";
    warningEl.textContent = `⚠ ${analysis.reportWarning}`;
    elements.analysisReport.append(warningEl);
  }
  if (analysis.reportError || analysis.reportErrorBody) {
    const heading = document.createElement("h3");
    heading.textContent = "Analysis Error";
    const errorBlock = document.createElement("pre");
    errorBlock.className = "analysis-error";
    errorBlock.textContent = [reportError, reportErrorBody].filter(Boolean).join("\n\n");
    errorBlock.textContent = [analysis.reportError, analysis.reportErrorBody].filter(Boolean).join("\n\n");
    elements.analysisReport.append(heading, errorBlock);
  }
}

function setInfoLabel(ddElement, labelText) {
  if (ddElement.previousElementSibling?.tagName === "DT") {
    ddElement.previousElementSibling.textContent = labelText;
  }
}

function showPickInfo(detection) {
  setInfoLabel(elements.selectedId, "ID");
  setInfoLabel(elements.selectedType, "Type");
  setInfoLabel(elements.selectedConfidence, "Confidence");
  setInfoLabel(elements.selectedPosition, "Position");
  setInfoLabel(elements.selectedExposure, "Steric exposure");
  setInfoLabel(elements.selectedCleanExtraction, "Clean extraction");
  setInfoLabel(elements.selectedAnisotropy, "Anisotropy");
  setInfoLabel(elements.selectedOpenDirection, "Open direction");
  setInfoLabel(elements.selectedGnnExposure, "GNN exposure");
  setInfoLabel(elements.selectedNeighbors, "Neighbors");
  setInfoLabel(elements.selectedNotes, "Notes");
  elements.selectedId.textContent = detection.id;
  elements.selectedType.textContent = detection.type;
  elements.selectedConfidence.textContent = detection.confidence;
  elements.selectedPosition.textContent = detection.position;
  elements.selectedExposure.textContent = Number.isFinite(detection.stericExposure)
    ? detection.stericExposure.toFixed(3)
    : "Run crowding analysis";
  elements.selectedCleanExtraction.textContent = Number.isFinite(detection.cleanExtractionScore)
    ? detection.cleanExtractionScore.toFixed(3)
    : "Run crowding analysis";
  elements.selectedAnisotropy.textContent = Number.isFinite(detection.anisotropyIndex)
    ? detection.anisotropyIndex.toFixed(3)
    : "Run crowding analysis";
  elements.selectedOpenDirection.textContent = detection.openDirection
    ? formatOpenDirectionLabel(detection.openDirection)
    : "Run crowding analysis";
  elements.selectedGnnExposure.textContent = Number.isFinite(detection.gnnExposure)
    ? detection.gnnExposure.toFixed(3)
    : "Run crowding analysis";
  elements.selectedNeighbors.textContent = Number.isFinite(detection.neighborCount)
    ? `${detection.neighborCount} within cutoff`
    : state.neighborMap.get(detection.id)?.length
      ? `${state.neighborMap.get(detection.id).length} kNN`
      : "Run crowding analysis";
  elements.selectedNotes.textContent = `${detection.notes} Render radius ${Math.round(moleculeRadiusAngstrom(detection))} angstrom.`;
}

function showMoleculeGroupInfo(group) {
  setInfoLabel(elements.selectedId, "Name");
  setInfoLabel(elements.selectedType, "Source");
  setInfoLabel(elements.selectedConfidence, "Picks");
  setInfoLabel(elements.selectedPosition, "Unit");
  setInfoLabel(elements.selectedNotes, "Notes");
  elements.selectedId.textContent = group.label;
  elements.selectedType.textContent = `${group.molecule}.json`;
  elements.selectedConfidence.textContent = `${group.picks.length} curated picks`;
  elements.selectedPosition.textContent = group.picks[0]?.physical?.unit || "angstrom";
  elements.selectedNotes.textContent = `${group.label} curated overlay picks from the TS_5_4 experiment run.`;
}

function selectMoleculeGroup(molecule) {
  state.selectedMolecule = molecule;
  state.selectedDetection = null;
  state.labelsHidden = false;

  const group = getMoleculeGroups().find((g) => g.molecule === molecule);
  if (group) {
    showMoleculeGroupInfo(group);
  }
  renderGraphPanel(null);
  syncViewerAnnotations();
}

function clearMoleculeSelection() {
  state.selectedDetection = null;
  renderGraphPanel(null);
  state.labelsHidden = !state.labelsHidden;
  syncViewerAnnotations();
  renderDetectionList();
}

function selectDetection(id) {
  const detection = state.detections.find((item) => item.id === id);
  if (!detection) {
    return;
  }

  state.selectedDetection = detection;
  state.selectedMolecule = detection.molecule || detection.type;
  state.labelsHidden = false;
  state.expandedMolecules.add(state.selectedMolecule);

  focusSlicesOnDetection(detection);
  showPickInfo(detection);
  renderGraphPanel(detection);
  syncViewerAnnotations();

  if (detection.voxel && state.sliceShape) {
    state.currentSlices.x = Math.max(0, Math.min(state.sliceShape.x - 1, Math.round(detection.voxel.x)));
    state.currentSlices.y = Math.max(0, Math.min(state.sliceShape.y - 1, Math.round(detection.voxel.y)));
    state.currentSlices.z = Math.max(0, Math.min(state.sliceShape.z - 1, Math.round(detection.voxel.z)));
    sliceAxes.forEach((axis) => {
      elements.sliceSliders[axis].value = String(state.currentSlices[axis]);
      elements.sliceSliderValues[axis].textContent = `${axis.toUpperCase()} ${state.currentSlices[axis]}`;
    });
    refreshSlicePreviews();
  }

  renderDetectionList();
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || response.statusText);
  }
  return response.json();
}

function sliceCacheKey(axis, index, maxSize = highDetailSliceMaxSize) {
  return [
    state.selectedScan,
    state.sliceLevel || elements.levelSelect.value,
    axis,
    index,
    maxSize
  ].join("|");
}

function getCachedSlice(axis, index, maxSize = highDetailSliceMaxSize) {
  const key = sliceCacheKey(axis, index, maxSize);
  const cached = sliceCache.get(key);
  if (!cached) {
    return null;
  }
  sliceCache.delete(key);
  sliceCache.set(key, cached);
  return cached;
}

function setCachedSlice(axis, index, maxSize, value) {
  const key = sliceCacheKey(axis, index, maxSize);
  if (sliceCache.has(key)) {
    sliceCache.delete(key);
  }
  sliceCache.set(key, value);
}

async function loadScans() {
  setStatus("Finding scans", true);
  const payload = await fetchJson("/api/scans");
  state.scans = [...payload.scans].sort((a, b) => {
    if (Boolean(a.metadataMissing) === Boolean(b.metadataMissing)) {
      return 0;
    }
    return a.metadataMissing ? 1 : -1;
  });
  elements.scanSelect.replaceChildren(
    ...state.scans.map((scan) => {
      const option = document.createElement("option");
      option.value = scan.path;
      option.textContent = scan.metadataMissing ? `${scan.name} (metadata missing)` : scan.name;
      option.disabled = scan.metadataMissing;
      return option;
    })
  );

  const firstUsableScan = state.scans.find((scan) => !scan.metadataMissing);
  if (!firstUsableScan) {
    state.selectedScan = null;
    setStatus("Open a local .zarr path", false);
    setProgress(0);
    elements.message.hidden = false;
    elements.message.textContent = "No readable Zarr scans found. Browser folder upload often skips hidden .zarray files — use the Local .zarr path field with the original folder.";
    return false;
  }

  state.selectedScan = firstUsableScan.path;
  elements.scanSelect.value = state.selectedScan;
  syncLevelSelect();
  return true;
}

function selectedScan() {
  return state.scans.find((scan) => scan.path === state.selectedScan);
}

function syncLevelSelect() {
  const scan = selectedScan();
  const levels = scan?.levels?.length ? scan.levels : ["2", "1", "0"];
  elements.levelSelect.replaceChildren(
    ...levels.map((level) => {
      const option = document.createElement("option");
      option.value = level;
      option.textContent = `Level ${level}${level === scan?.defaultLevel ? " - default" : ""}`;
      return option;
    })
  );
  elements.levelSelect.value = levels.includes(scan?.defaultLevel) ? scan.defaultLevel : levels[0];
}

async function loadPreview() {
  if (!state.selectedScan) {
    return;
  }

  setStatus("Decoding", true);
  setProgress(22);
  elements.message.classList.remove("is-cache-status");
  elements.message.hidden = false;
  elements.message.textContent = "Decoding Zarr preview";

  const params = new URLSearchParams({
    path: state.selectedScan,
    level: elements.levelSelect.value,
    stride: elements.strideSelect.value,
    limit: elements.pointLimit.value
  });
  const payload = await fetchJson(`/api/zarr/preview?${params.toString()}`);

  setProgress(76);
  if (payload.level && payload.level !== elements.levelSelect.value) {
    elements.levelSelect.value = payload.level;
  }
  state.volume = payload;
  state.sliceLevel = payload.levelShapes?.[highDetailSliceLevel] ? highDetailSliceLevel : payload.level;
  state.sliceShape = payload.levelShapes?.[state.sliceLevel] || payload.shape;
  sliceCache.clear();
  state.detections = payload.detections;
  refreshAnnotationNumbers();
  state.selectedDetection = null;
  state.selectedMolecule = null;
  state.labelsHidden = false;
  state.expandedMolecules.clear();
  state.neighborMap.clear();
  state.crowdingSummary = null;
  state.datasetAnalysis = null;
  renderExposureSummary(null);
  renderGraphPanel(null);
  renderGvi(null);
  state.minOcclusionFilter = 0;
  updateOcclusionFilterUI();
  state.currentSlices = {
    x: Math.floor(state.sliceShape.x / 2),
    y: Math.floor(state.sliceShape.y / 2),
    z: Math.floor(state.sliceShape.z / 2)
  };
  state.activeSliceAxis = "z";
  sliceAxes.forEach((axis) => {
    elements.sliceSliders[axis].max = String(Math.max(0, state.sliceShape[axis] - 1));
    elements.sliceSliders[axis].value = String(state.currentSlices[axis]);
    setSliceSliderValue(axis);
  });

  renderVolumeScene(payload);
  const initialGroups = getMoleculeGroups();
  if (initialGroups.length > 0) {
    state.expandedMolecules.add(initialGroups[0].molecule);
  }
  syncViewerAnnotations();
  syncPickOverlays();
  renderDetectionList();
  renderAnalysis(payload.analysis);
  if (state.detections.length > 0) {
    fetchClaudeAnalysis(state.selectedScan).catch(() => {});
  }
  await Promise.all(sliceAxes.map((axis) => loadSlice(axis)));
  hideLoadingScreen();
  elements.message.hidden = true;
  startSlicePrecache("Caching scan slices");
  runGviAfterPicks().catch((error) => console.warn("GVI assessment failed:", error));
  setActiveSliceAxis("z");
  camera.position.set(1.55, 1.25, 1.65);
  controls.target.set(0, 0, 0);
  controls.update();

  setProgress(100);
  setStatus("Ready", false);
}

function startSlicePrecache(reason = "Caching slices") {
  precacheAllSlices(reason).catch((error) => {
    console.warn("Slice cache failed", error);
    if (elements.message.classList.contains("is-cache-status")) {
      hideLoadingScreen();
    }
  });
}

async function precacheAllSlices(reason = "Caching slices") {
  if (!state.volume || !state.sliceShape) {
    return;
  }

  const runId = precacheRunId + 1;
  precacheRunId = runId;
  const jobs = [];
  const axisIndices = Object.fromEntries(sliceAxes.map((axis) => {
    const count = Math.max(0, state.sliceShape[axis]);
    const center = Math.max(0, Math.min(count - 1, state.currentSlices[axis] || 0));
    const indices = [];
    for (let offset = 0; indices.length < count; offset += 1) {
      const lower = center - offset;
      const upper = center + offset;
      if (lower >= 0) {
        indices.push(lower);
      }
      if (offset > 0 && upper < count) {
        indices.push(upper);
      }
    }
    return [axis, indices];
  }));
  const maxAxisCount = Math.max(...sliceAxes.map((axis) => axisIndices[axis].length));
  for (let offset = 0; offset < maxAxisCount; offset += 1) {
    sliceAxes.forEach((axis) => {
      const index = axisIndices[axis][offset];
      if (Number.isInteger(index)) {
        jobs.push({ axis, index });
      }
    });
  }

  if (jobs.length === 0) {
    return;
  }

  let completed = 0;
  setStatus(reason, true);
  setProgress(0);
  showLoadingScreen(reason, `Preparing ${jobs.length.toLocaleString()} X/Y/Z slices`);

  const batches = [];
  for (let start = 0; start < jobs.length; start += slicePrecacheBatchSize) {
    batches.push(jobs.slice(start, start + slicePrecacheBatchSize));
  }

  for (const [batchIndex, batch] of batches.entries()) {
    if (runId !== precacheRunId) {
      hideLoadingScreen();
      return;
    }

    await Promise.all(batch.map(async (job) => {
      try {
        await loadSlice(job.axis, {
          index: job.index,
          maxSize: highDetailSliceMaxSize,
          render: false,
          prefetch: true
        });
      } catch (error) {
        console.warn(`Could not cache ${job.axis} slice ${job.index}`, error);
      }
    }));

    completed += batch.length;
    const percent = Math.round((completed / jobs.length) * 100);
    setProgress(percent);
    showLoadingScreen(
      reason,
      `Batch ${batchIndex + 1} / ${batches.length} · ${completed.toLocaleString()} / ${jobs.length.toLocaleString()} slices cached`
    );
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }

  if (runId !== precacheRunId) {
    hideLoadingScreen();
    return;
  }

  setProgress(100);
  setStatus("Ready", false);
  hideLoadingScreen();
}

function prefetchNeighborSlices(axis) {
  clearTimeout(prefetchTimers[axis]);
  prefetchTimers[axis] = setTimeout(() => {
    if (!state.volume) {
      return;
    }
    const maxIndex = Math.max(0, state.sliceShape[axis] - 1);
    const center = state.currentSlices[axis];
    [-2, -1, 1, 2].forEach((offset) => {
      const index = center + offset;
      if (index < 0 || index > maxIndex) {
        return;
      }
      loadSlice(axis, {
        index,
        maxSize: highDetailSliceMaxSize,
        render: false,
        prefetch: true
      }).catch(() => {});
    });
  }, 80);
}

function scheduleInteractiveSlice(axis) {
  clearTimeout(sliceTimers[axis]);
  const cachedHighDetail = getCachedSlice(axis, state.currentSlices[axis], highDetailSliceMaxSize);
  if (cachedHighDetail) {
    renderSlice(axis, cachedHighDetail);
    prefetchNeighborSlices(axis);
    return;
  }

  const requestedIndex = state.currentSlices[axis];
  setStatus("Loading slice", true);
  loadSlice(axis, {
    index: requestedIndex,
    maxSize: highDetailSliceMaxSize
  })
    .then(() => {
      setStatus("Ready", false);
      prefetchNeighborSlices(axis);
    })
    .catch(showError);
}

function renderSlice(axis, slice) {
  updateVolumeSlicePlane(axis, slice.bytes, slice.width, slice.height);
  elements.sliceSliders[axis].value = String(state.currentSlices[axis]);
  setSliceSliderValue(axis);
  drawSlicePreview(axis, slice.bytes, slice.width, slice.height, slice.renderedLevel, slice.renderedIndex);
}

async function loadSlice(axis = "z", options = {}) {
  if (!state.volume) {
    return;
  }

  const {
    maxSize = highDetailSliceMaxSize,
    render = true,
    prefetch = false,
    index = state.currentSlices[axis]
  } = options;
  const requestedSlice = index;
  const cached = getCachedSlice(axis, requestedSlice, maxSize);
  if (cached) {
    if (render) {
      renderSlice(axis, cached);
    }
    return cached;
  }

  const requestId = prefetch ? sliceRequestIds[axis] : sliceRequestIds[axis] + 1;
  if (!prefetch) {
    sliceRequestIds[axis] = requestId;
  }

  const previewLevel = elements.levelSelect.value;
  const detailLevel = state.sliceLevel || previewLevel;
  const params = new URLSearchParams({
    path: state.selectedScan,
    axis,
    level: detailLevel,
    maxSize,
    index: String(requestedSlice)
  });
  let response = await fetch(`/api/zarr/slice?${params.toString()}`);
  if (!response.ok && detailLevel !== previewLevel) {
    params.set("level", previewLevel);
    params.set("sourceLevel", detailLevel);
    response = await fetch(`/api/zarr/slice?${params.toString()}`);
  }
  if (!response.ok) {
    return;
  }

  const bytes = new Uint8ClampedArray(await response.arrayBuffer());
  if (!prefetch && (requestId !== sliceRequestIds[axis] || requestedSlice !== state.currentSlices[axis])) {
    return null;
  }

  const width = Number(response.headers.get("X-QBI-Slice-Width")) || state.volume.shape.x;
  const height = Number(response.headers.get("X-QBI-Slice-Height")) || state.volume.shape.y;
  const renderedLevel = response.headers.get("X-QBI-Slice-Level") || previewLevel;
  const renderedIndex = response.headers.get("X-QBI-Slice-Index") || String(requestedSlice);
  const slice = { bytes, width, height, renderedLevel, renderedIndex };
  setCachedSlice(axis, requestedSlice, maxSize, slice);
  if (render) {
    renderSlice(axis, slice);
  }
  return slice;
}

async function uploadZarrFolder() {
  const files = Array.from(elements.upload.files || []);
  if (files.length === 0) {
    return;
  }

  setStatus("Uploading", true);
  setProgress(12);
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("files", file, file.webkitRelativePath || file.name);
  });

  const response = await fetch("/api/upload-zarr", {
    method: "POST",
    body: formData
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Upload failed");
  }

  await loadScans();
  const uploaded = payload.scans?.find((scan) => !scan.metadataMissing);
  if (uploaded?.path) {
    state.selectedScan = uploaded.path;
    elements.scanSelect.value = state.selectedScan;
    syncLevelSelect();
  } else if (!state.selectedScan) {
    throw new Error(payload.error || "Uploaded .zarr is missing metadata. Use the Local .zarr path field instead.");
  }
  await loadPreview();
}

async function uploadLabelsFolder() {
  const files = Array.from(elements.labelsUpload.files || []);
  if (files.length === 0 || !state.selectedScan || !state.volume) {
    return;
  }

  setStatus("Uploading labels", true);
  setProgress(18);
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("files", file, file.webkitRelativePath || file.name);
  });

  const params = new URLSearchParams({ path: state.selectedScan });
  const response = await fetch(`/api/upload-picks?${params.toString()}`, {
    method: "POST",
    body: formData
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Label upload failed");
  }

  state.detections = payload.detections || [];
  refreshAnnotationNumbers();
  state.selectedDetection = null;
  state.selectedMolecule = null;
  state.labelsHidden = false;
  state.expandedMolecules.clear();
  state.neighborMap.clear();
  state.crowdingSummary = null;
  state.minOcclusionFilter = 0;
  updateOcclusionFilterUI();
  renderExposureSummary(null);
  renderGraphPanel(null);
  const initialGroups = getMoleculeGroups();
  if (initialGroups.length > 0) {
    state.expandedMolecules.add(initialGroups[0].molecule);
  }

  sliceCache.clear();
  renderVolumeScene(state.volume);
  syncViewerAnnotations();
  syncPickOverlays();
  renderDetectionList();
  renderAnalysis(payload.analysis);
  fetchClaudeAnalysis(state.selectedScan).catch(() => {});
  await Promise.all(sliceAxes.map((axis) => loadSlice(axis)));
  startSlicePrecache("Caching label overlays");

  elements.detectionCount.textContent = `${initialGroups.length} type${initialGroups.length !== 1 ? "s" : ""} · ${state.detections.length} picks`;
  setProgress(100);
  setStatus(`Loaded ${payload.jsonFiles} label files`, false);
  await runGviAfterPicks();
}

async function openLocalZarrPath() {
  const localPath = elements.localZarrPath.value.trim();
  if (!localPath) {
    return;
  }

  setStatus("Opening local Zarr", true);
  const payload = await fetchJson("/api/local-zarr", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: localPath })
  });

  state.scans = [...state.scans.filter((scan) => scan.path !== payload.scan.path), payload.scan];
  state.selectedScan = payload.scan.path;
  elements.scanSelect.replaceChildren(
    ...state.scans.map((scan) => {
      const option = document.createElement("option");
      option.value = scan.path;
      option.textContent = scan.name;
      option.disabled = scan.metadataMissing;
      return option;
    })
  );
  elements.scanSelect.value = state.selectedScan;
  syncLevelSelect();
  await loadPreview();
}

async function openSelectedScanInSlicer() {
  if (!state.selectedScan) {
    return;
  }

  setStatus("Exporting for Slicer", true);
  const params = new URLSearchParams({
    path: state.selectedScan,
    level: elements.levelSelect.value
  });
  const payload = await fetchJson(`/api/slicer/open?${params.toString()}`, {
    method: "POST"
  });

  if (payload.opened) {
    setStatus("Opened in Slicer", false);
    return;
  }

  setStatus("Download Slicer or NRRD", false);
  const shouldDownload = window.confirm(
    "3D Slicer was not found on this machine. Download the exported NRRD file instead?"
  );
  if (shouldDownload) {
    window.location.href = payload.exportUrl;
  } else {
    window.open(payload.downloadUrl, "_blank", "noopener");
  }
}

async function runCrowdingAnalysis() {
  if (!state.detections.length) {
    showError(new Error("Load or infer picks before running crowding analysis."));
    return;
  }

  elements.runCrowding.disabled = true;
  elements.runCrowding.classList.add("is-running");
  elements.runCrowding.innerHTML = '<span class="inference-icon">⏳</span> Analyzing…';
  elements.crowdingProgress.hidden = false;
  elements.crowdingProgressFill.style.width = "0%";
  elements.crowdingStatus.textContent = "Starting crowding analysis…";

  const jobId = `crowd-${Date.now()}`;

  try {
    const response = await fetch("/api/graph/crowding/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jobId,
        tomoId: currentTomoId(),
        detections: state.detections
      })
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Failed to start crowding analysis");
    }

    const eventSource = new EventSource(`/api/graph/crowding/progress?jobId=${jobId}`);

    eventSource.addEventListener("progress", (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.percent >= 0) {
          elements.crowdingProgressFill.style.width = `${data.percent}%`;
        }
        if (data.message) {
          elements.crowdingStatus.textContent = data.message;
        }
      } catch {
        // ignore
      }
    });

    eventSource.addEventListener("complete", (event) => {
      try {
        const data = JSON.parse(event.data);
        applyCrowdingResult(data);
        elements.crowdingStatus.textContent = "✓ Crowding analysis complete";
        elements.crowdingProgressFill.style.width = "100%";
        elements.crowdingProgressFill.style.background = "#16743a";
      } catch (error) {
        showError(new Error("Failed to parse crowding results"));
      }
      resetCrowdingButton();
      eventSource.close();
    });

    eventSource.addEventListener("error", (event) => {
      let errorMsg = "Crowding analysis failed";
      try {
        const data = JSON.parse(event.data);
        errorMsg = data.error || errorMsg;
      } catch {
        // ignore
      }
      showError(new Error(errorMsg));
      resetCrowdingButton();
      eventSource.close();
    });
  } catch (error) {
    showError(error);
    resetCrowdingButton();
  }
}

function resetCrowdingButton() {
  elements.runCrowding.disabled = state.detections.length === 0;
  elements.runCrowding.classList.remove("is-running");
  elements.runCrowding.innerHTML = '<span class="inference-icon">◎</span> Analyze Crowding';
  setTimeout(() => {
    elements.crowdingProgress.hidden = true;
    elements.crowdingProgressFill.style.background = "";
  }, 5000);
}

// ─── Inference (model prediction) ──────────────────────────────────────

let inferenceScanId = null;

async function runInference() {
  if (!state.selectedScan) {
    showError(new Error("Select a scan first before running inference."));
    return;
  }

  if (shouldBlockInferenceForGvi()) {
    const proceed = window.confirm(
      `${state.gvi?.message || "Grid viability check failed."}\n\nRun inference anyway?`
    );
    if (!proceed) {
      return;
    }
  }

  // Disable button and show progress
  elements.runInference.disabled = true;
  elements.runInference.classList.add("is-running");
  elements.runInference.innerHTML = '<span class="inference-icon">⏳</span> Running...';
  elements.inferenceProgress.hidden = false;
  elements.inferenceProgressFill.style.width = "0%";
  elements.inferenceStatus.textContent = "Starting inference...";

  inferenceScanId = `infer-${Date.now()}`;

  try {
    // Start the inference via API
    const response = await fetch("/api/inference/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        zarrPath: state.selectedScan,
        scanId: inferenceScanId
      })
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Failed to start inference");
    }

    // Connect to SSE for progress
    const eventSource = new EventSource(`/api/inference/progress?scanId=${inferenceScanId}`);

    eventSource.addEventListener("progress", (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.percent >= 0) {
          elements.inferenceProgressFill.style.width = `${data.percent}%`;
        }
        if (data.message) {
          elements.inferenceStatus.textContent = data.message;
        }
        // Render tqdm-style bar if available
        if (data.bar) {
          elements.inferenceStatus.innerHTML = `Tile ${data.batch} / ${data.total} <span class="tqdm-bar">${data.bar}</span>`;
        }
      } catch (e) {
        // Ignore parse errors
      }
    });

    eventSource.addEventListener("complete", (event) => {
      try {
        const data = JSON.parse(event.data);
        onInferenceComplete(data);
      } catch (e) {
        showError(new Error("Failed to parse inference results"));
        resetInferenceButton();
      }
      eventSource.close();
    });

    eventSource.addEventListener("analysis", (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.analysis) {
          renderAnalysis(data.analysis);
        }
      } catch (e) {
        // Analysis is best-effort
      }
    });

    eventSource.addEventListener("error", (event) => {
      let errorMsg = "Inference failed";
      try {
        const data = JSON.parse(event.data);
        errorMsg = data.error || errorMsg;
      } catch (e) {
        // Use default error message
      }
      showError(new Error(errorMsg));
      resetInferenceButton();
      eventSource.close();
    });

    // Fallback timeout (30 minutes)
    setTimeout(() => {
      if (eventSource.readyState !== EventSource.CLOSED) {
        eventSource.close();
        showError(new Error("Inference timed out after 30 minutes"));
        resetInferenceButton();
      }
    }, 30 * 60 * 1000);

  } catch (err) {
    showError(err);
    resetInferenceButton();
  }
}

function onInferenceComplete(data) {
  if (data.detections) {
    // Update state with new detections
    state.detections = data.detections;
    refreshAnnotationNumbers();
    state.selectedDetection = null;
    state.selectedMolecule = null;
    state.labelsHidden = false;
    state.expandedMolecules.clear();
    state.neighborMap.clear();
    state.crowdingSummary = null;
    state.minOcclusionFilter = 0;
    updateOcclusionFilterUI();
    renderExposureSummary(null);
    renderGraphPanel(null);

    const initialGroups = getMoleculeGroups();
    if (initialGroups.length > 0) {
      state.expandedMolecules.add(initialGroups[0].molecule);
    }

    // Refresh the viewer
    sliceCache.clear();
    renderVolumeScene(state.volume);
    renderDetectionList();
    syncPickOverlays();
    syncViewerAnnotations();
    renderAnalysis(data.analysis || {});
    fetchClaudeAnalysis(state.selectedScan).catch(() => {});
    if (state.volume) {
      Promise.all(sliceAxes.map((axis) => loadSlice(axis))).then(() => {
        startSlicePrecache("Caching inference overlays");
      });
    }

    elements.detectionCount.textContent = `${initialGroups.length} type${initialGroups.length !== 1 ? "s" : ""} · ${state.detections.length} picks`;
    elements.inferenceStatus.textContent = `✓ ${data.numDetections} particles detected`;
    elements.inferenceProgressFill.style.width = "100%";
    elements.inferenceProgressFill.style.background = "#16743a";
    runGviAfterPicks();
  }

  resetInferenceButton();
}

function resetInferenceButton() {
  elements.runInference.disabled = false;
  elements.runInference.classList.remove("is-running");
  elements.runInference.innerHTML = '<span class="inference-icon">▶</span> Run Model Inference';
  // Keep progress visible for a bit, then hide
  setTimeout(() => {
    elements.inferenceProgress.hidden = true;
    elements.inferenceProgressFill.style.background = "";
  }, 5000);
}

function animate() {
  requestAnimationFrame(animate);
  if (state.autoRotate) {
    volumeGroup.rotation.z += 0.002;
  }
  controls.update();
  updateAnnotationPositions();
  renderer.render(scene, camera);
  if (!elements.graphPanel.hidden) {
    graphControls.update();
    graphRenderer.render(graphScene, graphCamera);
  }
}

async function fetchClaudeAnalysis(scanPath) {
  if (!scanPath) {
    return;
  }
  analysisRequestScan = scanPath;
  elements.analysisStatus.textContent = "Generating AI analysis...";
  elements.analysisStatus.classList.add("active");
  try {
    const result = await fetchJson("/api/analysis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ zarrPath: scanPath })
    });
    if (analysisRequestScan === scanPath) {
      renderAnalysis(result);
    }
  } catch {
    if (analysisRequestScan === scanPath) {
      elements.analysisStatus.classList.remove("active");
    }
  }
}

async function fetchClaudeAnalysis(scanPath) {
  if (!scanPath) {
    return;
  }
  analysisRequestScan = scanPath;
  elements.analysisStatus.textContent = "Generating AI analysis...";
  elements.analysisStatus.classList.add("active");
  try {
    const result = await fetchJson("/api/analysis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ zarrPath: scanPath })
    });
    if (analysisRequestScan === scanPath) {
      renderAnalysis(result);
    }
  } catch {
    if (analysisRequestScan === scanPath) {
      elements.analysisStatus.classList.remove("active");
    }
  }
}

elements.scanSelect.addEventListener("change", async () => {
  state.selectedScan = elements.scanSelect.value;
  syncLevelSelect();
  await loadPreview().catch(showError);
});
elements.levelSelect.addEventListener("change", () => loadPreview().catch(showError));
elements.strideSelect.addEventListener("change", () => loadPreview().catch(showError));
elements.pointLimit.addEventListener("change", () => loadPreview().catch(showError));
elements.upload.addEventListener("change", () => uploadZarrFolder().catch(showError));
elements.labelsUpload.addEventListener("change", () => uploadLabelsFolder().catch(showError));
elements.openLocalZarr.addEventListener("click", () => openLocalZarrPath().catch(showError));
elements.resetCamera.addEventListener("click", () => {
  camera.position.set(1.55, 1.25, 1.65);
  controls.target.set(0, 0, 0);
  controls.update();
});
elements.resetGraphCamera.addEventListener("click", () => {
  if (state.selectedDetection) {
    renderGraphPanel(state.selectedDetection);
    return;
  }
  frameGraphCamera(0.35);
});
elements.showAllBtn.addEventListener("click", clearMoleculeSelection);
elements.runInference.addEventListener("click", () => runInference().catch(showError));
elements.runCrowding.addEventListener("click", () => runCrowdingAnalysis().catch(showError));
elements.overlayColorMode.addEventListener("change", () => {
  state.overlayColorMode = elements.overlayColorMode.value;
  refreshSlicePreviews();
  syncPickOverlays();
  syncViewerAnnotations();
});
elements.occlusionFilterSlider?.addEventListener("input", applyOcclusionFilter);
elements.exportCleanPicks?.addEventListener("click", exportCleanCopickPicks);
sliceAxes.forEach((axis) => {
  elements.sliceSliders[axis].addEventListener("input", () => {
    if (!state.volume) {
      return;
    }
    state.currentSlices[axis] = Number(elements.sliceSliders[axis].value);
    setSliceSliderValue(axis);
    setActiveSliceAxis(axis);
    updateSliceSeams();
    if (state.selectedMolecule && !state.selectedDetection) {
      syncViewerAnnotations();
    } else {
      updateAnnotationPositions();
    }
    refreshSlicePreviews();
    scheduleInteractiveSlice(axis);
  });
});
window.addEventListener("resize", () => {
  resizeViewer();
  resizeGraphViewer();
});

function showError(error) {
  setStatus("Error", false);
  setProgress(0);
  elements.message.classList.remove("is-cache-status");
  elements.message.hidden = false;
  elements.message.textContent = error.message;
  console.error(error);
}

setupSlicePreviewInteractions();
resizeViewer();
animate();

loadScans()
  .then((hasScan) => (hasScan ? loadPreview() : undefined))
  .catch(showError);
