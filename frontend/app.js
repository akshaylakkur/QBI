import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const state = {
  scans: [],
  selectedScan: null,
  selectedLevel: "2",
  volume: null,
  detections: [],
  selectedDetection: null,
  currentSlice: 0,
  autoRotate: false
};

const elements = {
  viewer: document.querySelector("#volume-viewer"),
  message: document.querySelector("#viewer-message"),
  scanSelect: document.querySelector("#scan-select"),
  levelSelect: document.querySelector("#level-select"),
  strideSelect: document.querySelector("#stride-select"),
  pointLimit: document.querySelector("#point-limit"),
  pointSize: document.querySelector("#point-size"),
  upload: document.querySelector("#zarr-upload"),
  loadStatus: document.querySelector("#load-status"),
  progress: document.querySelector("#load-progress"),
  pointCount: document.querySelector("#point-count"),
  shape: document.querySelector("#volume-shape"),
  range: document.querySelector("#volume-range"),
  stride: document.querySelector("#volume-stride"),
  detectionCount: document.querySelector("#detection-count"),
  detectionList: document.querySelector("#detection-list"),
  selectedId: document.querySelector("#selected-id"),
  selectedType: document.querySelector("#selected-type"),
  selectedConfidence: document.querySelector("#selected-confidence"),
  selectedPosition: document.querySelector("#selected-position"),
  selectedNotes: document.querySelector("#selected-notes"),
  sliceCanvas: document.querySelector("#slice-canvas"),
  sliceLabel: document.querySelector("#slice-label"),
  resetCamera: document.querySelector("#reset-camera"),
  rotateToggle: document.querySelector("#rotate-toggle"),
  reloadScan: document.querySelector("#reload-scan"),
  sliceBack: document.querySelector("#slice-back"),
  sliceForward: document.querySelector("#slice-forward")
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf8fbfb);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(1.55, 1.25, 1.65);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
elements.viewer.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

const volumeGroup = new THREE.Group();
const labelGroup = new THREE.Group();
scene.add(volumeGroup);
scene.add(labelGroup);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let pointsObject = null;
const markerObjects = new Map();

scene.add(new THREE.AmbientLight(0xffffff, 1));
const grid = new THREE.GridHelper(1.2, 12, 0x8aa1a6, 0xd8e3e4);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

function setStatus(text, active = false) {
  elements.loadStatus.textContent = text;
  elements.loadStatus.classList.toggle("active", active);
}

function setProgress(percent) {
  elements.progress.style.width = `${Math.max(0, Math.min(100, percent))}%`;
}

function resizeViewer() {
  const rect = elements.viewer.getBoundingClientRect();
  camera.aspect = rect.width / Math.max(1, rect.height);
  camera.updateProjectionMatrix();
  renderer.setSize(rect.width, rect.height, false);
}

function clearGroup(group) {
  while (group.children.length > 0) {
    const child = group.children.pop();
    child.geometry?.dispose();
    if (Array.isArray(child.material)) {
      child.material.forEach((material) => material.dispose());
    } else {
      child.material?.dispose();
    }
  }
}

function renderPointCloud(payload) {
  clearGroup(volumeGroup);
  clearGroup(labelGroup);
  markerObjects.clear();

  const positions = new Float32Array(payload.points.length * 3);
  const colors = new Float32Array(payload.points.length * 3);
  const low = new THREE.Color(0x315f9d);
  const high = new THREE.Color(0x12a187);

  payload.points.forEach(([x, y, z, intensity], index) => {
    positions[index * 3] = x;
    positions[index * 3 + 1] = -y;
    positions[index * 3 + 2] = z;

    const color = low.clone().lerp(high, intensity);
    colors[index * 3] = color.r;
    colors[index * 3 + 1] = color.g;
    colors[index * 3 + 2] = color.b;
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: Number(elements.pointSize.value) / 1000,
    vertexColors: true,
    transparent: true,
    opacity: 0.86,
    depthWrite: false
  });

  pointsObject = new THREE.Points(geometry, material);
  volumeGroup.add(pointsObject);

  payload.detections.forEach((detection) => {
    const markerGeometry = new THREE.SphereGeometry(0.028, 24, 16);
    const markerMaterial = new THREE.MeshStandardMaterial({
      color: detection.id === state.selectedDetection?.id ? 0xb96d12 : 0x0b7f83,
      emissive: 0x063f42,
      emissiveIntensity: 0.2,
      roughness: 0.4
    });
    const marker = new THREE.Mesh(markerGeometry, markerMaterial);
    marker.position.set(detection.coords[0], -detection.coords[1], detection.coords[2]);
    marker.userData.detection = detection;
    labelGroup.add(marker);
    markerObjects.set(detection.id, marker);
  });

  elements.message.hidden = true;
}

function renderDetectionList() {
  elements.detectionList.replaceChildren();
  elements.detectionCount.textContent = `${state.detections.length} labels`;

  state.detections.forEach((detection, index) => {
    const row = document.createElement("button");
    row.className = "detection-row";
    row.type = "button";
    row.dataset.id = detection.id;
    row.innerHTML = `
      <span class="dot ${index === 1 ? "secondary" : index === 2 ? "accent" : ""}"></span>
      <span>
        <strong>${detection.id}</strong>
        <small>${detection.type}</small>
      </span>
      <b>${index + 1}</b>
    `;
    row.addEventListener("click", () => selectDetection(detection.id));
    elements.detectionList.append(row);
  });
}

function selectDetection(id) {
  const detection = state.detections.find((item) => item.id === id);
  if (!detection) {
    return;
  }

  state.selectedDetection = detection;
  elements.selectedId.textContent = detection.id;
  elements.selectedType.textContent = detection.type;
  elements.selectedConfidence.textContent = detection.confidence;
  elements.selectedPosition.textContent = detection.position;
  elements.selectedNotes.textContent = detection.notes;

  document.querySelectorAll(".detection-row").forEach((row) => {
    row.classList.toggle("is-selected", row.dataset.id === id);
  });

  markerObjects.forEach((marker, markerId) => {
    marker.material.color.set(markerId === id ? 0xb96d12 : 0x0b7f83);
    marker.scale.setScalar(markerId === id ? 1.45 : 1);
  });

  const marker = markerObjects.get(id);
  if (marker) {
    controls.target.copy(marker.position);
  }
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || response.statusText);
  }
  return response.json();
}

async function loadScans() {
  setStatus("Finding scans", true);
  const payload = await fetchJson("/api/scans");
  state.scans = payload.scans;
  elements.scanSelect.replaceChildren(
    ...state.scans.map((scan) => {
      const option = document.createElement("option");
      option.value = scan.path;
      option.textContent = scan.name;
      return option;
    })
  );

  if (state.scans.length === 0) {
    throw new Error("No sample or uploaded Zarr scans found");
  }

  state.selectedScan = state.scans[0].path;
  elements.scanSelect.value = state.selectedScan;
}

async function loadPreview() {
  if (!state.selectedScan) {
    return;
  }

  setStatus("Decoding", true);
  setProgress(22);
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
  state.volume = payload;
  state.detections = payload.detections;
  state.currentSlice = Math.floor(payload.shape.z / 2);

  renderPointCloud(payload);
  renderDetectionList();
  selectDetection(payload.detections[0]?.id);
  await loadSlice();

  elements.pointCount.textContent = `${payload.stats.points.toLocaleString()} points`;
  elements.shape.textContent = `${payload.shape.x} x ${payload.shape.y} x ${payload.shape.z}`;
  elements.range.textContent = `${payload.stats.min.toFixed(3)} to ${payload.stats.max.toFixed(3)}`;
  elements.stride.textContent = `${payload.stats.stride}`;
  setProgress(100);
  setStatus("Ready", false);
}

async function loadSlice() {
  if (!state.volume) {
    return;
  }

  const params = new URLSearchParams({
    path: state.selectedScan,
    level: elements.levelSelect.value,
    z: String(state.currentSlice)
  });
  const response = await fetch(`/api/zarr/slice?${params.toString()}`);
  if (!response.ok) {
    return;
  }

  const bytes = new Uint8ClampedArray(await response.arrayBuffer());
  const { x: width, y: height } = state.volume.shape;
  const canvas = elements.sliceCanvas;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const image = context.createImageData(width, height);

  for (let index = 0; index < bytes.length; index += 1) {
    const value = bytes[index];
    image.data[index * 4] = value;
    image.data[index * 4 + 1] = value;
    image.data[index * 4 + 2] = value;
    image.data[index * 4 + 3] = 255;
  }

  context.putImageData(image, 0, 0);
  elements.sliceLabel.textContent = `Z ${state.currentSlice}`;
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
  if (payload.scans?.[0]?.path) {
    state.selectedScan = payload.scans[0].path;
    elements.scanSelect.value = state.selectedScan;
  }
  await loadPreview();
}

function onPointerDown(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const intersections = raycaster.intersectObjects([...markerObjects.values()]);
  const selected = intersections[0]?.object?.userData?.detection;
  if (selected) {
    selectDetection(selected.id);
  }
}

function animate() {
  requestAnimationFrame(animate);
  if (state.autoRotate) {
    volumeGroup.rotation.z += 0.002;
    labelGroup.rotation.z += 0.002;
  }
  controls.update();
  renderer.render(scene, camera);
}

elements.scanSelect.addEventListener("change", async () => {
  state.selectedScan = elements.scanSelect.value;
  await loadPreview().catch(showError);
});
elements.levelSelect.addEventListener("change", () => loadPreview().catch(showError));
elements.strideSelect.addEventListener("change", () => loadPreview().catch(showError));
elements.pointLimit.addEventListener("change", () => loadPreview().catch(showError));
elements.pointSize.addEventListener("input", () => {
  if (pointsObject) {
    pointsObject.material.size = Number(elements.pointSize.value) / 1000;
  }
});
elements.upload.addEventListener("change", () => uploadZarrFolder().catch(showError));
elements.resetCamera.addEventListener("click", () => {
  camera.position.set(1.55, 1.25, 1.65);
  controls.target.set(0, 0, 0);
});
elements.rotateToggle.addEventListener("click", () => {
  state.autoRotate = !state.autoRotate;
  elements.rotateToggle.classList.toggle("is-active", state.autoRotate);
});
elements.reloadScan.addEventListener("click", () => loadPreview().catch(showError));
elements.sliceBack.addEventListener("click", async () => {
  state.currentSlice = Math.max(0, state.currentSlice - 1);
  await loadSlice();
});
elements.sliceForward.addEventListener("click", async () => {
  if (!state.volume) {
    return;
  }
  state.currentSlice = Math.min(state.volume.shape.z - 1, state.currentSlice + 1);
  await loadSlice();
});
renderer.domElement.addEventListener("pointerdown", onPointerDown);
window.addEventListener("resize", resizeViewer);

function showError(error) {
  setStatus("Error", false);
  setProgress(0);
  elements.message.hidden = false;
  elements.message.textContent = error.message;
  console.error(error);
}

resizeViewer();
animate();

loadScans()
  .then(loadPreview)
  .catch(showError);
