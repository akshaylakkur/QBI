import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const state = {
  scans: [],
  selectedScan: null,
  selectedLevel: "2",
  volume: null,
  detections: [],
  selectedDetection: null,
  currentSlices: { x: 0, y: 0, z: 0 },
  activeSliceAxis: "z",
  autoRotate: false
};

const highDetailSliceLevel = "0";
const highDetailSliceMaxSize = "600";
const sliceAxes = ["x", "y", "z"];

const elements = {
  viewer: document.querySelector("#volume-viewer"),
  message: document.querySelector("#viewer-message"),
  scanSelect: document.querySelector("#scan-select"),
  levelSelect: document.querySelector("#level-select"),
  strideSelect: document.querySelector("#stride-select"),
  pointLimit: document.querySelector("#point-limit"),
  pointSize: document.querySelector("#point-size"),
  upload: document.querySelector("#zarr-upload"),
  localZarrPath: document.querySelector("#local-zarr-path"),
  openLocalZarr: document.querySelector("#open-local-zarr"),
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
  resetCamera: document.querySelector("#reset-camera"),
  rotateToggle: document.querySelector("#rotate-toggle"),
  openSlicer: document.querySelector("#open-slicer"),
  reloadScan: document.querySelector("#reload-scan"),
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
  }
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
const slicePlaneObjects = { x: null, y: null, z: null };
const sliceRequestIds = { x: 0, y: 0, z: 0 };
const markerObjects = new Map();

scene.add(new THREE.AmbientLight(0xffffff, 1.05));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.35);
keyLight.position.set(2.2, 2.4, 3);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x8ddbf0, 0.7);
rimLight.position.set(-2.5, -1.8, -1.5);
scene.add(rimLight);

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
  return (state.currentSlices[axis] / Math.max(1, state.volume.shape[axis] - 1)) - 0.5;
}

function createSliceGeometry(axis) {
  if (axis === "x") {
    return new THREE.PlaneGeometry(1, 1).rotateY(Math.PI / 2);
  }
  if (axis === "y") {
    return new THREE.PlaneGeometry(1, 1).rotateX(Math.PI / 2);
  }
  return new THREE.PlaneGeometry(1, 1);
}

function positionSlicePlane(axis, plane) {
  if (axis === "x") {
    plane.position.set(normalizedSlicePosition("x"), 0, 0);
    return;
  }
  if (axis === "y") {
    plane.position.set(0, -normalizedSlicePosition("y"), 0);
    return;
  }
  plane.position.set(0, 0, normalizedSlicePosition("z"));
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
    uX: { value: normalizedSlicePosition("x") },
    uY: { value: -normalizedSlicePosition("y") },
    uZ: { value: normalizedSlicePosition("z") },
    uGap: { value: 0.0035 },
    uBrightness: { value: 1.35 },
    uGamma: { value: 0.72 },
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
    x: normalizedSlicePosition("x"),
    y: -normalizedSlicePosition("y"),
    z: normalizedSlicePosition("z")
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
  const textureData = new Uint8Array(width * height * 4);
  for (let index = 0; index < bytes.length; index += 1) {
    const value = bytes[index];
    textureData[index * 4] = Math.min(255, Math.round(value * 1.04));
    textureData[index * 4 + 1] = Math.min(255, Math.round(value * 1.08));
    textureData[index * 4 + 2] = Math.min(255, Math.round(value * 1.1 + 8));
    textureData[index * 4 + 3] = 255;
  }

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

function drawSlicePreview(axis, bytes, width, height, renderedLevel, renderedIndex) {
  const canvas = elements.sliceCanvases[axis];
  if (!canvas) {
    return;
  }

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

  const xRatio = state.currentSlices.x / Math.max(1, state.volume.shape.x - 1);
  const yRatio = state.currentSlices.y / Math.max(1, state.volume.shape.y - 1);
  const zRatio = state.currentSlices.z / Math.max(1, state.volume.shape.z - 1);
  const lines = axis === "x"
    ? [
        { color: "#26c943", vertical: yRatio },
        { color: "#2465ff", horizontal: zRatio }
      ]
    : axis === "y"
      ? [
          { color: "#ff2b23", vertical: xRatio },
          { color: "#2465ff", horizontal: zRatio }
        ]
      : [
          { color: "#ff2b23", vertical: xRatio },
          { color: "#26c943", horizontal: yRatio }
        ];

  context.save();
  context.lineWidth = Math.max(1, Math.round(Math.min(width, height) * 0.006));
  lines.forEach((line) => {
    context.strokeStyle = line.color;
    context.beginPath();
    if (Number.isFinite(line.vertical)) {
      const x = Math.round(line.vertical * (width - 1)) + 0.5;
      context.moveTo(x, 0);
      context.lineTo(x, height);
    } else {
      const y = Math.round(line.horizontal * (height - 1)) + 0.5;
      context.moveTo(0, y);
      context.lineTo(width, y);
    }
    context.stroke();
  });
  context.restore();

  elements.sliceLabels[axis].textContent = `${axis.toUpperCase()} ${state.currentSlices[axis]} | L${renderedLevel} ${renderedIndex}`;
}

function refreshSlicePreviews() {
  sliceAxes.forEach((axis) => {
    const canvas = elements.sliceCanvases[axis];
    if (!canvas?.width || !canvas?.height) {
      return;
    }

    const context = canvas.getContext("2d");
    const image = context.getImageData(0, 0, canvas.width, canvas.height);
    const bytes = new Uint8ClampedArray(canvas.width * canvas.height);
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = image.data[index * 4];
    }

    drawSlicePreview(axis, bytes, canvas.width, canvas.height, "-", "-");
  });
}

function renderPointCloud(payload) {
  clearGroup(volumeGroup);
  clearGroup(labelGroup);
  markerObjects.clear();
  sliceAxes.forEach((axis) => {
    slicePlaneObjects[axis] = null;
  });

  const positions = new Float32Array(payload.points.length * 3);
  const colors = new Float32Array(payload.points.length * 3);
  const strengths = new Float32Array(payload.points.length);
  const densities = new Float32Array(payload.points.length);

  payload.points.forEach(([x, y, z, strength, density = strength], index) => {
    positions[index * 3] = x;
    positions[index * 3 + 1] = -y;
    positions[index * 3 + 2] = z;

    strengths[index] = strength;
    densities[index] = density;
    const color = densityColor(strength, density);
    colors[index * 3] = color.r;
    colors[index * 3 + 1] = color.g;
    colors[index * 3 + 2] = color.b;
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute("strength", new THREE.BufferAttribute(strengths, 1));
  geometry.setAttribute("density", new THREE.BufferAttribute(densities, 1));

  pointsObject = new THREE.Points(geometry, createVolumePointMaterial());
  pointsObject.renderOrder = 2;
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

async function fetchJson(url, options) {
  const response = await fetch(url, options);
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
      option.disabled = scan.metadataMissing;
      return option;
    })
  );

  const firstUsableScan = state.scans.find((scan) => !scan.metadataMissing);
  if (!firstUsableScan) {
    throw new Error("No sample or uploaded Zarr scans found");
  }

  state.selectedScan = firstUsableScan.path;
  elements.scanSelect.value = state.selectedScan;
  syncLevelSelect();
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
  state.detections = payload.detections;
  state.currentSlices = {
    x: Math.floor(payload.shape.x / 2),
    y: Math.floor(payload.shape.y / 2),
    z: Math.floor(payload.shape.z / 2)
  };
  state.activeSliceAxis = "z";
  sliceAxes.forEach((axis) => {
    elements.sliceSliders[axis].max = String(Math.max(0, payload.shape[axis] - 1));
    elements.sliceSliders[axis].value = String(state.currentSlices[axis]);
    elements.sliceSliderValues[axis].textContent = `${axis.toUpperCase()} ${state.currentSlices[axis]}`;
  });

  renderPointCloud(payload);
  renderDetectionList();
  selectDetection(payload.detections[0]?.id);
  await Promise.all(sliceAxes.map((axis) => loadSlice(axis)));
  setActiveSliceAxis("z");

  elements.pointCount.textContent = `${payload.stats.points.toLocaleString()} points`;
  elements.shape.textContent = `${payload.shape.x} x ${payload.shape.y} x ${payload.shape.z}`;
  elements.range.textContent = `${payload.stats.min.toFixed(3)} to ${payload.stats.max.toFixed(3)}`;
  elements.stride.textContent = `${payload.stats.stride}`;
  setProgress(100);
  setStatus("Ready", false);
}

async function loadSlice(axis = "z") {
  if (!state.volume) {
    return;
  }

  const requestId = sliceRequestIds[axis] + 1;
  sliceRequestIds[axis] = requestId;
  const requestedSlice = state.currentSlices[axis];

  const previewLevel = elements.levelSelect.value;
  const detailLevel = highDetailSliceLevel;
  const params = new URLSearchParams({
    path: state.selectedScan,
    axis,
    level: detailLevel,
    sourceLevel: previewLevel,
    maxSize: highDetailSliceMaxSize,
    index: String(requestedSlice)
  });
  let response = await fetch(`/api/zarr/slice?${params.toString()}`);
  if (!response.ok && detailLevel !== previewLevel) {
    params.set("level", previewLevel);
    params.delete("sourceLevel");
    response = await fetch(`/api/zarr/slice?${params.toString()}`);
  }
  if (!response.ok) {
    return;
  }

  const bytes = new Uint8ClampedArray(await response.arrayBuffer());
  if (requestId !== sliceRequestIds[axis] || requestedSlice !== state.currentSlices[axis]) {
    return;
  }

  const width = Number(response.headers.get("X-QBI-Slice-Width")) || state.volume.shape.x;
  const height = Number(response.headers.get("X-QBI-Slice-Height")) || state.volume.shape.y;
  const renderedLevel = response.headers.get("X-QBI-Slice-Level") || previewLevel;
  const renderedIndex = response.headers.get("X-QBI-Slice-Index") || String(requestedSlice);
  updateVolumeSlicePlane(axis, bytes, width, height);

  elements.sliceSliders[axis].value = String(state.currentSlices[axis]);
  elements.sliceSliderValues[axis].textContent = `${axis.toUpperCase()} ${state.currentSlices[axis]}`;
  drawSlicePreview(axis, bytes, width, height, renderedLevel, renderedIndex);
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
  syncLevelSelect();
  await loadPreview();
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
  syncLevelSelect();
  await loadPreview().catch(showError);
});
elements.levelSelect.addEventListener("change", () => loadPreview().catch(showError));
elements.strideSelect.addEventListener("change", () => loadPreview().catch(showError));
elements.pointLimit.addEventListener("change", () => loadPreview().catch(showError));
elements.pointSize.addEventListener("input", () => {
  if (pointsObject) {
    pointsObject.material.uniforms.uPointSize.value = pointSizePixels();
  }
});
elements.upload.addEventListener("change", () => uploadZarrFolder().catch(showError));
elements.openLocalZarr.addEventListener("click", () => openLocalZarrPath().catch(showError));
elements.resetCamera.addEventListener("click", () => {
  camera.position.set(1.55, 1.25, 1.65);
  controls.target.set(0, 0, 0);
});
elements.rotateToggle.addEventListener("click", () => {
  state.autoRotate = !state.autoRotate;
  elements.rotateToggle.classList.toggle("is-active", state.autoRotate);
});
elements.openSlicer.addEventListener("click", () => openSelectedScanInSlicer().catch(showError));
elements.reloadScan.addEventListener("click", () => loadPreview().catch(showError));
sliceAxes.forEach((axis) => {
  elements.sliceSliders[axis].addEventListener("input", () => {
    if (!state.volume) {
      return;
    }
    state.currentSlices[axis] = Number(elements.sliceSliders[axis].value);
    elements.sliceSliderValues[axis].textContent = `${axis.toUpperCase()} ${state.currentSlices[axis]}`;
    setActiveSliceAxis(axis);
    updateSliceSeams();
    refreshSlicePreviews();
    loadSlice(axis).catch(showError);
  });
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
