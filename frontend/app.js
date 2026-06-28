import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  Deck,
  COORDINATE_SYSTEM,
  ScatterplotLayer,
  VolumeLayer,
  VolumeView,
  getDefaultInitialViewState,
  loadOmeZarr
} from "./vendor/viv-bundle.js";

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
  expandedMolecules: new Set(),
  currentSlices: { x: 0, y: 0, z: 0 },
  activeSliceAxis: "z",
  autoRotate: false,
  vivActive: false
};

function getMoleculeGroups() {
  const map = new Map();
  state.detections.forEach((det) => {
    const key = det.molecule || det.type;
    if (!map.has(key)) {
      map.set(key, { molecule: key, label: det.type, color: det.color || "#0b7f83", picks: [] });
    }
    map.get(key).picks.push(det);
  });
  return [...map.values()];
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
let precacheRunId = 0;
let vivDeck = null;
let vivLoader = null;
let vivView = null;
let vivViewState = null;

const elements = {
  viewer: document.querySelector("#volume-viewer"),
  message: document.querySelector("#viewer-message"),
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
  pointCount: document.querySelector("#point-count"),
  shape: document.querySelector("#volume-shape"),
  range: document.querySelector("#volume-range"),
  stride: document.querySelector("#volume-stride"),
  detectionCount: document.querySelector("#detection-count"),
  analysisStatus: document.querySelector("#analysis-status"),
  analysisSummary: document.querySelector("#analysis-summary"),
  analysisReport: document.querySelector("#analysis-report"),
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

const vivContainer = document.createElement("div");
vivContainer.className = "viv-volume-viewer";
vivContainer.hidden = true;
elements.viewer.appendChild(vivContainer);

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

function resizeViewer() {
  const rect = elements.viewer.getBoundingClientRect();
  camera.aspect = rect.width / Math.max(1, rect.height);
  camera.updateProjectionMatrix();
  renderer.setSize(rect.width, rect.height, false);

  if (vivDeck && state.vivActive) {
    vivDeck.setProps({ width: rect.width, height: rect.height });
  }
}

function scanSupportsViv(scanPath = state.selectedScan) {
  return Boolean(scanPath && !scanPath.startsWith("local:"));
}

function scanPathToVivUrl(scanPath) {
  return `/${String(scanPath).split("/").map(encodeURIComponent).join("/")}`;
}

function sliceInterval(axis) {
  const shape = state.sliceShape || state.volume?.shape;
  if (!shape) {
    return null;
  }
  const voxel = state.currentSlices[axis];
  if (axis === "y") {
    // Viv 3D texture has y inverted: world y=0 = image bottom (voxel y=max)
    const worldY = shape.y - voxel;
    return [worldY, worldY];
  }
  if (axis === "z") {
    // Volume is loaded at coarser resolution; z world units = level-0 z / 2^resolution
    const res = vivLoader ? Math.max(0, vivLoader.length - 1) : 0;
    const worldZ = voxel / (2 ** res);
    return [worldZ, worldZ];
  }
  // x: world x ≈ level-0 voxel x directly
  return [voxel, voxel];
}

function detectionColorArray(detection) {
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

function vivDetectionPosition(detection) {
  const shape = state.sliceShape || state.volume?.levelShapes?.["0"] || state.volume?.shape || { x: 1, y: 1, z: 1 };
  const res = vivLoader ? Math.max(0, vivLoader.length - 1) : 0;
  const voxel = detection.voxel && Number.isFinite(detection.voxel.x)
    ? detection.voxel
    : {
        x: (detection.coords?.[0] + 0.5) * Math.max(1, shape.x - 1),
        y: (detection.coords?.[1] + 0.5) * Math.max(1, shape.y - 1),
        z: (detection.coords?.[2] + 0.5) * Math.max(1, shape.z - 1)
      };
  // Viv world coords: x ≈ voxel x, y is inverted (world y=0 at image bottom),
  // z is scaled to the loaded resolution level
  return [voxel.x, shape.y - voxel.y, voxel.z / (2 ** res)];
}

function vivLayerProps() {
  return {
    id: "qbi-viv-volume",
    loader: vivLoader,
    contrastLimits: [[-0.00004, 0.00002]],
    domain: [[-0.0001, 0.0001]],
    colors: [[255, 226, 168]],
    channelsVisible: [true],
    selections: [{}],
    resolution: Math.max(0, vivLoader.length - 1),
    xSlice: sliceInterval("x"),
    ySlice: sliceInterval("y"),
    zSlice: sliceInterval("z"),
    useProgressIndicator: false,
    useWebGL1Warning: false
  };
}

function vivDetectionLayer() {
  const selectedId = state.selectedDetection?.id;
  const selectedMol = state.selectedMolecule;

  return new ScatterplotLayer({
    id: "qbi-pick-overlay",
    data: state.detections,
    pickable: true,
    coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
    opacity: 0.9,
    stroked: true,
    filled: true,
    lineWidthMinPixels: 1,
    radiusMinPixels: 3,
    radiusMaxPixels: 28,
    getPosition: vivDetectionPosition,
    getRadius: (det) => {
      const radius = moleculeRadiusVoxels(det);
      if (det.id === selectedId) return radius * 1.45;
      const inGroup = !selectedMol || (det.molecule || det.type) === selectedMol;
      return radius * (inGroup ? 1 : 0.72);
    },
    getFillColor: (det) => {
      if (det.id === selectedId) return [255, 185, 50, 255];
      const c = detectionColorArray(det);
      const inGroup = !selectedMol || (det.molecule || det.type) === selectedMol;
      return [...c, inGroup ? 210 : 80];
    },
    getLineColor: (det) => {
      if (det.id === selectedId) return [255, 255, 255, 255];
      const inGroup = !selectedMol || (det.molecule || det.type) === selectedMol;
      return [12, 22, 30, inGroup ? 220 : 50];
    },
    updateTriggers: {
      getFillColor: [selectedId, selectedMol],
      getLineColor: [selectedId, selectedMol],
      getRadius: [selectedId, selectedMol]
    },
    onClick: ({ object }) => {
      if (object?.id) {
        selectDetection(object.id);
      }
    }
  });
}

function updateVivLayers() {
  if (!vivDeck || !vivLoader || !state.vivActive) {
    return;
  }

  vivDeck.setProps({
    layers: [
      new VolumeLayer(vivLayerProps()),
      vivDetectionLayer()
    ]
  });
}

function disableVivViewer() {
  state.vivActive = false;
  vivContainer.hidden = true;
  renderer.domElement.hidden = false;
  if (vivDeck) {
    vivDeck.finalize();
    vivDeck = null;
  }
  vivLoader = null;
  vivView = null;
  vivViewState = null;
}

async function setupVivViewer() {
  if (!scanSupportsViv()) {
    disableVivViewer();
    return false;
  }

  const rect = elements.viewer.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));

  try {
    const { data } = await loadOmeZarr(scanPathToVivUrl(state.selectedScan), { type: "multiscales" });
    vivLoader = data;
    vivViewState = {
      ...getDefaultInitialViewState(vivLoader, { width, height }, 1, true),
      id: "3d",
      rotationX: 25,
      rotationOrbit: 35
    };
    vivView = new VolumeView({
      id: "3d",
      width,
      height,
      target: vivViewState.target,
      useFixedAxis: false
    });

    if (vivDeck) {
      vivDeck.finalize();
    }

    clearGroup(volumeGroup);
    clearGroup(labelGroup);
    markerObjects.clear();
    sliceAxes.forEach((axis) => {
      slicePlaneObjects[axis] = null;
    });
    renderer.domElement.hidden = true;
    vivContainer.hidden = false;
    state.vivActive = true;
    vivDeck = new Deck({
      parent: vivContainer,
      width,
      height,
      views: [vivView.getDeckGlView()],
      viewState: { "3d": vivViewState },
      controller: true,
      useDevicePixels: false,
      onViewStateChange: ({ viewState }) => {
        vivViewState = { ...viewState, id: "3d" };
        vivDeck?.setProps({ viewState: { "3d": vivViewState } });
      },
      layers: [
        new VolumeLayer(vivLayerProps()),
        vivDetectionLayer()
      ]
    });
    return true;
  } catch (error) {
    console.warn("Viv viewer failed; falling back to Three.js slice planes.", error);
    disableVivViewer();
    return false;
  }
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
    // Row 0 of the texture (z=0) must land at world z=-halfZ to match marker positions
    positions = [
      0, halfY, -halfZ,
      0, -halfY, -halfZ,
      0, -halfY, halfZ,
      0, halfY, halfZ
    ];
  } else if (axis === "y") {
    // Same z-direction fix for the y-slice
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
  elements.pointCount.textContent = `${axis.toUpperCase()} slice L${renderedLevel} ${renderedIndex}`;
}

function drawSlicePreview(axis, bytes, width, height, renderedLevel, renderedIndex) {
  const canvas = elements.sliceCanvases[axis];
  if (!canvas) {
    return;
  }

  drawColorizedCanvas(axis, canvas, bytes, width, height);
  elements.sliceLabels[axis].textContent = `${axis.toUpperCase()} ${state.currentSlices[axis]} | L${renderedLevel} ${renderedIndex}`;
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
  clearGroup(labelGroup);
  markerObjects.clear();
  pointsObject = null;
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

  const sharedGeometry = new THREE.SphereGeometry(1, 24, 16);

  payload.detections.forEach((detection) => {
    const markerMaterial = new THREE.MeshStandardMaterial({
      color: detectionColorNumber(detection),
      emissive: new THREE.Color(detectionColorNumber(detection)),
      emissiveIntensity: 0.15,
      roughness: 0.4
    });
    const marker = new THREE.Mesh(sharedGeometry, markerMaterial);
    marker.position.set(
      detection.coords[0] * dimensions.x,
      -detection.coords[1] * dimensions.y,
      detection.coords[2] * dimensions.z
    );
    marker.userData.baseScale = moleculeRadiusWorld(detection);
    marker.scale.setScalar(marker.userData.baseScale);
    marker.userData.detection = detection;
    labelGroup.add(marker);
    markerObjects.set(detection.id, marker);
  });

  updateMarkerHighlights();
  elements.message.hidden = true;
}

function renderDetectionList() {
  elements.detectionList.replaceChildren();
  const groups = getMoleculeGroups();
  const totalPicks = state.detections.length;
  elements.detectionCount.textContent = `${groups.length} type${groups.length !== 1 ? "s" : ""} · ${totalPicks} picks`;
  elements.showAllBtn.hidden = state.selectedMolecule === null;

  groups.forEach((group) => {
    const isSelected = state.selectedMolecule === group.molecule;
    const isExpanded = state.expandedMolecules.has(group.molecule);

    const groupEl = document.createElement("div");
    groupEl.className = "molecule-group";

    const header = document.createElement("button");
    header.className = `molecule-group-header${isSelected ? " is-selected" : ""}`;
    header.type = "button";
    header.innerHTML = `
      <span class="dot" style="background:${group.color}"></span>
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
      group.picks.forEach((det) => {
        const row = document.createElement("button");
        row.className = `molecule-pick-row${det.id === state.selectedDetection?.id ? " is-selected" : ""}`;
        row.type = "button";
        row.dataset.id = det.id;
        row.textContent = det.id;
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

function renderAnalysis(analysis) {
  elements.analysisSummary.replaceChildren();

  if (!analysis?.aggregation?.items?.length) {
    elements.analysisStatus.textContent = "Waiting for labels";
    elements.analysisReport.textContent = "Upload a Picks labels folder to generate molecule counts and a Claude-backed interpretation.";
    return;
  }

  elements.analysisStatus.textContent = analysis.reportStatus || "Generated";
  analysis.aggregation.items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "analysis-card";
    card.innerHTML = `
      <span class="dot" style="background:${item.color}"></span>
      <strong>${item.label}</strong>
      <b>${item.count.toLocaleString()}</b>
      <small>${item.frequencyPercent.toFixed(1)}% · ${item.difficulty}</small>
    `;
    elements.analysisSummary.append(card);
  });

  const report = analysis.report || analysis.localSummary || "No narrative report was returned.";
  elements.analysisReport.replaceChildren();
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

  if (analysis.reportError || analysis.reportErrorBody) {
    const heading = document.createElement("h3");
    heading.textContent = "Claude Error";
    elements.analysisReport.append(heading);

    const errorBlock = document.createElement("pre");
    errorBlock.className = "analysis-error";
    errorBlock.textContent = [analysis.reportError, analysis.reportErrorBody].filter(Boolean).join("\n\n");
    elements.analysisReport.append(errorBlock);
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
  setInfoLabel(elements.selectedNotes, "Notes");
  elements.selectedId.textContent = detection.id;
  elements.selectedType.textContent = detection.type;
  elements.selectedConfidence.textContent = detection.confidence;
  elements.selectedPosition.textContent = detection.position;
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

function updateMarkerHighlights() {
  const selectedId = state.selectedDetection?.id || null;
  const selectedMol = state.selectedMolecule;

  markerObjects.forEach((marker, markerId) => {
    const det = marker.userData.detection;
    const isSelected = markerId === selectedId;
    const inGroup = !selectedMol || (det.molecule || det.type) === selectedMol;

    marker.material.color.set(
      isSelected ? 0xb96d12 : (inGroup ? detectionColorNumber(det) : 0xb8c8cc)
    );
    marker.material.emissiveIntensity = isSelected ? 0.35 : (inGroup ? 0.15 : 0);
    const baseScale = marker.userData.baseScale || moleculeRadiusWorld(det);
    marker.scale.setScalar(baseScale * (isSelected ? 1.45 : inGroup ? 1.0 : 0.6));
  });
}

function selectMoleculeGroup(molecule) {
  state.selectedMolecule = molecule;
  state.selectedDetection = null;

  const group = getMoleculeGroups().find((g) => g.molecule === molecule);
  if (group) {
    showMoleculeGroupInfo(group);
  }

  updateMarkerHighlights();
  updateVivLayers();
}

function clearMoleculeSelection() {
  state.selectedMolecule = null;
  state.selectedDetection = null;
  updateMarkerHighlights();
  updateVivLayers();
  renderDetectionList();
}

function selectDetection(id) {
  const detection = state.detections.find((item) => item.id === id);
  if (!detection) {
    return;
  }

  state.selectedDetection = detection;
  state.selectedMolecule = detection.molecule || detection.type;
  state.expandedMolecules.add(state.selectedMolecule);

  showPickInfo(detection);
  updateMarkerHighlights();

  const marker = markerObjects.get(id);
  if (marker) {
    controls.target.copy(marker.position);
  }

  updateVivLayers();
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
  state.selectedDetection = null;
  state.selectedMolecule = null;
  state.expandedMolecules.clear();
  state.currentSlices = {
    x: Math.floor(state.sliceShape.x / 2),
    y: Math.floor(state.sliceShape.y / 2),
    z: Math.floor(state.sliceShape.z / 2)
  };
  state.activeSliceAxis = "z";
  sliceAxes.forEach((axis) => {
    elements.sliceSliders[axis].max = String(Math.max(0, state.sliceShape[axis] - 1));
    elements.sliceSliders[axis].value = String(state.currentSlices[axis]);
    elements.sliceSliderValues[axis].textContent = `${axis.toUpperCase()} ${state.currentSlices[axis]}`;
  });

  const usingViv = await setupVivViewer();
  if (!usingViv) {
    renderVolumeScene(payload);
  }
  const initialGroups = getMoleculeGroups();
  if (initialGroups.length > 0) {
    state.expandedMolecules.add(initialGroups[0].molecule);
  }
  renderDetectionList();
  renderAnalysis(payload.analysis);
  await Promise.all(sliceAxes.map((axis) => loadSlice(axis)));
  startSlicePrecache("Caching scan slices");
  setActiveSliceAxis("z");
  camera.position.set(1.55, 1.25, 1.65);
  controls.target.set(0, 0, 0);
  controls.update();

  elements.shape.textContent = `${payload.shape.x} x ${payload.shape.y} x ${payload.shape.z}`;
  elements.range.textContent = `${payload.stats.min.toFixed(3)} to ${payload.stats.max.toFixed(3)}`;
  elements.stride.textContent = `${payload.stats.stride}`;
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
  if (!state.vivActive) {
    updateVolumeSlicePlane(axis, slice.bytes, slice.width, slice.height);
  }
  elements.sliceSliders[axis].value = String(state.currentSlices[axis]);
  elements.sliceSliderValues[axis].textContent = `${axis.toUpperCase()} ${state.currentSlices[axis]}`;
  drawSlicePreview(axis, slice.bytes, slice.width, slice.height, slice.renderedLevel, slice.renderedIndex);
  elements.pointCount.textContent = `${axis.toUpperCase()} slice L${slice.renderedLevel} ${slice.renderedIndex}`;
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
  if (payload.scans?.[0]?.path) {
    state.selectedScan = payload.scans[0].path;
    elements.scanSelect.value = state.selectedScan;
  }
  syncLevelSelect();
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
  state.selectedDetection = null;
  state.selectedMolecule = null;
  state.expandedMolecules.clear();
  const initialGroups = getMoleculeGroups();
  if (initialGroups.length > 0) {
    state.expandedMolecules.add(initialGroups[0].molecule);
  }

  sliceCache.clear();
  if (state.vivActive) {
    updateVivLayers();
  } else {
    renderVolumeScene(state.volume);
  }
  renderDetectionList();
  renderAnalysis(payload.analysis);
  await Promise.all(sliceAxes.map((axis) => loadSlice(axis)));
  startSlicePrecache("Caching label overlays");

  elements.detectionCount.textContent = `${initialGroups.length} type${initialGroups.length !== 1 ? "s" : ""} · ${state.detections.length} picks`;
  setProgress(100);
  setStatus(`Loaded ${payload.jsonFiles} label files`, false);
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
elements.labelsUpload.addEventListener("change", () => uploadLabelsFolder().catch(showError));
elements.openLocalZarr.addEventListener("click", () => openLocalZarrPath().catch(showError));
elements.resetCamera.addEventListener("click", () => {
  if (state.vivActive && vivDeck && vivLoader) {
    const rect = elements.viewer.getBoundingClientRect();
    vivViewState = {
      ...getDefaultInitialViewState(vivLoader, { width: rect.width, height: rect.height }, 1, true),
      id: "3d",
      rotationX: 25,
      rotationOrbit: 35
    };
    vivDeck.setProps({ viewState: { "3d": vivViewState } });
    return;
  }
  camera.position.set(1.55, 1.25, 1.65);
  controls.target.set(0, 0, 0);
  controls.update();
});
elements.rotateToggle.addEventListener("click", () => {
  state.autoRotate = !state.autoRotate;
  elements.rotateToggle.classList.toggle("is-active", state.autoRotate);
});
elements.openSlicer.addEventListener("click", () => openSelectedScanInSlicer().catch(showError));
elements.reloadScan.addEventListener("click", () => loadPreview().catch(showError));
elements.showAllBtn.addEventListener("click", clearMoleculeSelection);
sliceAxes.forEach((axis) => {
  elements.sliceSliders[axis].addEventListener("input", () => {
    if (!state.volume) {
      return;
    }
    state.currentSlices[axis] = Number(elements.sliceSliders[axis].value);
    elements.sliceSliderValues[axis].textContent = `${axis.toUpperCase()} ${state.currentSlices[axis]}`;
    setActiveSliceAxis(axis);
    updateSliceSeams();
    updateVivLayers();
    refreshSlicePreviews();
    scheduleInteractiveSlice(axis);
  });
});
renderer.domElement.addEventListener("pointerdown", onPointerDown);
window.addEventListener("resize", resizeViewer);

function showError(error) {
  setStatus("Error", false);
  setProgress(0);
  elements.message.classList.remove("is-cache-status");
  elements.message.hidden = false;
  elements.message.textContent = error.message;
  console.error(error);
}

resizeViewer();
animate();

loadScans()
  .then(loadPreview)
  .catch(showError);
