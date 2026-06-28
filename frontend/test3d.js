import {
  Deck,
  VolumeView,
  getDefaultInitialViewState,
  loadOmeZarr
} from "./vendor/viv-bundle.js";

const zarrPath = "sample_data/cryo_object detection/train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr";
const viewId = "test3d";
const canvas = document.querySelector("#test3d-canvas");
const message = document.querySelector("#test3d-message");
const status = document.querySelector("#test3d-status");
const shape = document.querySelector("#test3d-shape");

let deck = null;
let loader = null;
let layerProps = null;
let volumeView = null;
let viewState = null;

function zarrUrlFromPath(path) {
  return `${window.location.origin}/${path.split("/").map(encodeURIComponent).join("/")}`;
}

function setStatus(text) {
  status.textContent = text;
}

function setMessage(text, hidden = false) {
  message.textContent = text;
  message.hidden = hidden;
}

function viewportSize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  return {
    width: Math.max(1, Math.floor(rect.width)),
    height: Math.max(1, Math.floor(rect.height))
  };
}

function renderDeck() {
  if (!deck || !loader || !layerProps || !volumeView || !viewState) {
    return;
  }

  const { width, height } = viewportSize();
  volumeView.width = width;
  volumeView.height = height;

  deck.setProps({
    width,
    height,
    views: [volumeView.getDeckGlView()],
    viewState: { [viewId]: { ...viewState, width, height, id: viewId } },
    layers: volumeView.getLayers({ props: layerProps }),
    layerFilter: ({ layer, viewport }) => layer.id.includes(`-#${viewport.id}#`),
    onViewStateChange: ({ viewId: changedViewId, viewState: nextViewState }) => {
      if (changedViewId === viewId) {
        viewState = nextViewState;
      }
    },
    getCursor: ({ isDragging }) => isDragging ? "grabbing" : "grab"
  });
}

async function loadStats() {
  const params = new URLSearchParams({
    path: zarrPath,
    level: "2",
    stride: "8",
    limit: "1000"
  });
  const response = await fetch(`/api/zarr/preview?${params.toString()}`);
  if (!response.ok) {
    return null;
  }
  return response.json();
}

async function init() {
  try {
    setStatus("Loading");
    const [vivPayload, statsPayload] = await Promise.all([
      loadOmeZarr(zarrUrlFromPath(zarrPath), { type: "multiscales" }),
      loadStats()
    ]);

    loader = vivPayload.data;
    const source = loader[loader.length - 1];
    const zIndex = source.labels.indexOf("z");
    const yIndex = source.labels.indexOf("y");
    const xIndex = source.labels.indexOf("x");
    const sourceShape = {
      z: source.shape[zIndex],
      y: source.shape[yIndex],
      x: source.shape[xIndex]
    };
    const stats = statsPayload?.stats;
    const hasUsableStats = stats && Number.isFinite(stats.min) && Number.isFinite(stats.max) && stats.min < stats.max;
    const domain = hasUsableStats ? [stats.min, stats.max] : [-0.1, 0.1];
    const contrastLimits = [domain];
    const { width, height } = viewportSize();

    shape.textContent = `Shape ${sourceShape.x} x ${sourceShape.y} x ${sourceShape.z}`;
    volumeView = new VolumeView({
      id: viewId,
      target: [sourceShape.x / 2, sourceShape.y / 2, sourceShape.z / 2],
      useFixedAxis: false,
      height,
      width
    });
    viewState = {
      ...getDefaultInitialViewState(loader, { width, height }, 1, true),
      id: viewId,
      rotationX: 35,
      rotationOrbit: 35
    };
    layerProps = {
      loader,
      resolution: loader.length - 1,
      selections: [{}],
      channelsVisible: [true],
      colors: [[255, 255, 255]],
      contrastLimits,
      domain,
      xSlice: [0, sourceShape.x],
      ySlice: [0, sourceShape.y],
      zSlice: [0, sourceShape.z],
      useProgressIndicator: true,
      onUpdate: ({ progress }) => {
        setStatus(`Loading ${Math.round(progress * 100)}%`);
      },
      onViewportLoad: () => {
        setStatus("Ready");
        setMessage("", true);
      }
    };

    deck = new Deck({
      canvas,
      controller: true,
      useDevicePixels: false,
      initialViewState: { [viewId]: viewState }
    });
    renderDeck();
  } catch (error) {
    setStatus("Error");
    setMessage(error.message);
    console.error(error);
  }
}

window.addEventListener("resize", renderDeck);
init();
