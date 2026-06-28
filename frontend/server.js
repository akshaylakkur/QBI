const http = require("node:http");
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const path = require("node:path");
const os = require("node:os");
const { spawn } = require("node:child_process");
const { EventEmitter } = require("node:events");

const port = Number.parseInt(process.env.PORT || "3000", 10);
const publicDir = __dirname;
const sampleDir = path.join(__dirname, "sample_data");
const uploadsDir = path.join(__dirname, "uploaded_scans");
const labelUploadsDir = path.join(__dirname, "uploaded_labels");
const predictionsDir = path.join(__dirname, "predictions");
const slicerExportDir = path.join(os.tmpdir(), "qbi-slicer-exports");
const volumeCache = new Map();
const volumeLoadCache = new Map();
const sliceCache = new Map();
const sliceLoadCache = new Map();
let sliceCacheBytes = 0;
const maxSliceCacheBytes = 96 * 1024 * 1024;
const localScans = new Map();
const pickCache = new Map();
const latestAggregation = new Map();

// Inference progress tracking (SSE)
const inferenceEmitters = new Map();
const crowdingEmitters = new Map();

function emitJobEvent(emitter, event, payload) {
  if (emitter?.listenerCount(event) > 0) {
    emitter.emit(event, payload);
  }
}

function loadEnvFile() {
  const envPath = path.join(__dirname, ".env");
  if (!fs.existsSync(envPath)) {
    return;
  }

  const lines = fs.readFileSync(envPath, "utf8").split(/\r?\n/);
  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) {
      return;
    }
    const separator = trimmed.indexOf("=");
    const key = trimmed.slice(0, separator).trim();
    const value = trimmed.slice(separator + 1).trim().replace(/^["']|["']$/g, "");
    if (key && process.env[key] === undefined) {
      process.env[key] = value;
    }
  });
}

loadEnvFile();

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".gif": "image/gif",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".txt": "text/plain; charset=utf-8",
  ".webp": "image/webp"
};

function send(res, statusCode, body, contentType = "text/plain; charset=utf-8") {
  res.writeHead(statusCode, {
    "Content-Type": contentType,
    "Content-Length": Buffer.byteLength(body)
  });
  res.end(body);
}

function sendJson(res, statusCode, body) {
  send(res, statusCode, JSON.stringify(body), "application/json; charset=utf-8");
}

function sendBuffer(res, statusCode, body, contentType = "application/octet-stream", extraHeaders = {}) {
  res.writeHead(statusCode, {
    "Content-Type": contentType,
    "Content-Length": body.length,
    ...extraHeaders
  });
  res.end(body);
}

function sendDownload(res, body, filename, contentType = "application/octet-stream") {
  res.writeHead(200, {
    "Content-Type": contentType,
    "Content-Length": body.length,
    "Content-Disposition": `attachment; filename="${filename}"`
  });
  res.end(body);
}

function resolveStaticPath(requestUrl) {
  const url = new URL(requestUrl, `http://localhost:${port}`);
  const decodedPath = decodeURIComponent(url.pathname);
  const normalizedPath = path.normalize(decodedPath).replace(/^(\.\.[/\\])+/, "");
  const filePath = path.join(publicDir, normalizedPath);

  if (!filePath.startsWith(publicDir)) {
    return null;
  }

  return filePath;
}

function safeSegment(value) {
  return value.replace(/[^a-zA-Z0-9._-]/g, "_").replace(/^_+/, "").slice(0, 80) || "scan";
}

function resolveScanPath(scanPath) {
  if (scanPath?.startsWith("local:")) {
    return localScans.get(scanPath) || null;
  }

  const decoded = decodeURIComponent(scanPath || "");
  const normalized = path.normalize(decoded).replace(/^(\.\.[/\\])+/, "");
  const filePath = path.join(__dirname, normalized);
  const allowedRoots = [sampleDir, uploadsDir];

  if (!allowedRoots.some((root) => filePath === root || filePath.startsWith(`${root}${path.sep}`))) {
    return null;
  }

  return filePath;
}

async function readRequestJson(req) {
  const chunks = [];
  let total = 0;

  for await (const chunk of req) {
    chunks.push(chunk);
    total += chunk.length;
    if (total > 1_000_000) {
      throw new Error("Request body too large");
    }
  }

  return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
}

async function readMultipartFolder(req, targetRoot, maxBytes = 1_000_000_000) {
  await fsp.mkdir(targetRoot, { recursive: true });
  const chunks = [];
  let total = 0;

  req.on("data", (chunk) => {
    chunks.push(chunk);
    total += chunk.length;
    if (total > maxBytes) {
      req.destroy();
    }
  });

  return new Promise((resolve, reject) => {
    req.on("error", reject);
    req.on("end", async () => {
      try {
        const boundaryMatch = /boundary=(.+)$/.exec(req.headers["content-type"] || "");
        if (!boundaryMatch) {
          resolve({ error: "Multipart boundary not found", files: 0 });
          return;
        }

        const boundary = `--${boundaryMatch[1]}`;
        const body = Buffer.concat(chunks);
        const parts = body.toString("latin1").split(boundary).slice(1, -1);
        let files = 0;

        for (const part of parts) {
          const separator = part.indexOf("\r\n\r\n");
          if (separator === -1) {
            continue;
          }

          const headers = part.slice(0, separator);
          const filenameMatch = /filename="([^"]+)"/.exec(headers);
          if (!filenameMatch) {
            continue;
          }

          const relativeName = filenameMatch[1].split(/[\\/]/).map(safeSegment).join(path.sep);
          const targetPath = path.join(targetRoot, relativeName);
          if (!targetPath.startsWith(targetRoot)) {
            continue;
          }

          const payload = Buffer.from(part.slice(separator + 4, -2), "latin1");
          await fsp.mkdir(path.dirname(targetPath), { recursive: true });
          await fsp.writeFile(targetPath, payload);
          files += 1;
        }

        resolve({ files });
      } catch (error) {
        reject(error);
      }
    });
  });
}

async function listZarrLevels(zarrPath) {
  let entries = [];
  try {
    entries = await fsp.readdir(zarrPath, { withFileTypes: true });
  } catch {
    return [];
  }

  const levels = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }

    const metadataPath = path.join(zarrPath, entry.name, ".zarray");
    if (fs.existsSync(metadataPath)) {
      levels.push(entry.name);
    }
  }

  return levels.sort((a, b) => {
    const numericA = Number(a);
    const numericB = Number(b);
    if (Number.isFinite(numericA) && Number.isFinite(numericB)) {
      return numericB - numericA;
    }
    return b.localeCompare(a);
  });
}

async function readZarrLevelShapes(zarrPath) {
  const levels = await listZarrLevels(zarrPath);
  const shapes = {};

  await Promise.all(levels.map(async (level) => {
    try {
      const metadata = await readJson(path.join(zarrPath, level, ".zarray"));
      const [z, y, x] = metadata.shape;
      shapes[level] = { z, y, x };
    } catch {
      // Ignore malformed levels here; the loader will surface the error if used.
    }
  }));

  return shapes;
}

function formatMoleculeName(value) {
  return String(value || "unknown molecule")
    .replace(/[-_]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function pickColor(type, index) {
  const palette = [
    "#0ea5a6",
    "#f97316",
    "#7c3aed",
    "#22c55e",
    "#ef4444",
    "#3b82f6",
    "#d946ef",
    "#eab308"
  ];
  return palette[index % palette.length];
}

function pickRadiusAngstrom(type) {
  const radii = {
    "apo-ferritin": 60,
    "beta-amylase": 65,
    "beta-galactosidase": 90,
    "ribosome": 150,
    "thyroglobulin": 130,
    "virus-like-particle": 135
  };
  return radii[type] || 80;
}

const moleculeDifficulty = {
  "apo-ferritin": "easy",
  "beta-amylase": "impossible, not scored",
  "beta-galactosidase": "hard",
  "ribosome": "easy",
  "thyroglobulin": "hard",
  "virus-like-particle": "easy"
};

function buildMoleculeAggregation(detections, jsonFiles = 0, volume = null) {
  const groups = new Map();
  detections.forEach((detection) => {
    const molecule = detection.molecule || detection.type || "unknown";
    if (!groups.has(molecule)) {
      groups.set(molecule, {
        molecule,
        label: detection.type || formatMoleculeName(molecule),
        color: detection.color || pickColor(molecule, groups.size),
        difficulty: moleculeDifficulty[molecule] || "unknown",
        count: 0
      });
    }
    groups.get(molecule).count += 1;
  });

  const total = detections.length;
  const items = [...groups.values()]
    .sort((a, b) => b.count - a.count)
    .map((item) => {
      const clusterThresholdAngstroms = Math.max(80, pickRadiusAngstrom(item.molecule) * 1.5);
      const clusters = clusterDetections(
        detections.filter((detection) => detection.molecule === item.molecule),
        clusterThresholdAngstroms
      );

      return {
        ...item,
        frequencyPercent: total > 0 ? (item.count / total) * 100 : 0,
        clusterThresholdAngstroms,
        clusterCount: clusters.length,
        singletonClusters: clusters.filter((cluster) => cluster.count === 1).length,
        multiPickClusters: clusters.filter((cluster) => cluster.count > 1).length,
        largestClusterSize: clusters.reduce((max, cluster) => Math.max(max, cluster.count), 0),
        clusters: clusters.map((cluster, clusterIndex) => ({
          id: `${item.molecule}-cluster-${String(clusterIndex + 1).padStart(3, "0")}`,
          count: cluster.count,
          centroid: cluster.centroid,
          bounds: cluster.bounds,
          points: cluster.points
        }))
      };
    });

  return {
    totalPicks: total,
    jsonFiles,
    volume,
    totalClusters: items.reduce((sum, item) => sum + item.clusterCount, 0),
    items
  };
}

function buildVolumeMetadata(shape, spacing) {
  const dimensions = {
    z: Number(shape?.z || 0),
    y: Number(shape?.y || 0),
    x: Number(shape?.x || 0)
  };
  const voxelSpacing = spacing
    ? {
        z: Number(spacing.z || 1),
        y: Number(spacing.y || 1),
        x: Number(spacing.x || 1)
      }
    : { z: 1, y: 1, x: 1 };
  const physicalSizeAngstroms = {
    z: dimensions.z * voxelSpacing.z,
    y: dimensions.y * voxelSpacing.y,
    x: dimensions.x * voxelSpacing.x
  };

  return {
    dimensions,
    voxelSpacing,
    physicalSizeAngstroms
  };
}

function clusterDetections(points, thresholdAngstroms) {
  const clusters = [];
  const visited = new Set();

  function distance(a, b) {
    const dx = a.physical.x - b.physical.x;
    const dy = a.physical.y - b.physical.y;
    const dz = a.physical.z - b.physical.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  for (let index = 0; index < points.length; index += 1) {
    if (visited.has(index)) {
      continue;
    }

    const queue = [index];
    const clusterIndexes = [];
    visited.add(index);

    while (queue.length > 0) {
      const current = queue.pop();
      clusterIndexes.push(current);

      for (let candidate = 0; candidate < points.length; candidate += 1) {
        if (visited.has(candidate)) {
          continue;
        }
        if (distance(points[current], points[candidate]) <= thresholdAngstroms) {
          visited.add(candidate);
          queue.push(candidate);
        }
      }
    }

    const members = clusterIndexes.map((clusterIndex) => points[clusterIndex]);
    const count = members.length;
    const centroid = {
      x: members.reduce((sum, item) => sum + item.physical.x, 0) / count,
      y: members.reduce((sum, item) => sum + item.physical.y, 0) / count,
      z: members.reduce((sum, item) => sum + item.physical.z, 0) / count
    };
    const bounds = members.reduce((acc, item) => {
      acc.min.x = Math.min(acc.min.x, item.physical.x);
      acc.min.y = Math.min(acc.min.y, item.physical.y);
      acc.min.z = Math.min(acc.min.z, item.physical.z);
      acc.max.x = Math.max(acc.max.x, item.physical.x);
      acc.max.y = Math.max(acc.max.y, item.physical.y);
      acc.max.z = Math.max(acc.max.z, item.physical.z);
      return acc;
    }, {
      min: { x: Number.POSITIVE_INFINITY, y: Number.POSITIVE_INFINITY, z: Number.POSITIVE_INFINITY },
      max: { x: Number.NEGATIVE_INFINITY, y: Number.NEGATIVE_INFINITY, z: Number.NEGATIVE_INFINITY }
    });

    clusters.push({
      count,
      centroid,
      bounds,
      points: members.map((member) => ({
        id: member.id,
        voxel: member.voxel,
        physical: member.physical
      }))
    });
  }

  clusters.sort((a, b) => b.count - a.count);
  return clusters;
}

function buildLocalAnalysisSummary(aggregation) {
  if (!aggregation.items.length) {
    return "No uploaded molecule labels were available for analysis.";
  }

  const dominant = aggregation.items[0];
  const hardItems = aggregation.items.filter((item) => item.difficulty.includes("hard"));
  const easyItems = aggregation.items.filter((item) => item.difficulty === "easy");

  return [
    "### Dataset Composition",
    `The uploaded labels contain ${aggregation.totalPicks.toLocaleString()} curated molecule picks across ${aggregation.items.length} scored molecule classes. The most frequent class is ${dominant.label} with ${dominant.count.toLocaleString()} picks (${dominant.frequencyPercent.toFixed(1)}%).`,
    "### Biological Signal",
    "The frequencies can support dataset-level quality control and hypothesis generation, but they should not be treated as direct drug-response evidence by themselves. CryoET pick counts are affected by sample preparation, annotation policy, tomogram coverage, particle visibility, and scoring difficulty.",
    "### Downstream Use",
    `Easy classes (${easyItems.map((item) => item.label).join(", ") || "none"}) are useful as positive controls for viewer alignment and label quality. Hard classes (${hardItems.map((item) => item.label).join(", ") || "none"}) are better interpreted cautiously and may be useful for benchmarking model sensitivity.`,
    "### Drug Discovery Relevance",
    "This summary can help prioritize downstream review by showing which complexes are abundant or sparse in the uploaded run. To connect this to drug mechanism, the counts would need comparison against matched control/treatment tomograms, replicate runs, and normalized acquisition volume."
  ].join("\n\n");
}

function extractClaudeText(payload) {
  const parts = [];
  for (const content of payload?.content || []) {
    if (typeof content?.text === "string") {
      parts.push(content.text);
    }
  }
  return parts.join("\n").trim();
}

function normalizeInsightTone(value) {
  const tone = String(value || "analytical").toLowerCase();
  if (tone === "positive" || tone === "warning" || tone === "serious") {
    return tone;
  }
  return "analytical";
}

function buildStructuredFallbackAnalysis(aggregation) {
  if (!aggregation.items.length) {
    return {
      title: "CryoSight Analysis",
      headline: "No molecule labels loaded.",
      keywords: [],
      insights: [],
      spatialAnalysis: "",
      biologicalContext: "",
      datasetSummary: "No molecule label data was available for analysis.",
      caveats: [],
      nextSteps: ["Load a Picks labels folder to generate molecule-level insights."]
    };
  }

  const dominant = aggregation.items[0];
  const scored = aggregation.items.filter((i) => i.difficulty !== "impossible, not scored");
  const easyItems = aggregation.items.filter((i) => i.difficulty === "easy");
  const hardItems = aggregation.items.filter((i) => i.difficulty.includes("hard"));
  const decoyItems = aggregation.items.filter((i) => i.difficulty === "impossible, not scored");
  const vol = aggregation.volume?.dimensions;

  const overallSingletonRate = aggregation.items.reduce(
    (sum, i) => sum + (i.clusterCount > 0 ? i.singletonClusters / i.clusterCount : 0), 0
  ) / aggregation.items.length;

  const allSingleton = aggregation.items.every((i) => i.singletonClusters === i.clusterCount);
  const maxClusterSize = Math.max(...aggregation.items.map((i) => i.largestClusterSize));

  const keywords = [
    `${dominant.label.toLowerCase().replace(/\s+/g, "-")}-dominant`,
    overallSingletonRate > 0.95 ? "singleton-saturated" : overallSingletonRate > 0.7 ? "mostly-dispersed" : "clustered",
    `${aggregation.totalPicks}-picks`,
    `${aggregation.items.length}-class-annotation`,
    hardItems.length > 0 ? "hard-class-uncertain" : null,
    decoyItems.length > 0 ? "decoy-class-present" : null,
    easyItems.length > 0 ? "easy-controls-detected" : null,
    maxClusterSize > 2 ? `max-${maxClusterSize}-pick-cluster` : null
  ].filter(Boolean);

  const insights = aggregation.items.map((item) => {
    const singletonRate = item.clusterCount > 0 ? item.singletonClusters / item.clusterCount : 0;
    const isDecoy = item.difficulty === "impossible, not scored";
    const isEasy = item.difficulty === "easy";
    const isHard = item.difficulty.includes("hard");
    const isDominant = item === dominant;

    let tone, label, text;

    if (isDecoy) {
      tone = "serious";
      label = `${item.label}: unscored decoy — excluded from competition metric`;
      text = `Beta-amylase (${item.count} picks, ${item.frequencyPercent.toFixed(1)}%) is an intentional difficulty-calibration decoy not scored in the CZII challenge. Its ${item.clusterCount} fully isolated clusters confirm sample presence, but these picks carry no biological signal for scoring purposes and should not be used for inference.`;
    } else if (isDominant && isEasy) {
      tone = "positive";
      label = `${item.label} dominates — reliable easy-class enrichment`;
      text = `${item.label} is the most abundant annotated molecule at ${item.count} picks (${item.frequencyPercent.toFixed(1)}%). As an easy-class target, these detections carry high confidence. The ${(singletonRate * 100).toFixed(0)}% singleton rate across ${item.clusterCount} clusters indicates particles are spatially dispersed throughout the ${vol ? `${vol.z}×${vol.y}×${vol.x}` : "full"} voxel volume without preferential aggregation.`;
    } else if (isHard && singletonRate >= 1.0) {
      tone = "warning";
      label = `${item.label}: hard-class with total spatial dispersal — annotation-uncertain`;
      text = `All ${item.count} ${item.label} picks (${item.frequencyPercent.toFixed(1)}%) are fully isolated — 100% singleton rate within a ${item.clusterThresholdAngstroms.toFixed(0)}Å clustering threshold. For hard-class molecules, complete dispersal can reflect genuine biological scarcity, conservative annotation posture, or detection difficulty. Without replicate tomograms, these two explanations cannot be distinguished.`;
    } else if (isEasy && singletonRate >= 1.0) {
      tone = "analytical";
      label = `${item.label}: easy-class fully dispersed, no co-localization`;
      text = `${item.label} shows ${item.count} picks (${item.frequencyPercent.toFixed(1)}%), all as singletons within a ${item.clusterThresholdAngstroms.toFixed(0)}Å threshold. As an easy-class molecule, detection confidence is high — the all-singleton pattern suggests particles are genuinely well-separated in this tomogram without spatial preference or aggregation zones.`;
    } else if (item.multiPickClusters > 0) {
      tone = "analytical";
      label = `${item.label}: ${item.multiPickClusters} co-localization cluster${item.multiPickClusters !== 1 ? "s" : ""} detected`;
      text = `${item.label} (${item.count} picks, ${item.frequencyPercent.toFixed(1)}%) has ${item.multiPickClusters} clusters containing multiple picks, with the largest holding ${item.largestClusterSize} particles. This suggests ${item.multiPickClusters > 3 ? "notable spatial enrichment in specific subvolume regions" : "sparse but detectable co-localization"} alongside ${item.singletonClusters} isolated picks.`;
    } else {
      tone = "analytical";
      label = `${item.label}: ${item.frequencyPercent.toFixed(1)}% share, fully dispersed`;
      text = `${item.label} contributes ${item.count} picks (${item.frequencyPercent.toFixed(1)}% of total) across ${item.clusterCount} singleton clusters within a ${item.clusterThresholdAngstroms.toFixed(0)}Å spatial threshold. No particle aggregation was detected.`;
    }

    return {
      tone,
      label,
      text,
      evidence: [
        `${item.count} picks · ${item.frequencyPercent.toFixed(1)}%`,
        `${item.clusterCount} clusters · ${(singletonRate * 100).toFixed(0)}% singletons`,
        item.largestClusterSize > 1 ? `largest cluster: ${item.largestClusterSize} picks` : "all isolated"
      ]
    };
  });

  const spatialAnalysis = allSingleton
    ? `All ${aggregation.totalPicks} picks across all ${aggregation.items.length} molecule classes are fully singleton-clustered — every annotated particle is spatially isolated within its class-specific distance threshold. This near-total singleton pattern (${(overallSingletonRate * 100).toFixed(0)}% average) means no co-localization between same-class particles was detected anywhere in the ${vol ? `${vol.z}×${vol.y}×${vol.x}` : ""} voxel volume. For easy-class molecules this is a robust biological observation; for hard-class molecules it may additionally reflect annotation conservatism. No evidence of multi-particle complexes or spatially enriched regions was found in this volume.`
    : `The spatial distribution is heterogeneous across classes. Classes with multi-pick clusters (${aggregation.items.filter((i) => i.multiPickClusters > 0).map((i) => i.label).join(", ")}) show localized particle enrichment, while others remain fully dispersed. The largest cluster in the dataset contains ${maxClusterSize} picks, suggesting at least one region of local particle concentration. The overall singleton rate of ${(overallSingletonRate * 100).toFixed(0)}% indicates that most particles across all classes are non-aggregated.`;

  const biologicalContext = `This tomogram contains the standard CZII challenge molecule panel. Ribosome abundance is consistent with active translation machinery; apo-ferritin serves as a structural marker with well-characterized detectability. Thyroglobulin and beta-galactosidase at lower frequencies are expected given their harder detection profiles and smaller expected copy numbers. The absence of multi-particle clusters may indicate a sample fixed prior to complex formation, or simply reflects the density at this particular tomographic position. Cross-tomogram comparison is required before drawing sample-level biological conclusions.`;

  return {
    title: "CryoSight Dataset Analysis",
    headline: `${aggregation.totalPicks} picks across ${scored.length} scored class${scored.length !== 1 ? "es" : ""} — ${overallSingletonRate > 0.95 ? "singleton-saturated: no particle aggregation detected" : overallSingletonRate > 0.7 ? "predominantly dispersed with sparse co-localization" : "clustered distribution with multi-particle zones"}.`,
    keywords,
    insights,
    spatialAnalysis,
    biologicalContext,
    datasetSummary: `Tomogram dimensions: ${vol ? `${vol.z}×${vol.y}×${vol.x} voxels` : "unknown"} at ${aggregation.volume?.voxelSpacing?.x || "?"}Å/voxel. Total annotated picks: ${aggregation.totalPicks} across ${aggregation.items.length} molecule classes (${decoyItems.length} unscored decoy). Dominant class: ${dominant.label} at ${dominant.frequencyPercent.toFixed(1)}%. All-singleton classes: ${aggregation.items.filter((i) => i.singletonClusters === i.clusterCount).map((i) => i.label).join(", ") || "none"}. Classes with co-localization: ${aggregation.items.filter((i) => i.multiPickClusters > 0).map((i) => `${i.label} (${i.multiPickClusters} multi-pick clusters)`).join(", ") || "none"}.`,
    caveats: [
      "Results derive from one tomographic volume and cannot be generalized to population-level conclusions without replicate data.",
      overallSingletonRate > 0.9 ? "The high singleton rate may reflect conservative annotation practices or fixed-step particle picking, not necessarily true biological dispersal." : "Multi-pick clusters may arise from close particle packing or annotation overlap.",
      hardItems.length > 0 ? `Hard-class molecules (${hardItems.map((i) => i.label).join(", ")}) carry higher detection uncertainty; their spatial patterns should be interpreted with additional caution.` : null,
      "Beta-amylase picks are not scored in the CZII challenge — exclude from any scoring or biological frequency comparisons."
    ].filter(Boolean),
    nextSteps: [
      "Compare pick frequencies and singleton rates across multiple tomograms from the same experiment run before making biological claims.",
      "Inspect hard-class singleton clusters (beta-galactosidase, thyroglobulin) manually to assess annotation consistency.",
      hardItems.some((i) => i.multiPickClusters > 0) ? "Examine the multi-pick clusters in hard-class molecules — these are the most likely true positive detections for difficult targets." : "Consider lowering cluster thresholds for hard-class molecules to test sensitivity vs. specificity tradeoffs.",
      "Cross-reference ribosome cluster positions with expected membrane-proximal translation zones if membrane segmentation is available."
    ].filter(Boolean)
  };
}

function buildClaudeAnalysisInput(aggregation) {
  const maxItems = 12;
  const maxClustersPerItem = 8;
  return {
    volume: aggregation.volume,
    totalPicks: aggregation.totalPicks,
    totalClusters: aggregation.totalClusters,
    jsonFiles: aggregation.jsonFiles,
    items: aggregation.items.slice(0, maxItems).map((item) => ({
      molecule: item.molecule,
      label: item.label,
      difficulty: item.difficulty,
      count: item.count,
      frequencyPercent: Number(item.frequencyPercent.toFixed(3)),
      clusterThresholdAngstroms: Number(item.clusterThresholdAngstroms.toFixed(3)),
      clusterCount: item.clusterCount,
      singletonClusters: item.singletonClusters,
      multiPickClusters: item.multiPickClusters,
      largestClusterSize: item.largestClusterSize,
      clustersShown: Math.min(item.clusters.length, maxClustersPerItem),
      clustersTotal: item.clusters.length,
      clusters: item.clusters.slice(0, maxClustersPerItem).map((cluster) => ({
        id: cluster.id,
        count: cluster.count,
        centroid: {
          x: Number(cluster.centroid.x.toFixed(3)),
          y: Number(cluster.centroid.y.toFixed(3)),
          z: Number(cluster.centroid.z.toFixed(3))
        },
        bounds: {
          min: {
            x: Number(cluster.bounds.min.x.toFixed(3)),
            y: Number(cluster.bounds.min.y.toFixed(3)),
            z: Number(cluster.bounds.min.z.toFixed(3))
          },
          max: {
            x: Number(cluster.bounds.max.x.toFixed(3)),
            y: Number(cluster.bounds.max.y.toFixed(3)),
            z: Number(cluster.bounds.max.z.toFixed(3))
          }
        }
      }))
    }))
  };
}

function buildClaudeSystemPrompt() {
  return [
    "You are a CryoET data scientist. Analyze particle pick data from one cryo-electron tomography volume.",
    "Every claim must cite specific numbers from the input. No generic filler text.",
    "INSIGHT LABELS: describe a biological pattern or data quality finding — never 'MoleculeName: N picks'.",
    "Good labels: 'Ribosome dominates with singleton-only dispersal', 'Apo-ferritin: easy-class positive control at 100% isolation', 'Beta-galactosidase hard-class signal is annotation-uncertain'.",
    "Singleton rate = singletonClusters / clusterCount. 100% singleton = every annotated particle is spatially isolated, no co-localization.",
    "Easy-class: apo-ferritin, ribosome, VLP — high detection confidence. Hard-class: beta-galactosidase, thyroglobulin — 100% singleton may reflect annotation difficulty, not biology.",
    "Beta-amylase is UNSCORED DECOY — intentional calibration class, excluded from competition metric. Mark tone=serious.",
    "Tones allowed: positive, analytical, warning, serious.",
    "Return ONLY valid compact JSON — no markdown fences, no prose, no trailing commas, no newlines inside string values.",
    "Schema (keep arrays short — max 5 keywords, max 5 insights, max 3 caveats, max 3 nextSteps, max 2 evidence items per insight):",
    "{\"title\":string,\"headline\":string,\"keywords\":[string],\"insights\":[{\"tone\":string,\"label\":string,\"text\":string,\"evidence\":[string]}],\"spatialAnalysis\":string,\"biologicalContext\":string,\"caveats\":[string],\"nextSteps\":[string]}"
  ].join(" ");
}

function buildClaudeUserPrompt(aggregation) {
  const input = buildClaudeAnalysisInput(aggregation);

  const derived = input.items.map((item) => {
    const singletonPct = item.clusterCount > 0
      ? ((item.singletonClusters / item.clusterCount) * 100).toFixed(1)
      : "0.0";
    return `  ${item.label} (${item.difficulty}): ${item.count} picks · ${item.frequencyPercent.toFixed(1)}% · ${item.clusterCount} clusters · ${singletonPct}% singleton rate · largest cluster ${item.largestClusterSize}`;
  });

  const overallSingletonRate = input.items.reduce(
    (sum, item) => sum + (item.clusterCount > 0 ? item.singletonClusters / item.clusterCount : 0), 0
  ) / Math.max(1, input.items.length);

  const vol = input.volume?.dimensions;
  const voxelSpacing = input.volume?.voxelSpacing?.x || 10;
  const volumeNm3 = vol
    ? ((vol.z * vol.y * vol.x * Math.pow(voxelSpacing, 3)) / 1e9).toFixed(1)
    : "unknown";

  return [
    `Analyze CryoET particle pick data from one tomographic volume.`,
    `Volume: ${vol ? `${vol.z}×${vol.y}×${vol.x} voxels` : "unknown"} at ${voxelSpacing}Å/voxel ≈ ${volumeNm3} nm³.`,
    `Total picks: ${input.totalPicks} across ${input.items.length} classes.`,
    `Overall average singleton rate across all classes: ${(overallSingletonRate * 100).toFixed(1)}%.`,
    ``,
    `Per-class breakdown (pre-computed for accuracy):`,
    ...derived,
    ``,
    `Classes fully at 100% singleton rate: ${input.items.filter((i) => i.singletonClusters === i.clusterCount).map((i) => i.label).join(", ") || "none"}.`,
    `Classes with multi-pick clusters: ${input.items.filter((i) => i.multiPickClusters > 0).map((i) => `${i.label} (${i.multiPickClusters} clusters, largest ${i.largestClusterSize})`).join(", ") || "none"}.`,
    ``,
    `Generate a specific, data-driven interpretation. Reference exact numbers. Do not use molecule names as labels.`
  ].join("\n");
}

function parseClaudeAnalysisJson(text) {
  const cleaned = text.trim();
  if (!cleaned) {
    return null;
  }

  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    return null;
  }

  const candidate = cleaned.slice(start, end + 1);
  try {
    const parsed = JSON.parse(candidate);
    const insights = Array.isArray(parsed.insights) ? parsed.insights.map((insight) => ({
      tone: normalizeInsightTone(insight.tone),
      label: String(insight.label || "").trim(),
      text: String(insight.text || "").trim(),
      evidence: Array.isArray(insight.evidence) ? insight.evidence.map((entry) => String(entry)) : []
    })).filter((insight) => insight.label || insight.text) : [];

    return {
      title: String(parsed.title || "CryoSight Analysis").trim(),
      headline: String(parsed.headline || "").trim(),
      keywords: Array.isArray(parsed.keywords)
        ? parsed.keywords.map((k) => String(k).trim()).filter(Boolean)
        : [],
      insights,
      spatialAnalysis: String(parsed.spatialAnalysis || "").trim(),
      biologicalContext: String(parsed.biologicalContext || "").trim(),
      datasetSummary: String(parsed.datasetSummary || "").trim(),
      caveats: Array.isArray(parsed.caveats) ? parsed.caveats.map((entry) => String(entry)).filter(Boolean) : [],
      nextSteps: Array.isArray(parsed.nextSteps) ? parsed.nextSteps.map((entry) => String(entry)).filter(Boolean) : []
    };
  } catch {
    return null;
  }
}

function structuredAnalysisToMarkdown(structured) {
  if (!structured) {
    return "";
  }

  const blocks = [];
  if (structured.title) {
    blocks.push(`### ${structured.title}`);
  }
  if (structured.headline) {
    blocks.push(structured.headline);
  }
  if (structured.keywords?.length) {
    blocks.push(`**Tags:** ${structured.keywords.join(" · ")}`);
  }
  if (structured.datasetSummary) {
    blocks.push(`### Dataset Composition\n${structured.datasetSummary}`);
  }
  if (structured.insights?.length) {
    const lines = structured.insights.map((insight) => `- [${insight.tone}] ${insight.label}: ${insight.text}`);
    blocks.push(`### Insights\n${lines.join("\n")}`);
  }
  if (structured.spatialAnalysis) {
    blocks.push(`### Spatial Distribution\n${structured.spatialAnalysis}`);
  }
  if (structured.biologicalContext) {
    blocks.push(`### Biological Context\n${structured.biologicalContext}`);
  }
  if (structured.caveats?.length) {
    blocks.push(`### Caveats\n${structured.caveats.map((item) => `- ${item}`).join("\n")}`);
  }
  if (structured.nextSteps?.length) {
    blocks.push(`### Next Validation Steps\n${structured.nextSteps.map((item) => `- ${item}`).join("\n")}`);
  }
  return blocks.join("\n\n");
}

async function generateClaudeAnalysis(aggregation, timeoutMs = 45000) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    const structured = buildStructuredFallbackAnalysis(aggregation);
    return {
      structured,
      report: structuredAnalysisToMarkdown(structured),
      reportStatus: "Ready"
    };
  }

  const model = process.env.ANTHROPIC_MODEL || "claude-sonnet-4-6";

  // AbortController with 15s timeout to prevent hanging
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  let response;
  try {
    response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      signal: controller.signal,
      headers: {
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model,
        max_tokens: 4096,
        temperature: 0.2,
        system: buildClaudeSystemPrompt(),
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: buildClaudeUserPrompt(aggregation)
              }
            ]
          }
        ]
      })
    });
  } catch (fetchErr) {
    clearTimeout(timeoutId);
    const structured = buildStructuredFallbackAnalysis(aggregation);
    return {
      structured,
      report: structuredAnalysisToMarkdown(structured),
      reportStatus: "Ready",
      reportWarning: fetchErr.name === "AbortError" ? "Claude API request timed out; using local analysis." : `Claude API unavailable; using local analysis. ${fetchErr.message}`
    };
  }
  clearTimeout(timeoutId);

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    let message = body || `Claude request failed with HTTP ${response.status}`;
    try {
      const parsed = JSON.parse(body);
      if (parsed?.error?.message) {
        message = parsed.error.message;
      }
    } catch {
      // Keep the raw response excerpt.
    }
    const structured = buildStructuredFallbackAnalysis(aggregation);
    return {
      structured,
      report: structuredAnalysisToMarkdown(structured),
      reportStatus: "Ready",
      reportWarning: `Claude API returned HTTP ${response.status}; using local analysis.`,
      reportErrorStatus: response.status
    };
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    const structured = buildStructuredFallbackAnalysis(aggregation);
    return {
      structured,
      report: structuredAnalysisToMarkdown(structured),
      reportStatus: "Ready",
      reportWarning: "Claude API returned invalid JSON; using local analysis."
    };
  }
  const structured = parseClaudeAnalysisJson(extractClaudeText(payload)) || buildStructuredFallbackAnalysis(aggregation);
  return {
    structured,
    report: structuredAnalysisToMarkdown(structured) || extractClaudeText(payload) || buildLocalAnalysisSummary(aggregation),
    reportStatus: "Ready"
  };
}

function parseVoxelSpacingFromPath(zarrPath) {
  const match = zarrPath.match(/VoxelSpacing([0-9.]+)/i);
  const parsed = match ? Number.parseFloat(match[1]) : NaN;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

async function readLevelZeroSpacing(zarrPath) {
  const attrs = await readJson(path.join(zarrPath, ".zattrs")).catch(() => null);
  const datasets = attrs?.multiscales?.[0]?.datasets || [];
  const levelZero = datasets.find((dataset) => dataset.path === "0") || datasets[0];
  const scale = levelZero?.coordinateTransformations?.find((item) => item.type === "scale")?.scale;
  if (Array.isArray(scale) && scale.length >= 3 && scale.every((value) => Number.isFinite(Number(value)) && Number(value) > 0)) {
    return { z: Number(scale[0]), y: Number(scale[1]), x: Number(scale[2]) };
  }

  const fallback = parseVoxelSpacingFromPath(zarrPath) || 1;
  return { z: fallback, y: fallback, x: fallback };
}

function findPicksPathForZarr(zarrPath) {
  const parts = zarrPath.split(path.sep);
  const staticIndex = parts.lastIndexOf("static");
  const experimentIndex = parts.lastIndexOf("ExperimentRuns");
  if (staticIndex === -1 || experimentIndex === -1 || experimentIndex + 1 >= parts.length) {
    return null;
  }

  const runName = parts[experimentIndex + 1];
  const root = parts.slice(0, staticIndex).join(path.sep) || path.sep;
  const picksPath = path.join(root, "overlay", "ExperimentRuns", runName, "Picks");
  return fs.existsSync(picksPath) ? picksPath : null;
}

async function loadPickDetections(zarrPath, levelZeroShape) {
  const picksPath = findPicksPathForZarr(zarrPath);
  if (!picksPath) {
    return [];
  }

  const cacheKey = `${picksPath}:${levelZeroShape.x}x${levelZeroShape.y}x${levelZeroShape.z}`;
  if (pickCache.has(cacheKey)) {
    return pickCache.get(cacheKey);
  }

  const entries = await fsp.readdir(picksPath, { withFileTypes: true }).catch(() => []);
  const files = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
    .map((entry) => path.join(picksPath, entry.name))
    .sort();
  const detections = await buildPickDetectionsFromFiles(zarrPath, levelZeroShape, files, path.basename(path.dirname(path.dirname(picksPath))));

  pickCache.set(cacheKey, detections);
  return detections;
}

async function buildPickDetectionsFromFiles(zarrPath, levelZeroShape, files, volumeLabel = "uploaded labels") {
  const spacing = await readLevelZeroSpacing(zarrPath);
  const detections = [];

  for (const [fileIndex, filePath] of files.entries()) {
    const payload = await readJson(filePath).catch(() => null);
    const points = Array.isArray(payload?.points) ? payload.points : [];
    const type = payload?.pickable_object_name || path.basename(filePath, ".json");
    const label = formatMoleculeName(type);
    const color = pickColor(type, fileIndex);
    const radiusAngstrom = pickRadiusAngstrom(type);
    const radiusVoxel = radiusAngstrom / ((spacing.x + spacing.y + spacing.z) / 3);

    points.forEach((point, pointIndex) => {
      const location = point.location || {};
      const voxel = {
        x: Number(location.x) / spacing.x,
        y: Number(location.y) / spacing.y,
        z: Number(location.z) / spacing.z
      };

      if (!Number.isFinite(voxel.x) || !Number.isFinite(voxel.y) || !Number.isFinite(voxel.z)) {
        return;
      }

      const clipped = {
        x: Math.max(0, Math.min(levelZeroShape.x - 1, voxel.x)),
        y: Math.max(0, Math.min(levelZeroShape.y - 1, voxel.y)),
        z: Math.max(0, Math.min(levelZeroShape.z - 1, voxel.z))
      };

      detections.push({
        id: `${type}-${String(pointIndex + 1).padStart(3, "0")}`,
        type: label,
        molecule: type,
        confidence: "curated pick",
        volume: payload?.run_name || volumeLabel,
        color,
        radiusAngstrom,
        radiusVoxel,
        coords: [
          (clipped.x / Math.max(1, levelZeroShape.x - 1)) - 0.5,
          (clipped.y / Math.max(1, levelZeroShape.y - 1)) - 0.5,
          (clipped.z / Math.max(1, levelZeroShape.z - 1)) - 0.5
        ],
        voxel: {
          x: clipped.x,
          y: clipped.y,
          z: clipped.z
        },
        physical: {
          x: Number(location.x),
          y: Number(location.y),
          z: Number(location.z),
          unit: payload?.unit || "angstrom"
        },
        position: `X ${Math.round(clipped.x)}, Y ${Math.round(clipped.y)}, Z ${Math.round(clipped.z)}`,
        notes: `${label} curated overlay pick from ${path.basename(filePath)}. Physical position ${Math.round(Number(location.x))}, ${Math.round(Number(location.y))}, ${Math.round(Number(location.z))} ${payload?.unit || "angstrom"}.`
      });
    });
  }

  return detections;
}

function scanPayload(id, name, scanPath, levels) {
  return {
    id,
    name,
    path: id,
    levels,
    defaultLevel: levels[0] || "0",
    metadataMissing: levels.length === 0
  };
}

async function findZarrs(rootDir, baseLabel) {
  const results = [];

  async function walk(dir, depth) {
    if (depth > 8) {
      return;
    }

    let entries;
    try {
      entries = await fsp.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }

    const hasZgroup = entries.some((entry) => entry.name === ".zgroup");
    const hasZarrJson = entries.some((entry) => entry.name === "zarr.json");

    if (dir.endsWith(".zarr") || hasZgroup || hasZarrJson) {
      const relativePath = path.relative(__dirname, dir);
      results.push(scanPayload(relativePath, `${baseLabel}: ${path.basename(dir)}`, dir, await listZarrLevels(dir)));
      return;
    }

    await Promise.all(
      entries
        .filter((entry) => entry.isDirectory() && !entry.name.startsWith("."))
        .map((entry) => walk(path.join(dir, entry.name), depth + 1))
    );
  }

  await walk(rootDir, 0);
  return results;
}

async function readJson(filePath) {
  return JSON.parse(await fsp.readFile(filePath, "utf8"));
}

function decodeLz4Block(source, expectedLength) {
  const output = Buffer.allocUnsafe(expectedLength);
  let ip = 0;
  let op = 0;

  while (ip < source.length) {
    const token = source[ip++];
    let literalLength = token >> 4;

    if (literalLength === 15) {
      let value;
      do {
        value = source[ip++];
        literalLength += value;
      } while (value === 255);
    }

    source.copy(output, op, ip, ip + literalLength);
    ip += literalLength;
    op += literalLength;

    if (ip >= source.length) {
      break;
    }

    const offset = source[ip] | (source[ip + 1] << 8);
    ip += 2;

    let matchLength = token & 15;
    if (matchLength === 15) {
      let value;
      do {
        value = source[ip++];
        matchLength += value;
      } while (value === 255);
    }
    matchLength += 4;

    const matchStart = op - offset;
    for (let index = 0; index < matchLength; index += 1) {
      output[op + index] = output[matchStart + index];
    }
    op += matchLength;
  }

  return output;
}

function unshuffleBytes(source, typeSize) {
  if (typeSize <= 1) {
    return source;
  }

  const output = Buffer.allocUnsafe(source.length);
  const elementCount = Math.floor(source.length / typeSize);
  const leftoverOffset = elementCount * typeSize;

  for (let element = 0; element < elementCount; element += 1) {
    for (let byte = 0; byte < typeSize; byte += 1) {
      output[element * typeSize + byte] = source[byte * elementCount + element];
    }
  }

  if (leftoverOffset < source.length) {
    source.copy(output, leftoverOffset, leftoverOffset);
  }

  return output;
}

async function decodeBloscChunk(chunkPath) {
  const compressed = await fsp.readFile(chunkPath);
  const flags = compressed[2];
  const typeSize = compressed[3];
  const nbytes = compressed.readUInt32LE(4);
  const blockSize = compressed.readUInt32LE(8);
  const cbytes = compressed.readUInt32LE(12);
  const blockCount = Math.ceil(nbytes / blockSize);
  const output = Buffer.allocUnsafe(nbytes);
  let written = 0;

  for (let blockIndex = 0; blockIndex < blockCount; blockIndex += 1) {
    const dataOffset = compressed.readUInt32LE(16 + blockIndex * 4);
    const nextOffset = blockIndex === blockCount - 1 ? cbytes : compressed.readUInt32LE(16 + (blockIndex + 1) * 4);
    const packedSize = nextOffset - dataOffset;
    const remaining = nbytes - written;
    const expectedSize = Math.min(blockSize, remaining);

    let block;
    if ((flags & 4) === 4 || packedSize === expectedSize) {
      block = compressed.subarray(dataOffset, dataOffset + expectedSize);
    } else {
      block = decodeLz4Block(compressed.subarray(dataOffset, dataOffset + packedSize), expectedSize);
    }

    if ((flags & 1) === 1) {
      block = unshuffleBytes(block, typeSize);
    }

    block.copy(output, written, 0, expectedSize);
    written += expectedSize;
  }

  return output;
}

async function loadVolumePreview(zarrPath, levelName = "2") {
  let resolvedLevel = levelName;
  if (!fs.existsSync(path.join(zarrPath, resolvedLevel, ".zarray"))) {
    const levels = await listZarrLevels(zarrPath);
    if (levels.length === 0) {
      throw new Error(`No Zarr array metadata found in ${path.basename(zarrPath)}. The folder must include hidden metadata files such as .zarray/.zattrs or zarr.json; browser folder upload may omit these dotfiles. Use "Open local .zarr path" instead.`);
    }
    resolvedLevel = levels.includes(levelName) ? levelName : levels[0];
  }

  const cacheKey = `${zarrPath}:${resolvedLevel}`;
  if (volumeCache.has(cacheKey)) {
    return volumeCache.get(cacheKey);
  }
  if (volumeLoadCache.has(cacheKey)) {
    return volumeLoadCache.get(cacheKey);
  }

  const loadPromise = loadVolumeData(zarrPath, resolvedLevel, cacheKey);
  volumeLoadCache.set(cacheKey, loadPromise);
  try {
    return await loadPromise;
  } finally {
    volumeLoadCache.delete(cacheKey);
  }
}

async function loadVolumeData(zarrPath, resolvedLevel, cacheKey) {
  const arrayPath = path.join(zarrPath, resolvedLevel);
  const metadata = await readJson(path.join(arrayPath, ".zarray"));
  const [depth, height, width] = metadata.shape;
  const [chunkDepth, chunkHeight, chunkWidth] = metadata.chunks;
  const raw = Buffer.alloc(width * height * depth * 4);

  for (let zChunk = 0; zChunk < Math.ceil(depth / chunkDepth); zChunk += 1) {
    for (let yChunk = 0; yChunk < Math.ceil(height / chunkHeight); yChunk += 1) {
      for (let xChunk = 0; xChunk < Math.ceil(width / chunkWidth); xChunk += 1) {
        const chunkPath = path.join(arrayPath, String(zChunk), String(yChunk), String(xChunk));
        if (!fs.existsSync(chunkPath)) {
          continue;
        }

        const decoded = await decodeBloscChunk(chunkPath);
        const chunkZ = Math.min(chunkDepth, depth - zChunk * chunkDepth);
        const chunkY = Math.min(chunkHeight, height - yChunk * chunkHeight);
        const chunkX = Math.min(chunkWidth, width - xChunk * chunkWidth);

        for (let z = 0; z < chunkZ; z += 1) {
          for (let y = 0; y < chunkY; y += 1) {
            const sourceOffset = ((z * chunkHeight * chunkWidth) + (y * chunkWidth)) * 4;
            const targetOffset = (((zChunk * chunkDepth + z) * height * width) + ((yChunk * chunkHeight + y) * width) + (xChunk * chunkWidth)) * 4;
            decoded.copy(raw, targetOffset, sourceOffset, sourceOffset + chunkX * 4);
          }
        }
      }
    }
  }

  const volume = {
    level: resolvedLevel,
    metadata,
    data: new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4),
    shape: { z: depth, y: height, x: width }
  };

  volumeCache.set(cacheKey, volume);
  return volume;
}

function buildPointCloud(volume, requestedStride = 2, requestedLimit = 38000) {
  const { data, shape } = volume;
  const stride = Math.max(1, Math.min(8, requestedStride));
  const limit = Math.max(1000, Math.min(90000, requestedLimit));
  const sampled = [];

  for (let z = 0; z < shape.z; z += stride) {
    for (let y = 0; y < shape.y; y += stride) {
      for (let x = 0; x < shape.x; x += stride) {
        const value = data[z * shape.y * shape.x + y * shape.x + x];
        if (!Number.isFinite(value) || Math.abs(value) > 1_000_000) {
          continue;
        }

        sampled.push({ x, y, z, value });
      }
    }
  }

  const sortedValues = sampled.map((point) => point.value).sort((a, b) => a - b);
  const low = sortedValues[Math.floor(sortedValues.length * 0.01)] || 0;
  const high = sortedValues[Math.floor(sortedValues.length * 0.99)] || 1;
  const center = (low + high) / 2;
  const spread = Math.max(0.000001, high - low);
  const ranked = sampled
    .map((point) => ({
      ...point,
      score: Math.min(1, Math.abs(point.value - center) / spread)
    }))
    .sort((a, b) => b.score - a.score);
  const strongest = ranked.slice(0, limit);
  const scoreMax = strongest.reduce((current, point) => Math.max(current, point.score), 0) || 1;

  return {
    shape,
    stats: { min: low, max: high, points: strongest.length, stride },
    points: strongest.map((point) => [
      Number(((point.x / Math.max(1, shape.x - 1)) - 0.5).toFixed(5)),
      Number(((point.y / Math.max(1, shape.y - 1)) - 0.5).toFixed(5)),
      Number(((point.z / Math.max(1, shape.z - 1)) - 0.5).toFixed(5)),
      Number((point.score / scoreMax).toFixed(5)),
      Number(Math.max(0, Math.min(1, (point.value - low) / spread)).toFixed(5))
    ])
  };
}

function buildNrrdVolume(volume) {
  const { data, shape } = volume;
  const values = [];

  for (let index = 0; index < data.length; index += 1) {
    const value = data[index];
    if (Number.isFinite(value) && Math.abs(value) <= 1_000_000) {
      values.push(value);
    }
  }

  values.sort((a, b) => a - b);
  const low = values[Math.floor(values.length * 0.01)] || 0;
  const high = values[Math.floor(values.length * 0.99)] || 1;
  const range = high - low || 1;
  const voxels = Buffer.alloc(shape.x * shape.y * shape.z);

  for (let index = 0; index < data.length; index += 1) {
    const value = data[index];
    const safeValue = Number.isFinite(value) && Math.abs(value) <= 1_000_000 ? value : low;
    voxels[index] = Math.max(0, Math.min(255, Math.round(((safeValue - low) / range) * 255)));
  }

  const header = [
    "NRRD0005",
    "# QBI export for 3D Slicer",
    "type: uchar",
    "dimension: 3",
    "space: left-posterior-superior",
    `sizes: ${shape.x} ${shape.y} ${shape.z}`,
    "space directions: (1,0,0) (0,1,0) (0,0,1)",
    "kinds: domain domain domain",
    "encoding: raw",
    "endian: little",
    "space origin: (0,0,0)",
    "",
    ""
  ].join("\n");

  return Buffer.concat([Buffer.from(header, "ascii"), voxels]);
}

function slicerExportName(scanPath, level) {
  const scanName = safeSegment(path.basename(scanPath || "scan").replace(/\.zarr$/i, ""));
  return `${scanName}-level-${safeSegment(level)}.nrrd`;
}

async function writeSlicerExport(zarrPath, level) {
  const volume = await loadVolumePreview(zarrPath, level);
  const filename = slicerExportName(zarrPath, level);
  const body = buildNrrdVolume(volume);
  await fsp.mkdir(slicerExportDir, { recursive: true });
  const filePath = path.join(slicerExportDir, filename);
  await fsp.writeFile(filePath, body);
  return { body, filename, filePath };
}

async function findSlicerExecutable() {
  if (process.env.SLICER_PATH && fs.existsSync(process.env.SLICER_PATH)) {
    return process.env.SLICER_PATH;
  }

  if (process.platform === "darwin") {
    const roots = ["/Applications", path.join(os.homedir(), "Applications")];
    for (const root of roots) {
      let entries = [];
      try {
        entries = await fsp.readdir(root);
      } catch {
        continue;
      }

      const appName = entries.find((entry) => /^Slicer.*\.app$/i.test(entry));
      if (appName) {
        const executable = path.join(root, appName, "Contents", "MacOS", "Slicer");
        if (fs.existsSync(executable)) {
          return executable;
        }
      }
    }
  }

  const candidates = process.platform === "win32"
    ? [
        "C:\\Program Files\\Slicer 5.12.0\\Slicer.exe",
        "C:\\Program Files\\Slicer\\Slicer.exe"
      ]
    : [
        "/usr/local/bin/Slicer",
        "/usr/bin/Slicer",
        "/opt/Slicer/Slicer"
      ];

  return candidates.find((candidate) => fs.existsSync(candidate)) || null;
}

async function openInSlicer(filePath) {
  const slicerExecutable = await findSlicerExecutable();
  if (!slicerExecutable) {
    return false;
  }

  const child = spawn(slicerExecutable, [filePath], {
    detached: true,
    stdio: "ignore"
  });
  child.unref();
  return true;
}

function renderPythonZarrSlice(zarrPath, level, axis, requestedIndex, maxSize) {
  const script = `
import json
import sys

import numpy as np
import zarr

store_path, level, axis, requested_index, requested_max_size = sys.argv[1:6]
requested_index = int(requested_index)
requested_max_size = int(requested_max_size)

root = zarr.open(store_path, mode="r")
volume = root if hasattr(root, "shape") else root[level]
z_count, y_count, x_count = volume.shape
shape = {"z": z_count, "y": y_count, "x": x_count}
axis_size = shape[axis]
slice_index = max(0, min(axis_size - 1, requested_index))

if axis == "x":
    image = np.asarray(volume[:, :, slice_index])
elif axis == "y":
    image = np.asarray(volume[:, slice_index, :])
else:
    image = np.asarray(volume[slice_index, :, :])

source_height, source_width = image.shape
max_size = max(64, min(1024, requested_max_size or max(source_width, source_height)))
scale = min(1.0, max_size / max(source_width, source_height))
width = max(1, int(round(source_width * scale)))
height = max(1, int(round(source_height * scale)))

if width != source_width or height != source_height:
    y_indices = np.linspace(0, source_height - 1, height).astype(np.int64)
    x_indices = np.linspace(0, source_width - 1, width).astype(np.int64)
    image = image[np.ix_(y_indices, x_indices)]

finite = np.isfinite(image) & (np.abs(image) <= 1_000_000)
if np.any(finite):
    low, high = np.percentile(image[finite], [1, 99])
else:
    low, high = 0.0, 1.0
if not np.isfinite(high - low) or high == low:
    high = low + 1.0

safe = np.where(finite, image, low)
pixels = np.clip(np.rint(((safe - low) / (high - low)) * 255), 0, 255).astype(np.uint8)
meta = {
    "axis": axis,
    "level": level,
    "index": slice_index,
    "width": int(width),
    "height": int(height),
    "sourceWidth": int(source_width),
    "sourceHeight": int(source_height),
    "depth": int(axis_size),
    "low": float(low),
    "high": float(high),
}
sys.stderr.write("__QBI_META__" + json.dumps(meta) + "\\n")
sys.stdout.buffer.write(pixels.tobytes(order="C"))
`;

  return new Promise((resolve, reject) => {
    const child = spawn(pythonExecutable(), [
      "-c",
      script,
      zarrPath,
      level,
      axis,
      String(requestedIndex),
      String(maxSize)
    ], {
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        MPLCONFIGDIR: path.join(os.tmpdir(), "qbi-matplotlib")
      }
    });

    const stdout = [];
    const stderr = [];
    child.stdout.on("data", (chunk) => stdout.push(chunk));
    child.stderr.on("data", (chunk) => stderr.push(chunk));
    child.on("error", reject);
    child.on("close", (code) => {
      const errorText = Buffer.concat(stderr).toString("utf8");
      if (code !== 0) {
        reject(new Error(errorText || `Python slice renderer exited with code ${code}`));
        return;
      }

      const metaLine = errorText.split(/\r?\n/).find((line) => line.startsWith("__QBI_META__"));
      if (!metaLine) {
        reject(new Error("Python slice renderer did not return metadata"));
        return;
      }

      resolve({
        meta: JSON.parse(metaLine.replace("__QBI_META__", "")),
        pixels: Buffer.concat(stdout)
      });
    });
  });
}

function sliceCacheKey(zarrPath, level, axis, index, maxSize) {
  return `${zarrPath}:${level}:${axis}:${index}:${maxSize}`;
}

function getCachedSlice(key) {
  const cached = sliceCache.get(key);
  if (!cached) {
    return null;
  }
  sliceCache.delete(key);
  sliceCache.set(key, cached);
  return cached;
}

function setCachedSlice(key, value) {
  if (sliceCache.has(key)) {
    sliceCacheBytes -= sliceCache.get(key).pixels.length;
    sliceCache.delete(key);
  }

  sliceCache.set(key, value);
  sliceCacheBytes += value.pixels.length;

  while (sliceCacheBytes > maxSliceCacheBytes && sliceCache.size > 0) {
    const oldestKey = sliceCache.keys().next().value;
    const oldest = sliceCache.get(oldestKey);
    sliceCacheBytes -= oldest.pixels.length;
    sliceCache.delete(oldestKey);
  }
}

async function loadCachedPythonZarrSlice(zarrPath, level, axis, index, maxSize) {
  const key = sliceCacheKey(zarrPath, level, axis, index, maxSize);
  const cached = getCachedSlice(key);
  if (cached) {
    return cached;
  }
  if (sliceLoadCache.has(key)) {
    return sliceLoadCache.get(key);
  }

  const promise = renderPythonZarrSlice(zarrPath, level, axis, index, maxSize)
    .then((result) => {
      setCachedSlice(key, result);
      return result;
    })
    .finally(() => {
      sliceLoadCache.delete(key);
    });
  sliceLoadCache.set(key, promise);
  return promise;
}

async function handleApi(req, res) {
  const url = new URL(req.url, `http://localhost:${port}`);

  if (req.method === "GET" && url.pathname === "/api/scans") {
    const [sampleScans, uploadedScans] = await Promise.all([
      findZarrs(sampleDir, "Sample"),
      findZarrs(uploadsDir, "Uploaded")
    ]);
    const localScanList = await Promise.all(
      [...localScans.entries()].map(async ([id, scanPath]) => scanPayload(id, `Local: ${path.basename(scanPath)}`, scanPath, await listZarrLevels(scanPath)))
    );
    sendJson(res, 200, { scans: [...sampleScans, ...uploadedScans, ...localScanList] });
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/local-zarr") {
    const body = await readRequestJson(req);
    const requestedPath = path.resolve(String(body.path || ""));
    const stats = await fsp.stat(requestedPath).catch(() => null);
    if (!stats?.isDirectory()) {
      sendJson(res, 400, { error: "Local path must point to an existing .zarr directory" });
      return;
    }

    const levels = await listZarrLevels(requestedPath);
    if (levels.length === 0) {
      sendJson(res, 400, { error: "No .zarray metadata files were found in that .zarr directory" });
      return;
    }

    const id = `local:${Buffer.from(requestedPath).toString("base64url")}`;
    localScans.set(id, requestedPath);
    sendJson(res, 200, { scan: scanPayload(id, `Local: ${path.basename(requestedPath)}`, requestedPath, levels) });
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/zarr/metadata") {
    const zarrPath = resolveScanPath(url.searchParams.get("path"));
    if (!zarrPath) {
      sendJson(res, 400, { error: "Invalid scan path" });
      return;
    }

    const attrs = await readJson(path.join(zarrPath, ".zattrs"));
    const arrays = await Promise.all(
      attrs.multiscales[0].datasets.map(async (dataset) => ({
        path: dataset.path,
        transform: dataset.coordinateTransformations?.[0] || null,
        array: await readJson(path.join(zarrPath, dataset.path, ".zarray"))
      }))
    );
    sendJson(res, 200, { attrs, arrays });
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/zarr/preview") {
    const zarrPath = resolveScanPath(url.searchParams.get("path"));
    if (!zarrPath) {
      sendJson(res, 400, { error: "Invalid scan path" });
      return;
    }

    const level = url.searchParams.get("level") || "2";
    const stride = Number.parseInt(url.searchParams.get("stride") || "2", 10);
    const levels = await listZarrLevels(zarrPath);
    const levelShapes = await readZarrLevelShapes(zarrPath);
    const resolvedLevel = levelShapes[level] ? level : levels[0];
    const shape = levelShapes[resolvedLevel];
    if (!shape) {
      sendJson(res, 400, { error: `No Zarr array metadata found in ${path.basename(zarrPath)}` });
      return;
    }
    const levelZeroShape = levelShapes["0"] || shape;
    const pickDetections = await loadPickDetections(zarrPath, levelZeroShape);
    const spacing = await readLevelZeroSpacing(zarrPath);
    const aggregation = buildMoleculeAggregation(pickDetections, 0, buildVolumeMetadata(levelZeroShape, spacing));
    latestAggregation.set(zarrPath, aggregation);

    sendJson(res, 200, {
      level: resolvedLevel,
      levels,
      levelShapes,
      shape,
      stats: { min: 0, max: 0, points: 0, stride },
      points: [],
      detections: pickDetections,
      analysis: {
        aggregation,
        structured: buildStructuredFallbackAnalysis(aggregation),
        reportStatus: pickDetections.length > 0 ? "Generating..." : "Waiting for labels"
      }
    });
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/zarr/slice") {
    try {
      const zarrPath = resolveScanPath(url.searchParams.get("path"));
      if (!zarrPath) {
        sendJson(res, 400, { error: "Invalid scan path" });
        return;
      }

      const axis = ["x", "y", "z"].includes(url.searchParams.get("axis")) ? url.searchParams.get("axis") : "z";
      const level = url.searchParams.get("level") || "2";
      const sourceLevel = url.searchParams.get("sourceLevel") || level;
      const sourceIndex = Number.parseInt(url.searchParams.get("index") || url.searchParams.get(axis) || url.searchParams.get("z") || "0", 10);
      const maxSize = Math.max(64, Math.min(1024, Number.parseInt(url.searchParams.get("maxSize") || "0", 10) || 1024));
      let sourceShape = null;

      if (sourceLevel !== level) {
        sourceShape = (await readZarrLevelShapes(zarrPath))[sourceLevel] || null;
      }

      const levelShapes = await readZarrLevelShapes(zarrPath);
      const shape = levelShapes[level];
      if (!shape) {
        sendJson(res, 400, { error: `Zarr level ${level} was not found` });
        return;
      }

      const axisSize = shape[axis];
      const sourceAxisSize = sourceShape?.[axis] || axisSize;
      const mappedIndex = sourceShape
        ? Math.round((Math.max(0, Math.min(sourceAxisSize - 1, sourceIndex)) / Math.max(1, sourceAxisSize - 1)) * (axisSize - 1))
        : sourceIndex;
      const sliceIndex = Math.max(0, Math.min(axisSize - 1, Number.isFinite(mappedIndex) ? mappedIndex : Math.floor(axisSize / 2)));
      const { meta, pixels } = await loadCachedPythonZarrSlice(zarrPath, level, axis, sliceIndex, maxSize);

      sendBuffer(res, 200, pixels, "application/octet-stream", {
        "X-QBI-Slice-Axis": axis,
        "X-QBI-Slice-Level": meta.level,
        "X-QBI-Slice-Index": String(meta.index),
        "X-QBI-Slice-Z": String(meta.index),
        "X-QBI-Slice-Width": String(meta.width),
        "X-QBI-Slice-Height": String(meta.height),
        "X-QBI-Slice-Source-Width": String(meta.sourceWidth),
        "X-QBI-Slice-Source-Height": String(meta.sourceHeight),
        "X-QBI-Slice-Depth": String(meta.depth),
        "X-QBI-Slice-Low": String(meta.low),
        "X-QBI-Slice-High": String(meta.high)
      });
    } catch (error) {
      sendJson(res, 500, { error: error.message || String(error) });
    }
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/slicer/export") {
    const zarrPath = resolveScanPath(url.searchParams.get("path"));
    if (!zarrPath) {
      sendJson(res, 400, { error: "Invalid scan path" });
      return;
    }

    const level = url.searchParams.get("level") || "2";
    const { body, filename } = await writeSlicerExport(zarrPath, level);
    sendDownload(res, body, filename, "application/octet-stream");
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/slicer/open") {
    const zarrPath = resolveScanPath(url.searchParams.get("path"));
    if (!zarrPath) {
      sendJson(res, 400, { error: "Invalid scan path" });
      return;
    }

    const level = url.searchParams.get("level") || "2";
    const { filename, filePath } = await writeSlicerExport(zarrPath, level);
    const opened = await openInSlicer(filePath);
    const exportUrl = `/api/slicer/export?path=${encodeURIComponent(path.relative(__dirname, zarrPath))}&level=${encodeURIComponent(level)}`;

    sendJson(res, 200, {
      opened,
      filename,
      filePath,
      exportUrl,
      downloadUrl: "https://download.slicer.org/"
    });
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/upload-picks") {
    const zarrPath = resolveScanPath(url.searchParams.get("path"));
    if (!zarrPath) {
      sendJson(res, 400, { error: "Select a valid Zarr scan before uploading labels" });
      return;
    }

    const uploadId = `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
    const targetRoot = path.join(labelUploadsDir, uploadId);
    const upload = await readMultipartFolder(req, targetRoot, 50_000_000);
    if (upload.error) {
      sendJson(res, 400, { error: upload.error });
      return;
    }

    const jsonFiles = [];
    async function collectJsonFiles(dir) {
      const entries = await fsp.readdir(dir, { withFileTypes: true }).catch(() => []);
      for (const entry of entries) {
        const entryPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          await collectJsonFiles(entryPath);
        } else if (entry.isFile() && entry.name.endsWith(".json")) {
          jsonFiles.push(entryPath);
        }
      }
    }
    await collectJsonFiles(targetRoot);
    jsonFiles.sort();

    if (jsonFiles.length === 0) {
      sendJson(res, 400, { error: "No pick JSON files were found in the uploaded labels folder" });
      return;
    }

    const levelShapes = await readZarrLevelShapes(zarrPath);
    const levelZeroShape = levelShapes["0"] || Object.values(levelShapes)[0];
    if (!levelZeroShape) {
      sendJson(res, 400, { error: "Could not read the selected Zarr shape for label scaling" });
      return;
    }
    const spacing = await readLevelZeroSpacing(zarrPath);
    const volume = buildVolumeMetadata(levelZeroShape, spacing);

    const detections = await buildPickDetectionsFromFiles(zarrPath, levelZeroShape, jsonFiles, "uploaded labels");
    const aggregation = buildMoleculeAggregation(detections, jsonFiles.length, volume);
    latestAggregation.set(zarrPath, aggregation);
    sendJson(res, 200, {
      files: upload.files,
      jsonFiles: jsonFiles.length,
      detections,
      analysis: {
        aggregation,
        structured: buildStructuredFallbackAnalysis(aggregation),
        reportStatus: "Generating..."
      }
    });
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/upload-zarr") {
    await fsp.mkdir(uploadsDir, { recursive: true });
    const scanId = `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
    const targetRoot = path.join(uploadsDir, scanId);
    await fsp.mkdir(targetRoot, { recursive: true });

    const chunks = [];
    let total = 0;
    req.on("data", (chunk) => {
      chunks.push(chunk);
      total += chunk.length;
      if (total > 1_000_000_000) {
        req.destroy();
      }
    });
    req.on("end", async () => {
      try {
        const boundaryMatch = /boundary=(.+)$/.exec(req.headers["content-type"] || "");
        if (!boundaryMatch) {
          sendJson(res, 400, { error: "Multipart boundary not found" });
          return;
        }

        const boundary = `--${boundaryMatch[1]}`;
        const body = Buffer.concat(chunks);
        const parts = body.toString("latin1").split(boundary).slice(1, -1);
        let files = 0;

        for (const part of parts) {
          const separator = part.indexOf("\r\n\r\n");
          if (separator === -1) {
            continue;
          }

          const headers = part.slice(0, separator);
          const filenameMatch = /filename="([^"]+)"/.exec(headers);
          if (!filenameMatch) {
            continue;
          }

          const relativeName = filenameMatch[1].split(/[\\/]/).map(safeSegment).join(path.sep);
          const targetPath = path.join(targetRoot, relativeName);
          if (!targetPath.startsWith(targetRoot)) {
            continue;
          }

          const payload = Buffer.from(part.slice(separator + 4, -2), "latin1");
          await fsp.mkdir(path.dirname(targetPath), { recursive: true });
          await fsp.writeFile(targetPath, payload);
          files += 1;
        }

        const scans = await findZarrs(targetRoot, "Uploaded");
        const validScans = scans.filter((scan) => scan.levels.length > 0);
        if (scans.length > 0 && validScans.length === 0) {
          sendJson(res, 400, {
            error: "The uploaded .zarr folder is missing .zarray metadata files. Browser folder upload often skips hidden dotfiles; use the Local .zarr path field with the original folder instead."
          });
          return;
        }

        sendJson(res, 200, { files, scans: validScans });
      } catch (error) {
        sendJson(res, 500, { error: error.message });
      }
    });
    return;
  }

  // ─── Inference endpoints ──────────────────────────────────────────────

  if (req.method === "GET" && url.pathname === "/api/inference/progress") {
    const scanId = url.searchParams.get("scanId");
    if (!scanId) {
      sendJson(res, 400, { error: "scanId required" });
      return;
    }

    // SSE: Server-Sent Events for progress streaming
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "Access-Control-Allow-Origin": "*"
    });

    const emitter = inferenceEmitters.get(scanId) || new EventEmitter();
    inferenceEmitters.set(scanId, emitter);

    const onProgress = (data) => {
      res.write(`event: progress\ndata: ${JSON.stringify(data)}\n\n`);
    };
    const onComplete = (data) => {
      res.write(`event: complete\ndata: ${JSON.stringify(data)}\n\n`);
      res.end();
      cleanup();
      clearInterval(keepAlive);
    };
    const onError = (err) => {
      res.write(`event: error\ndata: ${JSON.stringify({ error: err.message || String(err) })}\n\n`);
      res.end();
      cleanup();
      clearInterval(keepAlive);
    };

    const cleanup = () => {
      emitter.removeListener("progress", onProgress);
      emitter.removeListener("complete", onComplete);
      emitter.removeListener("error", onError);
      inferenceEmitters.delete(scanId);
    };

    emitter.on("progress", onProgress);
    emitter.on("complete", onComplete);
    emitter.on("error", onError);

    // Keep-alive
    const keepAlive = setInterval(() => {
      res.write(": keepalive\n\n");
    }, 15000);
    req.on("close", () => {
      clearInterval(keepAlive);
      cleanup();
    });
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/graph/hemisphere") {
    try {
      const body = await readRequestJson(req);
      const detections = Array.isArray(body.detections) ? body.detections : [];
      if (detections.length === 0) {
        sendJson(res, 400, { error: "No picks available for hemisphere query" });
        return;
      }
      const pickId = body.pickId || body.pick_id;
      if (!pickId) {
        sendJson(res, 400, { error: "pickId required" });
        return;
      }
      const result = await runGraphApi({
        mode: "hemisphere",
        tomo_id: body.tomoId || "scan",
        pick_id: pickId,
        detections: slimDetectionsForGraph(detections),
        n_rays: body.nRays || 2000
      });
      sendJson(res, 200, result);
    } catch (error) {
      sendJson(res, 500, { error: error.message || String(error) });
    }
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/graph/gvi") {
    try {
      const body = await readRequestJson(req);
      const detections = Array.isArray(body.detections) ? body.detections : [];
      if (detections.length === 0) {
        sendJson(res, 400, { error: "No picks available for grid viability assessment" });
        return;
      }
      const result = await runGraphApi({
        mode: "gvi",
        tomo_id: body.tomoId || "scan",
        detections: slimDetectionsForGraph(detections)
      });
      sendJson(res, 200, result);
    } catch (error) {
      sendJson(res, 500, { error: error.message || String(error) });
    }
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/graph/crowding/progress") {
    const jobId = url.searchParams.get("jobId");
    if (!jobId) {
      sendJson(res, 400, { error: "jobId required" });
      return;
    }

    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": "*"
    });

    const emitter = crowdingEmitters.get(jobId) || new EventEmitter();
    crowdingEmitters.set(jobId, emitter);

    const onProgress = (data) => {
      res.write(`event: progress\ndata: ${JSON.stringify(data)}\n\n`);
    };
    const onComplete = (data) => {
      res.write(`event: complete\ndata: ${JSON.stringify(data)}\n\n`);
      res.end();
      cleanup();
      clearInterval(keepAlive);
    };
    const onError = (err) => {
      res.write(`event: error\ndata: ${JSON.stringify({ error: err.message || String(err) })}\n\n`);
      res.end();
      cleanup();
      clearInterval(keepAlive);
    };

    const cleanup = () => {
      emitter.removeListener("progress", onProgress);
      emitter.removeListener("complete", onComplete);
      emitter.removeListener("error", onError);
      crowdingEmitters.delete(jobId);
    };

    emitter.on("progress", onProgress);
    emitter.on("complete", onComplete);
    emitter.on("error", onError);

    const keepAlive = setInterval(() => {
      res.write(": keepalive\n\n");
    }, 15000);
    req.on("close", () => {
      clearInterval(keepAlive);
      cleanup();
    });
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/graph/crowding/run") {
    try {
      const body = await readRequestJson(req);
      const detections = Array.isArray(body.detections) ? body.detections : [];
      if (detections.length === 0) {
        sendJson(res, 400, { error: "No picks available for crowding analysis" });
        return;
      }

      const jobId = body.jobId || `crowd-${Date.now()}`;
      if (!crowdingEmitters.has(jobId)) {
        crowdingEmitters.set(jobId, new EventEmitter());
      }
      const emitter = crowdingEmitters.get(jobId);

      sendJson(res, 200, { jobId, status: "started" });

      runCrowdingProcess(
        {
          mode: "crowding",
          tomo_id: body.tomoId || "scan",
          detections: slimDetectionsForGraph(detections),
          n_rays: body.nRays || 2000,
          checkpoint: body.checkpoint || defaultGnnCheckpoint
        },
        emitter
      ).catch((err) => {
        emitJobEvent(emitter, "error", { message: err.message || String(err) });
      });
    } catch (error) {
      sendJson(res, 500, { error: error.message || String(error) });
    }
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/inference/run") {
    const body = await readRequestJson(req);
    const zarrPath = body.zarrPath;
    const scanId = body.scanId || `infer-${Date.now()}`;

    if (!zarrPath) {
      sendJson(res, 400, { error: "zarrPath is required" });
      return;
    }

    // Resolve the actual filesystem path
    let resolvedPath = null;
    if (zarrPath.startsWith("local:")) {
      resolvedPath = localScans.get(zarrPath);
    } else {
      resolvedPath = resolveScanPath(zarrPath);
    }
    if (!resolvedPath) {
      // Try as a direct path
      resolvedPath = path.resolve(zarrPath);
      if (!fs.existsSync(resolvedPath)) {
        sendJson(res, 400, { error: `Zarr path not found: ${zarrPath}` });
        return;
      }
    }

    // Create emitter if not already present
    if (!inferenceEmitters.has(scanId)) {
      inferenceEmitters.set(scanId, new EventEmitter());
    }
    const emitter = inferenceEmitters.get(scanId);

    // Respond immediately that inference has started
    sendJson(res, 200, { scanId, status: "started", message: "Inference started" });

    // Run inference asynchronously
    runInferenceProcess(resolvedPath, scanId, emitter).catch((err) => {
      emitJobEvent(emitter, "error", { message: err.message || String(err) });
    });
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/analysis") {
    const body = await readRequestJson(req);
    const resolvedPath = resolveScanPath(body.zarrPath);
    if (!resolvedPath) {
      sendJson(res, 400, { error: "Invalid or unknown scan path" });
      return;
    }

    const aggregation = latestAggregation.get(resolvedPath);
    if (!aggregation) {
      sendJson(res, 404, { error: "No scan data cached for this path. Load the scan first." });
      return;
    }

    const aiAnalysis = await generateClaudeAnalysis(aggregation, 45000);
    sendJson(res, 200, { ...aiAnalysis, aggregation });
    return;
  }

  sendJson(res, 404, { error: "API route not found" });
}

// ─── Inference runner ──────────────────────────────────────────────────

const projectRoot = path.resolve(__dirname, "..");
const inferenceScript = path.join(projectRoot, "backend", "inference", "run_inference_direct.py");
const graphApiModule = "downstream.scripts.frontend_graph_api";
const defaultGnnCheckpoint = path.join(projectRoot, "runs", "exposure_gnn.pt");
const hardcodedCheckpoint = path.resolve(os.homedir(), "Downloads", "czii-weights", "weight_best.ckpt");

function pythonExecutable() {
  const venvPython = path.join(projectRoot, ".venv", "bin", "python");
  return fs.existsSync(venvPython) ? venvPython : "python3";
}

function slimDetectionsForGraph(detections) {
  return detections.map((det) => ({
    id: det.id,
    molecule: det.molecule || det.type,
    type: det.type,
    radiusAngstrom: det.radiusAngstrom,
    physical: det.physical
      ? { x: det.physical.x, y: det.physical.y, z: det.physical.z }
      : undefined
  }));
}

function runGraphApi(payload) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      pythonExecutable(),
      ["-m", graphApiModule],
      {
        cwd: projectRoot,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
        stdio: ["pipe", "pipe", "pipe"]
      }
    );

    const stderrChunks = [];
    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();

    let stdout = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk) => {
      stderrChunks.push(chunk.toString("utf8"));
    });

    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        const stderr = stderrChunks.join("");
        reject(new Error(stderr.split("\n").filter(Boolean).slice(-3).join("; ") || `Graph API exited ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(new Error(`Failed to parse graph API output: ${error.message}`));
      }
    });
  });
}

async function runCrowdingProcess(payload, emitter) {
  emitter.emit("progress", { percent: 0, message: "Starting crowding analysis…" });

  return new Promise((resolve, reject) => {
    const child = spawn(
      pythonExecutable(),
      ["-m", graphApiModule],
      {
        cwd: projectRoot,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
        stdio: ["pipe", "pipe", "pipe"]
      }
    );

    const stderrChunks = [];
    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();

    let stdout = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk) => {
      const text = chunk.toString("utf8");
      stderrChunks.push(text);
      const progressMatch = text.match(/__QBI_PROGRESS__:([0-9.]+):(.+)/);
      if (progressMatch) {
        const fraction = Number.parseFloat(progressMatch[1]);
        emitter.emit("progress", {
          percent: Math.round(fraction * 100),
          message: progressMatch[2].trim()
        });
      }
    });

    child.on("error", (err) => {
      emitJobEvent(emitter, "error", { message: err.message });
      reject(err);
    });

    child.on("close", (code) => {
      if (code !== 0) {
        const stderr = stderrChunks.join("");
        const errorMsg = stderr.split("\n").filter((l) => l && !l.startsWith("__QBI_")).slice(-3).join("; ");
        emitJobEvent(emitter, "error", { message: errorMsg || `Crowding analysis failed (exit ${code})` });
        reject(new Error(errorMsg));
        return;
      }
      try {
        const result = JSON.parse(stdout);
        emitter.emit("complete", result);
        resolve(result);
      } catch (error) {
        emitJobEvent(emitter, "error", { message: error.message });
        reject(error);
      }
    });
  });
}

async function runInferenceProcess(zarrPath, scanId, emitter) {
  const checkpoint = hardcodedCheckpoint;
  if (!fs.existsSync(checkpoint)) {
    emitJobEvent(emitter, "error", { message: `Checkpoint not found at ${checkpoint}. Please ensure the file exists.` });
    return;
  }

  await fsp.mkdir(predictionsDir, { recursive: true });
  const outputPath = path.join(predictionsDir, `predictions_${scanId}.json`);

  // Determine device
  let device = "cpu";
  if (process.platform === "darwin" && process.arch === "arm64") {
    device = "mps";
  } else if (process.platform !== "darwin") {
    device = "cuda";
  }

  const env = {
    ...process.env,
    PYTORCH_MPS_HIGH_WATERMARK_RATIO: "0.0",
    PYTHONUNBUFFERED: "1"
  };

  const args = [
    inferenceScript,
    "--checkpoint", checkpoint,
    "--zarr_path", zarrPath,
    "--device", device,
    "--dtype", "float32",
    "--window_size", "64", "64", "64",
    "--tiles_per_dim", "3", "10", "10",
    "--output", outputPath,
    "--iou_threshold", "0.85"
  ];

  emitter.emit("progress", { phase: "loading", percent: 0, message: "Starting inference..." });

  const child = spawn(pythonExecutable(), args, {
    cwd: projectRoot,
    env,
    stdio: ["ignore", "pipe", "pipe"]
  });

  const stderrChunks = [];

  child.stderr.on("data", (chunk) => {
    const text = chunk.toString("utf8");
    stderrChunks.push(text);

    // Parse progress lines — format: __QBI_PROGRESS__:<percent>:<batch>/<total>:<bar>
    const progressMatch = text.match(/__QBI_PROGRESS__:([0-9.]+)(?::(\d+)\/(\d+):([█░]+))?/);
    if (progressMatch) {
      const percent = Number.parseFloat(progressMatch[1]);
      const phase = percent < 0.1 ? "loading" : percent < 0.85 ? "inference" : "decoding";
      const messages = {
        loading: "Loading model and volume...",
        inference: "Running model inference on tiles...",
        decoding: "Decoding detections..."
      };

      const progressData = {
        phase,
        percent: Math.round(percent * 100),
        message: messages[phase] || "Processing..."
      };

      // Add tqdm-style bar info if available
      if (progressMatch[2] && progressMatch[3]) {
        const batch = Number.parseInt(progressMatch[2], 10);
        const total = Number.parseInt(progressMatch[3], 10);
        const bar = progressMatch[4] || "";
        progressData.batch = batch;
        progressData.total = total;
        progressData.bar = bar;
        progressData.message = `Tile ${batch} / ${total}  ${bar}`;
      }

      emitter.emit("progress", progressData);
    }

    // Also emit log lines for the UI
    const logMatch = text.match(/^(Loading|Running|Found|Saved|Volume|Output)/m);
    if (logMatch) {
      emitter.emit("progress", {
        phase: "inference",
        percent: -1,
        message: text.trim()
      });
    }
  });

  child.on("error", (err) => {
    emitJobEvent(emitter, "error", { message: `Failed to start inference: ${err.message}` });
  });

  child.on("close", async (code) => {
    if (code !== 0) {
      const stderr = stderrChunks.join("");
      const errorMsg = stderr.split("\n").filter((l) => l && !l.startsWith("__QBI_")).slice(-5).join("; ");
      emitJobEvent(emitter, "error", { message: `Inference failed (exit ${code}): ${errorMsg || "Unknown error"}` });
      return;
    }

    // Read the predictions file
    try {
      const predictionsRaw = await fsp.readFile(outputPath, "utf8");
      const predictions = JSON.parse(predictionsRaw);

      // Convert predictions to pick-style detections
      const levelShapes = await readZarrLevelShapes(zarrPath);
      let levelZeroShape = levelShapes["0"] || Object.values(levelShapes)[0];
      if (!levelZeroShape) {
        // Fallback: use a default shape based on the predictions
        const maxVoxel = { x: 0, y: 0, z: 0 };
        for (const det of (predictions.detections || [])) {
          maxVoxel.x = Math.max(maxVoxel.x, Math.ceil(det.x_pixel || 0));
          maxVoxel.y = Math.max(maxVoxel.y, Math.ceil(det.y_pixel || 0));
          maxVoxel.z = Math.max(maxVoxel.z, Math.ceil(det.z_pixel || 0));
        }
        levelZeroShape = { x: Math.max(1, maxVoxel.x + 100), y: Math.max(1, maxVoxel.y + 100), z: Math.max(1, maxVoxel.z + 100) };
      }
      const spacing = await readLevelZeroSpacing(zarrPath).catch(() => ({ x: 10, y: 10, z: 10 }));

      // Build pick files from predictions
      const pickFiles = await buildPickFilesFromPredictions(predictions, zarrPath, levelZeroShape, spacing, scanId);

      // Load the detections into the format the frontend expects
      const detections = [];
      for (const [fileIndex, pickFile] of pickFiles.entries()) {
        const payload = await readJson(pickFile.path);
        const points = Array.isArray(payload?.points) ? payload.points : [];
        const type = payload?.pickable_object_name || path.basename(pickFile.path, ".json");
        const label = formatMoleculeName(type);
        const color = pickColor(type, fileIndex);
        const radiusAngstrom = pickRadiusAngstrom(type);
        const radiusVoxel = radiusAngstrom / ((spacing.x + spacing.y + spacing.z) / 3);

        points.forEach((point, pointIndex) => {
          const location = point.location || {};
          const voxel = {
            x: Number(location.x) / spacing.x,
            y: Number(location.y) / spacing.y,
            z: Number(location.z) / spacing.z
          };

          if (!Number.isFinite(voxel.x) || !Number.isFinite(voxel.y) || !Number.isFinite(voxel.z)) {
            return;
          }

          const clipped = {
            x: Math.max(0, Math.min(levelZeroShape.x - 1, voxel.x)),
            y: Math.max(0, Math.min(levelZeroShape.y - 1, voxel.y)),
            z: Math.max(0, Math.min(levelZeroShape.z - 1, voxel.z))
          };

          detections.push({
            id: `${type}-${String(pointIndex + 1).padStart(3, "0")}`,
            type: label,
            molecule: type,
            confidence: `model: ${(point.score || 1.0).toFixed(3)}`,
            volume: path.basename(zarrPath),
            color,
            radiusAngstrom,
            radiusVoxel,
            coords: [
              (clipped.x / Math.max(1, levelZeroShape.x - 1)) - 0.5,
              (clipped.y / Math.max(1, levelZeroShape.y - 1)) - 0.5,
              (clipped.z / Math.max(1, levelZeroShape.z - 1)) - 0.5
            ],
            voxel: { x: clipped.x, y: clipped.y, z: clipped.z },
            physical: {
              x: Number(location.x),
              y: Number(location.y),
              z: Number(location.z),
              unit: "angstrom"
            },
            position: `X ${Math.round(clipped.x)}, Y ${Math.round(clipped.y)}, Z ${Math.round(clipped.z)}`,
            notes: `${label} model detection. Confidence ${(point.score || 1.0).toFixed(3)}.`
          });
        });
      }

      // Build analysis inline (with 15s timeout on Claude API)
      let aiAnalysis = null;
      try {
        const aggregation = buildMoleculeAggregation(detections, pickFiles.length, buildVolumeMetadata(levelZeroShape, spacing));
        latestAggregation.set(zarrPath, aggregation);
        aiAnalysis = { structured: buildStructuredFallbackAnalysis(aggregation), reportStatus: "Generating..." };
      } catch {
        // Analysis is best-effort
      }

      emitter.emit("complete", {
        detections,
        analysis: aiAnalysis,
        predictionsFile: path.relative(__dirname, outputPath),
        pickFiles: pickFiles.map((f) => path.relative(__dirname, f.path)),
        numDetections: detections.length
      });
    } catch (err) {
      emitJobEvent(emitter, "error", { message: `Failed to process predictions: ${err.message}` });
    }
  });
}

async function buildPickFilesFromPredictions(predictions, zarrPath, levelZeroShape, spacing, scanId) {
  const detections = predictions.detections || [];
  const studyName = path.basename(zarrPath).replace(/\.zarr$/i, "");

  // Group by particle type
  const groups = {};
  for (const det of detections) {
    const type = det.particle_type || "unknown";
    if (!groups[type]) groups[type] = [];
    groups[type].push(det);
  }

  const pickDir = path.join(predictionsDir, `picks_${scanId}`);
  await fsp.mkdir(pickDir, { recursive: true });
  const writtenFiles = [];

  for (const [moleculeName, moleculeDets] of Object.entries(groups)) {
    const points = moleculeDets.map((det) => ({
      location: {
        x: det.x_angstrom,
        y: det.y_angstrom,
        z: det.z_angstrom
      },
      transformation_: [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ],
      instance_id: 0,
      score: det.score || 1.0
    }));

    const pickPayload = {
      pickable_object_name: moleculeName,
      user_id: "model",
      session_id: "0",
      run_name: studyName,
      voxel_spacing: null,
      unit: "angstrom",
      points,
      trust_orientation: true
    };

    const outputPath = path.join(pickDir, `${moleculeName}.json`);
    await fsp.writeFile(outputPath, JSON.stringify(pickPayload, null, 2));
    writtenFiles.push({ path: outputPath, molecule: moleculeName });
  }

  return writtenFiles;
}

const server = http.createServer((req, res) => {
  if (req.url.startsWith("/api/")) {
    handleApi(req, res).catch((error) => {
      sendJson(res, 500, { error: error.message });
    });
    return;
  }

  if (!["GET", "HEAD"].includes(req.method)) {
    send(res, 405, "Method Not Allowed");
    return;
  }

  let filePath;

  try {
    filePath = resolveStaticPath(req.url);
  } catch {
    send(res, 400, "Bad Request");
    return;
  }

  if (!filePath) {
    send(res, 403, "Forbidden");
    return;
  }

  fs.stat(filePath, (statError, stats) => {
    if (statError) {
      send(res, 404, "Not Found");
      return;
    }

    const requestedPath = stats.isDirectory() ? path.join(filePath, "index.html") : filePath;

    fs.readFile(requestedPath, (readError, data) => {
      if (readError) {
        send(res, 404, "Not Found");
        return;
      }

      const contentType = mimeTypes[path.extname(requestedPath).toLowerCase()] || "application/octet-stream";

      res.writeHead(200, {
        "Content-Type": contentType,
        "Content-Length": data.length
      });

      if (req.method === "HEAD") {
        res.end();
      } else {
        res.end(data);
      }
    });
  });
});

server.listen(port, () => {
  console.log(`QBI server running at http://localhost:${port}`);
  console.log(`Serving static files from ${publicDir}`);
});
