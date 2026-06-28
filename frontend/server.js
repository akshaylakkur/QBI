const http = require("node:http");
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const path = require("node:path");
const os = require("node:os");
const { spawn } = require("node:child_process");

const port = Number.parseInt(process.env.PORT || "3000", 10);
const publicDir = __dirname;
const sampleDir = path.join(__dirname, "sample_data");
const uploadsDir = path.join(__dirname, "uploaded_scans");
const labelUploadsDir = path.join(__dirname, "uploaded_labels");
const slicerExportDir = path.join(os.tmpdir(), "qbi-slicer-exports");
const volumeCache = new Map();
const volumeLoadCache = new Map();
const sliceCache = new Map();
const sliceLoadCache = new Map();
let sliceCacheBytes = 0;
const maxSliceCacheBytes = 96 * 1024 * 1024;
const localScans = new Map();
const pickCache = new Map();

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

function buildClaudeAnalysisInput(aggregation) {
  return {
    volume: aggregation.volume,
    totalPicks: aggregation.totalPicks,
    totalClusters: aggregation.totalClusters,
    jsonFiles: aggregation.jsonFiles,
    items: aggregation.items.map((item) => ({
      molecule: item.molecule,
      label: item.label,
      difficulty: item.difficulty,
      count: item.count,
      frequencyPercent: Number(item.frequencyPercent.toFixed(3)),
      clusterThresholdAngstroms: Number(item.clusterThresholdAngstroms.toFixed(3)),
      clusterCount: item.clusterCount,
      clusters: item.clusters.map((cluster) => ({
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
    "You analyze CryoET object-pick label summaries for research triage.",
    "Be scientifically cautious.",
    "Do not claim clinical, mechanistic, or drug-response conclusions from a single uploaded label folder.",
    "Use only the supplied aggregated counts.",
    "If the data is insufficient, say so plainly.",
    "Write a concise report with headings: Dataset Composition, Frequency Interpretation, Caveats, Downstream Use, Drug-Discovery Relevance, Next Validation Steps."
  ].join(" ");
}

function buildClaudeUserPrompt(aggregation) {
  const input = buildClaudeAnalysisInput(aggregation);
  return [
    "Create a concise research-style report from the JSON below.",
    "Only use the fields in the JSON and do not infer extra measurements.",
    "If a molecule has no observations, do not fabricate one.",
    "JSON:",
    JSON.stringify(input, null, 2)
  ].join("\n\n");
}

async function generateClaudeAnalysis(aggregation) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return { report: buildLocalAnalysisSummary(aggregation), reportStatus: "Local summary" };
  }

  const model = process.env.ANTHROPIC_MODEL || "claude-sonnet-4-6";
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model,
      max_tokens: 900,
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
    return {
      report: `${buildLocalAnalysisSummary(aggregation)}\n\n### Claude Report Status\n${message}`,
      reportStatus: `Claude unavailable (${response.status})`,
      reportError: message,
      reportErrorBody: body,
      reportErrorStatus: response.status
    };
  }

  const payload = await response.json();
  return {
    report: extractClaudeText(payload) || buildLocalAnalysisSummary(aggregation),
    reportStatus: `Claude · ${model}`
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
    const child = spawn("python3", [
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

    sendJson(res, 200, {
      level: resolvedLevel,
      levels,
      levelShapes,
      shape,
      stats: { min: 0, max: 0, points: 0, stride },
      points: [],
      detections: pickDetections
    });
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/zarr/slice") {
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
    const aiAnalysis = await generateClaudeAnalysis(aggregation);
    sendJson(res, 200, {
      files: upload.files,
      jsonFiles: jsonFiles.length,
      detections,
      analysis: {
        aggregation,
        ...aiAnalysis
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

  sendJson(res, 404, { error: "API route not found" });
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

  if (new URL(req.url, `http://localhost:${port}`).pathname === "/test3d") {
    req.url = "/test3d.html";
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
