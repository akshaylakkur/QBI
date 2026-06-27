const http = require("node:http");
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const path = require("node:path");

const port = Number.parseInt(process.env.PORT || "3000", 10);
const publicDir = __dirname;
const sampleDir = path.join(__dirname, "sample_data");
const uploadsDir = path.join(__dirname, "uploaded_scans");
const volumeCache = new Map();

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

function sendBuffer(res, statusCode, body, contentType = "application/octet-stream") {
  res.writeHead(statusCode, {
    "Content-Type": contentType,
    "Content-Length": body.length
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
  const decoded = decodeURIComponent(scanPath || "");
  const normalized = path.normalize(decoded).replace(/^(\.\.[/\\])+/, "");
  const filePath = path.join(__dirname, normalized);
  const allowedRoots = [sampleDir, uploadsDir];

  if (!allowedRoots.some((root) => filePath === root || filePath.startsWith(`${root}${path.sep}`))) {
    return null;
  }

  return filePath;
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
      results.push({
        id: relativePath,
        name: `${baseLabel}: ${path.basename(dir)}`,
        path: relativePath
      });
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
  const cacheKey = `${zarrPath}:${levelName}`;
  if (volumeCache.has(cacheKey)) {
    return volumeCache.get(cacheKey);
  }

  const arrayPath = path.join(zarrPath, levelName);
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
      Number((point.score / scoreMax).toFixed(5))
    ])
  };
}

function buildDetections(volume) {
  const { shape } = volume;
  return [
    {
      id: "M01",
      type: "candidate ribosome",
      confidence: "sample label",
      volume: "preview marker",
      position: `X ${Math.round(shape.x * 0.36)}, Y ${Math.round(shape.y * 0.46)}, Z ${Math.round(shape.z * 0.42)}`,
      coords: [-0.14, -0.04, -0.08],
      notes: "Placeholder detection. This is where AI model output can be attached."
    },
    {
      id: "M02",
      type: "candidate membrane complex",
      confidence: "sample label",
      volume: "preview marker",
      position: `X ${Math.round(shape.x * 0.62)}, Y ${Math.round(shape.y * 0.54)}, Z ${Math.round(shape.z * 0.58)}`,
      coords: [0.12, 0.04, 0.08],
      notes: "Clicking this marker uses the same selection path future detections will use."
    },
    {
      id: "M03",
      type: "candidate vesicle",
      confidence: "sample label",
      volume: "preview marker",
      position: `X ${Math.round(shape.x * 0.52)}, Y ${Math.round(shape.y * 0.31)}, Z ${Math.round(shape.z * 0.64)}`,
      coords: [0.02, -0.19, 0.14],
      notes: "The current marker positions are illustrative until model detections are available."
    }
  ];
}

async function handleApi(req, res) {
  const url = new URL(req.url, `http://localhost:${port}`);

  if (req.method === "GET" && url.pathname === "/api/scans") {
    const [sampleScans, uploadedScans] = await Promise.all([
      findZarrs(sampleDir, "Sample"),
      findZarrs(uploadsDir, "Uploaded")
    ]);
    sendJson(res, 200, { scans: [...sampleScans, ...uploadedScans] });
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
    const limit = Number.parseInt(url.searchParams.get("limit") || "38000", 10);
    const volume = await loadVolumePreview(zarrPath, level);
    sendJson(res, 200, {
      level,
      ...buildPointCloud(volume, stride, limit),
      detections: buildDetections(volume)
    });
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/zarr/slice") {
    const zarrPath = resolveScanPath(url.searchParams.get("path"));
    if (!zarrPath) {
      sendJson(res, 400, { error: "Invalid scan path" });
      return;
    }

    const level = url.searchParams.get("level") || "2";
    const volume = await loadVolumePreview(zarrPath, level);
    const z = Math.max(0, Math.min(volume.shape.z - 1, Number.parseInt(url.searchParams.get("z") || `${Math.floor(volume.shape.z / 2)}`, 10)));
    const pixels = Buffer.alloc(volume.shape.x * volume.shape.y);
    const values = [];

    for (let y = 0; y < volume.shape.y; y += 1) {
      for (let x = 0; x < volume.shape.x; x += 1) {
        const value = volume.data[z * volume.shape.y * volume.shape.x + y * volume.shape.x + x];
        if (Number.isFinite(value) && Math.abs(value) <= 1_000_000) {
          values.push(value);
        }
      }
    }

    values.sort((a, b) => a - b);
    const min = values[Math.floor(values.length * 0.01)] || 0;
    const max = values[Math.floor(values.length * 0.99)] || 1;
    const range = max - min || 1;

    for (let y = 0; y < volume.shape.y; y += 1) {
      for (let x = 0; x < volume.shape.x; x += 1) {
        const value = volume.data[z * volume.shape.y * volume.shape.x + y * volume.shape.x + x];
        const safeValue = Number.isFinite(value) && Math.abs(value) <= 1_000_000 ? value : min;
        pixels[y * volume.shape.x + x] = Math.max(0, Math.min(255, Math.round(((safeValue - min) / range) * 255)));
      }
    }

    sendBuffer(res, 200, pixels, "application/octet-stream");
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
        sendJson(res, 200, { files, scans });
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
