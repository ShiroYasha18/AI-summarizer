const fs = require("fs");
const http = require("http");
const path = require("path");

function tryLoadDotEnv() {
  const candidates = [
    path.join(process.cwd(), ".env"),
    path.join(__dirname, ".env"),
    path.join(__dirname, "..", ".env")
  ];

  for (const fp of candidates) {
    try {
      if (!fs.existsSync(fp)) continue;
      const raw = fs.readFileSync(fp, "utf8");
      if (!String(raw || "").trim()) continue;
      applyDotEnv(raw);
      break;
    } catch {}
  }
}

function applyDotEnv(raw) {
  const text = String(raw || "");
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line) continue;
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq <= 0) continue;
    const key = trimmed.slice(0, eq).trim();
    if (!key || process.env[key] !== undefined) continue;
    let rest = trimmed.slice(eq + 1);
    if (rest.startsWith(" ")) rest = rest.trimStart();
    if (!rest) continue;

    let val = "";
    const q = rest[0];
    if (q === '"' || q === "'") {
      const end = rest.indexOf(q, 1);
      if (end > 0) val = rest.slice(1, end);
      else val = rest.slice(1);
      if (val) process.env[key] = val;
      continue;
    }

    if (rest.startsWith("{")) {
      let acc = rest;
      let balance = 0;
      for (const ch of rest) {
        if (ch === "{") balance++;
        else if (ch === "}") balance--;
      }
      while (balance > 0 && i + 1 < lines.length) {
        i++;
        const next = lines[i] ?? "";
        acc += `\n${next}`;
        for (const ch of next) {
          if (ch === "{") balance++;
          else if (ch === "}") balance--;
        }
      }
      val = acc.trim();
      if (val) process.env[key] = val;
      continue;
    }

    val = rest.trim();
    if (val) process.env[key] = val;
  }
}

function tryLoadServiceAccountJson() {
  if (String(process.env.FIREBASE_SERVICE_ACCOUNT_JSON || "").trim()) return;
  const fp = path.join(__dirname, "serviceAccountKey.json");
  try {
    const raw = fs.readFileSync(fp, "utf8");
    if (String(raw || "").trim()) process.env.FIREBASE_SERVICE_ACCOUNT_JSON = raw;
  } catch {}
}

function send(res, status, body, contentType) {
  res.statusCode = status;
  if (contentType) res.setHeader("Content-Type", contentType);
  res.end(body);
}

function contentTypeFor(fp) {
  switch (path.extname(fp).toLowerCase()) {
    case ".html":
      return "text/html; charset=utf-8";
    case ".js":
      return "text/javascript; charset=utf-8";
    case ".css":
      return "text/css; charset=utf-8";
    case ".json":
      return "application/json; charset=utf-8";
    case ".svg":
      return "image/svg+xml";
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    default:
      return "application/octet-stream";
  }
}

function safeJoin(root, urlPath) {
  const decoded = decodeURIComponent(urlPath);
  const joined = path.join(root, decoded);
  if (!joined.startsWith(root)) return null;
  return joined;
}

function parsePort() {
  const fromEnv = Number(process.env.PORT);
  if (Number.isFinite(fromEnv) && fromEnv > 0) return fromEnv;
  const raw = process.argv[2];
  const fromArg = raw ? Number(raw) : NaN;
  if (Number.isFinite(fromArg) && fromArg > 0) return fromArg;
  return 5176;
}

function main() {
  tryLoadDotEnv();
  tryLoadServiceAccountJson();
  const apiHandler = require("./api/router.js");
  const root = __dirname;
  const port = parsePort();

  const server = http.createServer((req, res) => {
    const url = new URL(req.url || "/", "http://localhost");
    if (url.pathname.startsWith("/api/")) return apiHandler(req, res);

    let p = url.pathname;
    if (p === "/" || p === "") p = "/index.html";
    const fp = safeJoin(root, p);
    if (!fp) return send(res, 403, "forbidden", "text/plain; charset=utf-8");

    fs.readFile(fp, (err, data) => {
      if (err) return send(res, 404, "not found", "text/plain; charset=utf-8");
      send(res, 200, data, contentTypeFor(fp));
    });
  });

  server.listen(port, "127.0.0.1", () => {
    process.stdout.write(`http://127.0.0.1:${port}\n`);
  });
}

main();
