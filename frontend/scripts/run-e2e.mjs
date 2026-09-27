import {spawn, spawnSync} from 'node:child_process';
import {createServer} from 'node:http';
import {randomBytes} from 'node:crypto';
import {mkdtempSync, readFileSync, readdirSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const frontend = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(frontend, '..');
const temporaryPrefix = 'medical-ai-suite-e2e-';
const requiredNodeVersion = 'v20.19.0';
const temporaryPython = mkdtempSync(path.join(tmpdir(), temporaryPrefix));
const children = new Set();
const serviceKey = randomBytes(48).toString('base64url');
const controlToken = randomBytes(32).toString('base64url');
const controlState = {fault: null, keySeen: false, paths: [], predictionBodies: [], reportCount: 0};
const reservedPorts = new Set();
let controlServer;
let apiPort;
let frontendPort;
let proxyPort;
let interrupted = false;

function runSync(executable, args, options = {}) {
  const result = spawnSync(executable, args, {cwd: options.cwd ?? root, encoding: 'utf8', stdio: 'pipe', ...options});
  if (result.status !== 0) {
    let details = `${result.stderr ?? ''}\n${result.stdout ?? ''}`;
    for (const sensitive of [serviceKey, options.env?.BACKEND_INTERNAL_URL].filter(Boolean)) details = details.replaceAll(sensitive, '[redacted]');
    throw new Error(`Command failed (${executable} ${args.join(' ')}): ${details.trim() || result.error || `exit ${result.status}`}`);
  }
  return result.stdout?.trim() ?? '';
}

function assertNodeVersion(context) {
  const executableVersion = runSync(process.execPath, ['-p', 'process.version']);
  console.log(`[node] ${context}: runner=${process.version}, executable=${executableVersion}`);
  if (process.version !== requiredNodeVersion || executableVersion !== requiredNodeVersion) {
    throw new Error(`${context} requires Node ${requiredNodeVersion}; received runner=${process.version}, executable=${executableVersion}.`);
  }
}

function findPython311() {
  const candidates = process.env.E2E_PYTHON ? [[process.env.E2E_PYTHON, []]] : process.platform === 'win32'
    ? [['py', ['-3.11']], ['python3.11', []], ['python', []]]
    : [['python3.11', []], ['python3', []], ['python', []]];
  for (const [command, prefix] of candidates) {
    const probe = spawnSync(command, [...prefix, '--version'], {encoding: 'utf8'});
    if (probe.status === 0 && /Python 3\.11\./.test(`${probe.stdout}${probe.stderr}`)) return {command, prefix};
  }
  throw new Error('Python 3.11 was not found. Install Python 3.11 or set E2E_PYTHON to a clean Python 3.11 environment.');
}

function installLockedPython() {
  const candidate = findPython311();
  if (process.env.E2E_PYTHON) {
    const pipCheck = spawnSync(candidate.command, [...candidate.prefix, '-m', 'pip', 'check'], {encoding: 'utf8'});
    if (pipCheck.status !== 0) {
      if (!/No module named pip/.test(`${pipCheck.stderr}${pipCheck.stdout}`)) throw new Error(pipCheck.stderr || 'Python dependency integrity check failed.');
      runSync('uv', ['pip', 'check', '--python', candidate.command]);
    }
    return candidate;
  }
  runSync(candidate.command, [...candidate.prefix, '-m', 'venv', path.join(temporaryPython, 'venv')]);
  const python = path.join(temporaryPython, 'venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  runSync(python, ['-m', 'pip', 'install', '--require-hashes', '-r', path.join(root, 'requirements-dev.lock')]);
  runSync(python, ['-m', 'pip', 'check']);
  return {command: python, prefix: []};
}

function reservePort() {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      const port = address.port;
      server.close(() => {
        if (reservedPorts.has(port)) return reservePort().then(resolve, reject);
        reservedPorts.add(port);
        resolve(port);
      });
    });
  });
}

function start(executable, args, env, name, cwd = root) {
  const child = spawn(executable, args, {cwd, env: {...process.env, ...env}, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true});
  children.add(child);
  child.once('error', (error) => process.stderr.write(`[${name}] failed to start: ${error.message}\n`));
  child.stdout.on('data', (chunk) => process.stdout.write(`[${name}] ${chunk}`));
  child.stderr.on('data', (chunk) => process.stderr.write(`[${name}] ${chunk}`));
  child.once('exit', (code) => children.delete(child));
  return child;
}

async function waitFor(url, child, name) {
  const until = Date.now() + 60_000;
  let lastError = 'not ready';
  while (Date.now() < until) {
    if (child.exitCode !== null) throw new Error(`${name} exited before becoming healthy (code ${child.exitCode}).`);
    try {
      const response = await fetch(url);
      if (response.ok) return;
      lastError = `HTTP ${response.status}`;
    } catch (error) { lastError = error.message; }
    await new Promise((resolve) => setTimeout(resolve, 400));
  }
  throw new Error(`${name} did not start at ${new URL(url).origin} (${lastError}).`);
}

function sendJson(response, status, body) {
  const bytes = Buffer.from(JSON.stringify(body));
  response.writeHead(status, {'Content-Type': 'application/json', 'Content-Length': bytes.length, 'Cache-Control': 'no-store'});
  response.end(bytes);
}

function abortableDelay(milliseconds, signal) {
  return new Promise((resolve) => {
    if (signal.aborted) return resolve();
    const timer = setTimeout(resolve, milliseconds);
    signal.addEventListener('abort', () => { clearTimeout(timer); resolve(); }, {once: true});
  });
}

function createProxyServer() {
  return createServer(async (request, response) => {
    if (request.url?.startsWith('/__control/')) {
      if (request.headers.authorization !== `Bearer ${controlToken}`) return sendJson(response, 404, {error: 'not_found'});
      if (request.method === 'GET' && request.url === '/__control/state') {
        return sendJson(response, 200, {keySeen: controlState.keySeen, paths: controlState.paths, predictionBodies: controlState.predictionBodies, reportCount: controlState.reportCount});
      }
      if (request.method === 'POST' && request.url === '/__control/fault') {
        const chunks = [];
        for await (const chunk of request) chunks.push(chunk);
        const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
        controlState.fault = body.mode ? {mode: body.mode, disease: body.disease ?? null} : null;
        return sendJson(response, 200, {ok: true});
      }
      return sendJson(response, 404, {error: 'not_found'});
    }

    controlState.keySeen ||= request.headers['x-api-key'] === serviceKey;
    const apiPath = request.url ?? '/';
    controlState.paths.push(`${request.method} ${apiPath}`);
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const body = Buffer.concat(chunks);
    const abort = new AbortController();
    request.once('aborted', () => abort.abort());
    response.once('close', () => { if (!response.writableEnded) abort.abort(); });
    if (apiPath.includes('/predictions/')) controlState.predictionBodies.push(body.toString('utf8'));
    if (apiPath.includes('/reports/')) controlState.reportCount += 1;

    const mode = controlState.fault?.disease && !apiPath.endsWith(`/${controlState.fault.disease}`) ? null : controlState.fault?.mode;
    console.log(`[E2E upstream] ${request.method} ${apiPath.split('/').slice(-2).join('/')} started`);
    if (mode === '500') return sendJson(response, 500, {detail: 'Model service is temporarily unavailable'});
    if (mode === 'malformed') {
      response.writeHead(200, {'Content-Type': 'application/json', 'Cache-Control': 'no-store'});
      return response.end('not-json');
    }
    if (mode === 'report-failure' && apiPath.includes('/reports/')) return sendJson(response, 503, {detail: 'Report service is temporarily unavailable'});
    if (mode === 'prediction-delay' && apiPath.includes('/predictions/')) await abortableDelay(400, abort.signal);
    if (mode === 'timeout') await abortableDelay(16_000, abort.signal);
    if (mode === 'report-delay' && apiPath.includes('/reports/')) await abortableDelay(400, abort.signal);
    if (abort.signal.aborted || response.destroyed) return;
    const startedAt = Date.now();
    try {
      const upstream = await fetch(`http://127.0.0.1:${apiPort}${apiPath}`, {
        method: request.method,
        headers: {'Content-Type': request.headers['content-type'] ?? 'application/json', 'X-API-Key': request.headers['x-api-key'] ?? ''},
        body: ['GET', 'HEAD'].includes(request.method ?? '') ? undefined : body,
        signal: abort.signal,
      });
      if (response.destroyed) return;
      console.log(`[E2E upstream] ${request.method} ${apiPath.split('/').slice(-2).join('/')} -> ${upstream.status} (${Date.now() - startedAt} ms)`);
      response.writeHead(upstream.status, Object.fromEntries(upstream.headers));
      response.end(Buffer.from(await upstream.arrayBuffer()));
    } catch {
      if (!response.destroyed) sendJson(response, 502, {detail: 'Upstream service unavailable'});
    }
  });
}

async function closeChild(child) {
  if (child.exitCode !== null) return;
  if (!child.pid) return;
  if (process.platform === 'win32') spawnSync('taskkill', ['/pid', String(child.pid), '/t', '/f'], {stdio: 'ignore'});
  else child.kill('SIGTERM');
  await Promise.race([new Promise((resolve) => child.once('exit', resolve)), new Promise((resolve) => setTimeout(resolve, 5_000))]);
  if (child.exitCode === null && process.platform !== 'win32') child.kill('SIGKILL');
}

function assertBundleClean(internalUrl, key) {
  const staticRoot = path.join(frontend, '.next', 'static');
  function walk(directory) {
    for (const entry of readdirSync(directory, {withFileTypes: true})) {
      const target = path.join(directory, entry.name);
      if (entry.isDirectory()) walk(target);
      else {
        const contents = readFileSync(target);
        if (contents.includes(Buffer.from(key)) || contents.includes(Buffer.from(internalUrl))) {
          throw new Error('A server-only E2E value was found in the client production bundle.');
        }
      }
    }
  }
  walk(staticRoot);
}

async function main() {
  assertNodeVersion('E2E runner');
  const python = installLockedPython();
  apiPort = await reservePort();
  frontendPort = await reservePort();
  proxyPort = await reservePort();
  controlServer = createProxyServer();
  await new Promise((resolve, reject) => {
    controlServer.once('error', reject);
    controlServer.listen(proxyPort, '127.0.0.1', resolve);
  });
  const internalUrl = `http://127.0.0.1:${proxyPort}`;
  const api = start(python.command, [...python.prefix, '-m', 'uvicorn', 'src.api.main:app', '--host', '127.0.0.1', '--port', String(apiPort), '--log-level', 'error'], {
    API_KEY: serviceKey, APP_ENV: 'production', API_HOST: '127.0.0.1', API_PORT: String(apiPort), ALLOWED_ORIGINS: `http://127.0.0.1:${frontendPort}`, LOG_LEVEL: 'ERROR',
  }, 'api');
  await waitFor(`http://127.0.0.1:${apiPort}/api/v1/health`, api, 'FastAPI');
  if (interrupted) return;
  const golden = JSON.parse(readFileSync(path.join(root, 'tests/fixtures/model_release_golden.json'), 'utf8'));
  for (const vector of golden.vectors) {
    const warm = await fetch(`http://127.0.0.1:${apiPort}/api/v1/predictions/${vector.disease_identifier}`, {
      method: 'POST', headers: {'Content-Type': 'application/json', 'X-API-Key': serviceKey},
      body: JSON.stringify({measurements: vector.measurements}),
    });
    if (!warm.ok) throw new Error(`FastAPI model warm-up failed with HTTP ${warm.status}.`);
  }

  assertNodeVersion('Next.js production build');
  runSync(process.execPath, ['node_modules/next/dist/bin/next', 'build'], {
    cwd: frontend,
    env: {...process.env, BACKEND_INTERNAL_URL: internalUrl, BACKEND_API_KEY: serviceKey},
  });
  if (interrupted) return;
  assertBundleClean(internalUrl, serviceKey);
  assertNodeVersion('Next.js production server');
  const web = start(process.execPath, ['node_modules/next/dist/bin/next', 'start', '-H', '127.0.0.1', '-p', String(frontendPort)], {
    BACKEND_INTERNAL_URL: internalUrl, BACKEND_API_KEY: serviceKey, BACKEND_TIMEOUT_MS: '15000', NODE_ENV: 'production',
  }, 'next', frontend);
  const baseUrl = `http://127.0.0.1:${frontendPort}`;
  await waitFor(baseUrl, web, 'Next.js production server');

  const playwright = start(process.execPath, ['node_modules/@playwright/test/cli.js', 'test', '--config=playwright.config.ts', ...process.argv.slice(2)], {
    E2E_BASE_URL: baseUrl, E2E_CONTROL_URL: internalUrl, E2E_CONTROL_TOKEN: controlToken, API_KEY: '', BACKEND_API_KEY: '',
  }, 'playwright', frontend);
  const status = await new Promise((resolve) => playwright.once('exit', (code) => resolve(code ?? 1)));
  if (status !== 0 && !interrupted) process.exitCode = status;
}

for (const [signal, exitCode] of [['SIGINT', 130], ['SIGTERM', 143]]) {
  process.once(signal, () => {
    interrupted = true;
    process.exitCode = exitCode;
    void Promise.all([...children].map(closeChild));
  });
}

try {
  await main();
} catch (error) {
  process.exitCode = 1;
  console.error(error.message);
} finally {
  await Promise.all([...children].map(closeChild));
  if (controlServer?.listening) await new Promise((resolve) => controlServer.close(resolve));
  const resolvedTemp = path.resolve(temporaryPython);
  const tempRoot = path.resolve(tmpdir()) + path.sep;
  if (resolvedTemp.startsWith(tempRoot) && path.basename(resolvedTemp).startsWith(temporaryPrefix)) rmSync(resolvedTemp, {recursive: true, force: true});
}
