import {readFileSync} from 'node:fs';
import path from 'node:path';

const origin = process.env.CONTAINER_BASE_URL || 'http://127.0.0.1:3000';
const root = path.resolve(import.meta.dirname, '..');
const fixture = JSON.parse(readFileSync(path.join(root, 'tests/fixtures/model_release_golden.json'), 'utf8'));
const manifest = JSON.parse(readFileSync(path.join(root, 'models/release_manifest.json'), 'utf8'));

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function request(pathname, body) {
  return fetch(new URL(pathname, origin), body ? {
    method: 'POST', headers: {'content-type': 'application/json'},
    body: JSON.stringify({measurements: body}),
  } : {cache: 'no-store'});
}

async function verifyGoldensAndReports() {
  for (const vector of fixture.vectors) {
    const {disease_identifier: disease, measurements} = vector;
    const prediction = await request(`/api/predictions/${disease}`, measurements);
    assert(prediction.ok, `prediction request failed for ${disease}: ${prediction.status}`);
    const result = await prediction.json();
    assert(result.prediction === vector.expected_prediction, `prediction mismatch for ${disease}`);
    assert(Math.abs(result.model_score - vector.expected_model_score) <= fixture.score_tolerance, `score mismatch for ${disease}`);
    assert(result.decision_threshold === vector.threshold, `threshold mismatch for ${disease}`);
    assert(result.model_identifier === vector.model_identifier, `model identifier mismatch for ${disease}`);
    assert(result.release_status === manifest.releases[disease].release_status, `release metadata mismatch for ${disease}`);

    const report = await request(`/api/reports/${disease}`, measurements);
    assert(report.ok, `report request failed for ${disease}: ${report.status}`);
    assert((report.headers.get('content-type') || '').includes('application/pdf'), `wrong report content type for ${disease}`);
    const bytes = Buffer.from(await report.arrayBuffer());
    assert(bytes.length > 4 && bytes.subarray(0, 4).toString('ascii') === '%PDF', `invalid PDF for ${disease}`);
    assert(!bytes.toString('latin1').includes(JSON.stringify(measurements)), `PDF unexpectedly contains raw serialized input for ${disease}`);
  }
}

async function boundedLoadSmoke() {
  const diseases = fixture.vectors;
  const jobs = Array.from({length: 30}, (_, index) => async () => {
    const vector = diseases[index % diseases.length];
    const response = await request(`/api/predictions/${vector.disease_identifier}`, vector.measurements);
    if (!response.ok) throw new Error(`bounded load request failed: ${response.status}`);
    const payload = await response.json();
    if (payload.prediction !== vector.expected_prediction) throw new Error('bounded load response validation failed');
  });
  let cursor = 0;
  const workers = Array.from({length: 5}, async () => {
    while (cursor < jobs.length) await jobs[cursor++]();
  });
  await Promise.all(workers);
}

async function main() {
  const health = await request('/api/health');
  assert(health.ok, `frontend health failed: ${health.status}`);
  const provenance = await health.json();
  assert(provenance.version === '0.9.0-rc.1', 'application version mismatch');
  assert(provenance.model_release?.identifier, 'model release identifier missing');
  const hiddenMetrics = await request('/api/internal/metrics');
  assert(hiddenMetrics.status === 404, 'metrics must not be routed through the browser BFF');
  const readiness = await request('/api/ready');
  assert(readiness.ok && (await readiness.json()).status === 'ready', 'readiness failed');
  assert(Object.keys(manifest.releases).length === 5, 'model release manifest disease count mismatch');
  await verifyGoldensAndReports();
  await boundedLoadSmoke();
  const recovered = await request('/api/ready');
  assert(recovered.ok && (await recovered.json()).status === 'ready', 'readiness failed after smoke traffic');
  console.log('CONTAINER_SMOKE PASS: provenance, readiness, 5 golden predictions, 5 PDFs, 30 validated concurrent requests, recovery');
}

main().catch((error) => {
  // Keep diagnostics free of request bodies, response scores, service keys and URLs.
  console.error(`CONTAINER_SMOKE FAIL: ${error instanceof Error ? error.message.replace(/https?:\/\/\S+/g, '[url]') : 'unknown error'}`);
  process.exitCode = 1;
});
