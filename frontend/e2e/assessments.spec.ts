import AxeBuilder from '@axe-core/playwright';
import {expect, test, type Page} from '@playwright/test';
import {readFileSync, readdirSync} from 'node:fs';
import path from 'node:path';
import goldenFixture from '../../tests/fixtures/model_release_golden.json';
import releaseManifest from '../../models/release_manifest.json';
import {diseaseContracts} from '../features/assessments/generated/contracts';
import {assessmentCopy} from '../features/assessments/assessment-copy';

type Disease = keyof typeof diseaseContracts;
type Golden = (typeof goldenFixture.vectors)[number];
const diseases = Object.keys(diseaseContracts) as Disease[];
const vectors = new Map(goldenFixture.vectors.map((vector) => [vector.disease_identifier, vector]));
const controlUrl = process.env.E2E_CONTROL_URL!;
const controlToken = process.env.E2E_CONTROL_TOKEN!;

function fixtureFor(disease: Disease): Golden {
  const fixture = vectors.get(disease);
  if (!fixture) throw new Error(`Missing golden input for ${disease}`);
  return fixture;
}

test.afterEach(async () => { await configureFault(null); });

async function configureFault(mode: string | null, disease?: string) {
  const response = await fetch(`${controlUrl}/__control/fault`, {
    method: 'POST',
    headers: {Authorization: `Bearer ${controlToken}`, 'Content-Type': 'application/json'},
    body: JSON.stringify({mode, disease}),
  });
  expect(response.ok).toBeTruthy();
}

async function proxyState() {
  const response = await fetch(`${controlUrl}/__control/state`, {headers: {Authorization: `Bearer ${controlToken}`} });
  expect(response.ok).toBeTruthy();
  return response.json() as Promise<{keySeen: boolean; paths: string[]; predictionBodies: string[]; reportCount: number}>;
}

async function startAssessment(page: Page, disease: Disease, options: {keyboardOnly?: boolean; navigate?: boolean} = {}) {
  if (options.navigate !== false) await page.goto(`/assessments/${disease}`);
  await expect(page.getByRole('heading', {level: 1})).toBeVisible();
  await expect(page.getByRole('button', {name: /Begin entering/})).toBeVisible();
  if (disease === 'parkinsons') await expect(page.getByText('Experimental', {exact: true}).first()).toBeVisible();
  const begin = page.getByRole('button', {name: /Begin entering/});
  if (options.keyboardOnly) { await tabUntil(page, begin); await page.keyboard.press('Enter'); }
  else await begin.click();
  await expect(page.getByRole('heading', {level: 1, name: diseaseContracts[disease].features[0].group})).toBeFocused();
}

async function tabUntil(page: Page, target: import('@playwright/test').Locator) {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    if (await target.evaluate((element) => element === document.activeElement).catch(() => false)) return;
    await page.keyboard.press('Tab');
  }
  throw new Error('The target control was not reachable in the keyboard tab order.');
}

async function fillStep(page: Page, disease: Disease, groupName: string, keyboardOnly = false) {
  const fields = diseaseContracts[disease].features.filter((field) => field.group === groupName);
  expect(fields.length).toBeGreaterThan(0);
  for (const field of fields) {
    const control = page.locator(`#${disease}-${field.display_order}`);
    await expect(control).toHaveCount(1);
    await expect(control).toBeVisible();
    const value = fixtureFor(disease).measurements[field.name];
    expect(value).not.toBeUndefined();
    if (field.control === 'select') {
      if (keyboardOnly) {
        await tabUntil(page, control);
        const index = field.options!.findIndex((option) => option.value === String(value));
        for (let current = 0; current <= index; current += 1) await page.keyboard.press('ArrowDown');
        await page.keyboard.press('Enter');
      } else await control.selectOption(String(value));
    } else if (keyboardOnly) {
      await tabUntil(page, control);
      await page.keyboard.press('Control+A');
      await page.keyboard.press('Backspace');
      await control.pressSequentially(String(value));
    } else await control.fill(String(value));
  }
}

async function fillAllSteps(page: Page, disease: Disease, keyboardOnly = false) {
  const groups = [...new Set(diseaseContracts[disease].features.map((field) => field.group))];
  for (const [index, group] of groups.entries()) {
    await expect(page.locator('.status-badge')).toContainText(disease === 'parkinsons' ? 'Experimental' : 'Educational / research');
    await expect(page.getByRole('heading', {level: 1, name: group})).toBeVisible();
    await expect(page.getByRole('progressbar', {name: `Step ${index + 1} of ${groups.length}`})).toBeVisible();
    if (index === 1) {
      const back = page.getByRole('button', {name: 'Back'});
      if (keyboardOnly) { await tabUntil(page, back); await page.keyboard.press('Enter'); }
      else await back.click();
      await expect(page.getByRole('heading', {level: 1, name: groups[0]})).toBeFocused();
      const firstField = diseaseContracts[disease].features.find((field) => field.group === groups[0])!;
      await expect(page.locator(`#${disease}-${firstField.display_order}`)).toHaveValue(String(fixtureFor(disease).measurements[firstField.name]));
      const forward = page.getByRole('button', {name: 'Continue'});
      if (keyboardOnly) { await tabUntil(page, forward); await page.keyboard.press('Enter'); }
      else await forward.click();
      await expect(page.getByRole('heading', {level: 1, name: group})).toBeFocused();
    }
    await fillStep(page, disease, group, keyboardOnly);
    const next = page.getByRole('button', {name: index === groups.length - 1 ? 'Review entries' : 'Continue'});
    if (keyboardOnly) {
      await tabUntil(page, next);
      await page.keyboard.press('Enter');
    } else await next.click();
    if (index < groups.length - 1) await expect(page.getByRole('heading', {level: 1})).toBeFocused();
  }
  await expect(page.getByRole('heading', {name: new RegExp(`Review .*${disease === 'parkinsons' ? 'voice biomarkers' : 'measurements'}`)})).toBeVisible();
}

function safeNoUnexpectedErrors(page: Page, errors: string[]) {
  const fatal = errors.filter((line) => !/favicon\.ico/i.test(line));
  expect(fatal, `Unexpected browser console or page errors: ${fatal.join('\n')}`).toEqual([]);
  return page.evaluate(() => ({
    localStorage: Object.keys(localStorage),
    sessionStorage: Object.keys(sessionStorage),
    requests: performance.getEntriesByType('resource').map((entry) => entry.name),
    html: document.documentElement.innerHTML,
    scriptText: Array.from(document.scripts).map((script) => script.outerHTML).join('\n'),
    createdUrls: (window as Window & {__createdObjectUrls?: string[]}).__createdObjectUrls ?? [],
    revokedUrls: (window as Window & {__revokedObjectUrls?: string[]}).__revokedObjectUrls ?? [],
  }));
}

async function scanAxe(page: Page) {
  const results = await new AxeBuilder({page})
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'])
    .analyze();
  const serious = results.violations.filter((violation) => ['serious', 'critical'].includes(violation.impact ?? ''));
  const other = results.violations.filter((violation) => !['serious', 'critical'].includes(violation.impact ?? ''));
  console.info(`[axe] ${new URL(page.url()).pathname}: ${serious.length} serious/critical, ${other.length} lower-impact findings${other.length ? ` (${other.map((item) => `${item.id}:${item.impact}`).join(', ')})` : ''}`);
  expect(serious.map((violation) => ({id: violation.id, impact: violation.impact, nodes: violation.nodes.map((node) => node.target)}))).toEqual([]);
}

function pdfPaths(directory: string): string[] {
  try {
    return readdirSync(directory, {withFileTypes: true}).flatMap((entry) => {
      const target = path.join(directory, entry.name);
      if (entry.isDirectory()) return ['.git', 'node_modules', '.next', 'test-results', '__pycache__', '.venv'].includes(entry.name) ? [] : pdfPaths(target);
      return entry.name.toLowerCase().endsWith('.pdf') ? [target] : [];
    });
  } catch { return []; }
}

for (const disease of diseases) {
  test(`${disease}: catalog to persisted prediction and PDF`, async ({page}) => {
    const errors: string[] = [];
    const consoleOutput: string[] = [];
    const apiResponses: {url: string; cache: string | undefined}[] = [];
    const browserRequests: {url: string; headers: Record<string, string>}[] = [];
    page.on('console', (message) => { consoleOutput.push(message.text()); if (message.type() === 'error') errors.push(message.text()); });
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('request', async (request) => browserRequests.push({url: request.url(), headers: await request.allHeaders()}));
    page.on('response', async (response) => {
      if (response.url().includes('/api/predictions/') || response.url().includes('/api/reports/')) {
        apiResponses.push({url: response.url(), cache: await response.headerValue('cache-control') ?? undefined});
      }
    });
    await page.addInitScript(() => {
      const created: string[] = [];
      const revoked: string[] = [];
      (window as Window & {__createdObjectUrls?: string[]}).__createdObjectUrls = created;
      (window as Window & {__revokedObjectUrls?: string[]}).__revokedObjectUrls = revoked;
      const create = URL.createObjectURL.bind(URL);
      const revoke = URL.revokeObjectURL.bind(URL);
      URL.createObjectURL = (blob) => { const value = create(blob); created.push(value); return value; };
      URL.revokeObjectURL = (value) => { revoked.push(value); revoke(value); };
    });

    await page.setViewportSize({width: 1440, height: 900});
    await page.goto('/');
    await expect(page).toHaveTitle(/Medical AI Suite/);
    await expectNoHorizontalOverflow(page);
    await expect(page.getByRole('heading', {name: 'Inspect educational model scores'})).toBeVisible();
    await page.getByRole('link', {name: 'Start an assessment'}).click();
    await expect(page.getByRole('heading', {name: 'Choose a disease module'})).toBeVisible();
    const catalogLink = page.locator(`a[href="/assessments/${disease}"]`);
    await expect(catalogLink).toBeVisible();
    await catalogLink.click();
    await expect(page).toHaveURL(new RegExp(`/assessments/${disease}$`));
    await expect(page.getByRole('heading', {name: assessmentCopy[disease].title})).toBeVisible();
    await expect(page.locator('.status-badge')).toContainText(disease === 'parkinsons' ? 'Experimental' : 'Educational / research');
    await scanAxe(page);

    const oldPdfs = new Set(pdfPaths(path.resolve(frontendRoot(), '..')));
    await startAssessment(page, disease);
    await fillAllSteps(page, disease);
    await expect(page.locator('.status-badge')).toContainText(disease === 'parkinsons' ? 'Experimental' : 'Educational / research');
    const fields = diseaseContracts[disease].features;
    for (const [index, field] of fields.entries()) {
      const expected = fixtureFor(disease).measurements[field.name];
      const row = page.locator('.review-list dt').nth(index).locator('xpath=following-sibling::dd[1]');
      await expect(row).toContainText(String(expected));
    }
    await scanAxe(page);
    const acknowledgement = page.getByRole('checkbox');
    await expect(acknowledgement).toBeVisible();
    const submit = page.getByRole('button', {name: 'Run prediction'});
    await expect(submit).toBeDisabled();
    if (disease === 'parkinsons') await expect(acknowledgement).toHaveAccessibleName(/7-subject holdout had ROC-AUC 0\.586 and specificity 0\.0/);
    await acknowledgement.check();
    await expect(submit).toBeEnabled();
    await scanAxe(page);
    await configureFault('prediction-delay', disease);
    await submit.click();
    await expect(page.getByRole('heading', {name: 'Preparing your result'})).toBeVisible();
    await expect(page.locator('.status-badge')).toContainText(disease === 'parkinsons' ? 'Experimental' : 'Educational / research');
    await configureFault(null);
    await expect(page.getByRole('heading', {name: new RegExp(`${disease === 'parkinsons' ? "Parkinson's" : disease} model threshold result`, 'i')})).toBeVisible();

    const golden = fixtureFor(disease);
    await expect(page.locator('.result-card')).toContainText(`Threshold class ${golden.expected_prediction}`);
    await expect(page.locator('.result-card')).toContainText(golden.expected_model_score.toFixed(4));
    await expect(page.locator('.result-card')).toContainText(golden.threshold.toFixed(4));
    await expect(page.locator('.result-card')).toContainText(golden.expected_model_score >= golden.threshold ? 'At or above' : 'Below');
    await expect(page.locator('.result-card')).toContainText(disease === 'parkinsons' ? 'Experimental' : 'Educational / research');
    await expect(page.locator('.result-card')).toContainText(golden.model_identifier);
    const release = releaseManifest.releases[disease];
    await expect(page.locator('.result-card')).toContainText(release.intended_use);
    for (const limitation of release.known_limitations) await expect(page.locator('.result-card')).toContainText(limitation);
    await expect(page.locator('.result-card')).toContainText('not a probability of disease, diagnosis, screening result');
    await scanAxe(page);
    await expectNoHorizontalOverflow(page);
    await page.setViewportSize({width: 390, height: 844});
    await expectNoHorizontalOverflow(page);
    await expect(page.getByRole('button', {name: 'Download PDF report'})).toBeVisible();
    await expect(page.locator('.result-facts')).toBeVisible();

    const beforeReports = (await proxyState()).reportCount;
    await configureFault('report-delay', disease);
    const report = page.getByRole('button', {name: 'Download PDF report'});
    const reportButton = await report.elementHandle();
    const downloadPromise = page.waitForEvent('download');
    await report.click();
    await expect(page.getByRole('button', {name: 'Generating report…'})).toBeDisabled();
    await reportButton?.evaluate((button) => (button as HTMLButtonElement).click());
    const download = await downloadPromise;
    await expect(download.suggestedFilename()).toBe(`${disease}_report.pdf`);
    const downloaded = await download.path();
    expect(downloaded).toBeTruthy();
    const header = readFileSync(downloaded!).subarray(0, 4).toString('ascii');
    expect(header).toBe('%PDF');
    expect(readFileSync(downloaded!).length).toBeGreaterThan(4);
    await configureFault(null);
    await expect.poll(async () => (await proxyState()).reportCount).toBe(beforeReports + 1);
    await expect.poll(async () => page.evaluate(() => (window as Window & {__revokedObjectUrls?: string[]}).__revokedObjectUrls?.length ?? 0)).toBeGreaterThan(0);

    const state = await safeNoUnexpectedErrors(page, errors);
    expect(state.localStorage).toEqual([]);
    expect(state.sessionStorage).toEqual([]);
    expect(state.html).not.toContain(controlUrl);
    expect(state.scriptText).not.toContain(controlUrl);
    expect(state.createdUrls).toEqual(state.revokedUrls);
    expect(state.requests.every((url) => new URL(url).origin === new URL(process.env.E2E_BASE_URL!).origin)).toBeTruthy();
    expect(browserRequests.every((request) => new URL(request.url).origin === new URL(process.env.E2E_BASE_URL!).origin)).toBeTruthy();
    expect(browserRequests.some((request) => Object.keys(request.headers).some((name) => name.toLowerCase() === 'x-api-key'))).toBeFalsy();
    expect(browserRequests.some((request) => request.url.includes('/api/v1/'))).toBeFalsy();
    expect(consoleOutput.every((line) => !/API_KEY|x-api-key|127\.0\.0\.1|\/api\/v1|Traceback/i.test(line))).toBeTruthy();
    expect(browserRequests.every((request) => !request.url.includes(JSON.stringify(golden.measurements)))).toBeTruthy();
    const values = Object.values(golden.measurements).map(String);
    const measurementKeys = new Set<string>(diseaseContracts[disease].features.map((field) => field.name));
    for (const request of browserRequests) {
      const url = new URL(request.url);
      expect([...url.searchParams.keys()].some((key) => measurementKeys.has(key))).toBeFalsy();
      expect([...url.searchParams.values()].some((value) => values.includes(value))).toBeFalsy();
    }
    expect(apiResponses.filter((response) => /\/api\/(predictions|reports)\//.test(response.url)).every((response) => response.cache === 'no-store')).toBeTruthy();
    const proxy = await proxyState();
    expect(proxy.keySeen).toBeTruthy();
    expect(proxy.paths).toContain(`POST /api/v1/predictions/${disease}`);
    expect(proxy.paths).toContain(`POST /api/v1/reports/${disease}`);
    expect(proxy.predictionBodies.map((body) => JSON.parse(body))).toContainEqual({measurements: golden.measurements});
    expect(new Set(pdfPaths(path.resolve(frontendRoot(), '..')))).toEqual(oldPdfs);

    await page.reload();
    await page.getByRole('button', {name: /Begin entering/}).click();
    const first = diseaseContracts[disease].features[0];
    await expect(page.locator(`#${disease}-${first.display_order}`)).toHaveValue('');
    await page.setViewportSize({width: 390, height: 844});
    await expectNoHorizontalOverflow(page);
    await expect(page.getByRole('link', {name: 'Assessments'})).toBeVisible();
    await expect(page.locator(`#${disease}-${first.display_order}`)).toBeVisible();
  });
}

function frontendRoot() { return path.resolve(process.cwd()); }

async function expectNoHorizontalOverflow(page: Page) {
  const sizes = await page.evaluate(() => ({width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth}));
  expect(sizes.scroll, `Horizontal overflow: ${sizes.scroll}px > ${sizes.width}px`).toBeLessThanOrEqual(sizes.width);
}

test('form errors block progression, identify the field, and reject numeric/category tampering', async ({page}) => {
  await startAssessment(page, 'liver');
  await page.getByRole('button', {name: 'Continue'}).click();
  const age = page.locator('#liver-0');
  await expect(age).toBeFocused();
  await expect(page.getByText('Age is required.')).toBeVisible();
  await expect(age).toHaveAttribute('aria-describedby', /liver-0-error/);
  await page.getByLabel('Age').fill('999');
  await page.getByLabel('Gender').selectOption('Male');
  await page.getByRole('button', {name: 'Continue'}).click();
  await expect(age).toBeFocused();
  await expect(page.getByText(/Enter no more than 120/)).toBeVisible();
  await page.getByLabel('Age').fill('45.5');
  await page.getByRole('button', {name: 'Continue'}).click();
  await expect(page.getByText('Age must be a whole number.')).toBeVisible();
  const gender = page.getByLabel('Gender');
  await gender.evaluate((element) => {
    const option = document.createElement('option'); option.value = 'invalid'; option.textContent = 'Invalid';
    (element as HTMLSelectElement).append(option); (element as HTMLSelectElement).value = 'invalid';
    element.dispatchEvent(new Event('input', {bubbles: true})); element.dispatchEvent(new Event('change', {bubbles: true}));
  });
  await page.getByRole('button', {name: 'Continue'}).click();
  await expect(page.getByText(/Choose gender/)).toBeVisible();
  await expect(page.getByRole('alert')).toBeVisible();
});

test('prediction failure, timeout, malformed response, and retry keep messages safe', async ({page}) => {
  for (const [mode, safeText] of [
    ['500', 'Model service is temporarily unavailable'],
    ['timeout', 'Backend service timed out'],
    ['malformed', 'The assessment could not be completed. Check your connection and try again.'],
  ] as const) {
    await configureFault(mode, 'liver');
    await startAssessment(page, 'liver');
    await fillAllSteps(page, 'liver');
    await page.getByRole('checkbox').check();
    await page.getByRole('button', {name: 'Run prediction'}).click();
    const alert = page.locator('.form-error[role="alert"]');
    await expect(alert).toContainText(safeText, {timeout: mode === 'timeout' ? 20_000 : 8_000});
    const visible = await alert.innerText();
    expect(visible).not.toMatch(/127\.0\.0\.1|\/api\/v1|Traceback|API_KEY|x-api-key|measurements|stack/i);
    await configureFault(null);
    await page.getByRole('button', {name: 'Run prediction'}).click();
    await expect(page.locator('.result-card')).toBeVisible();
  }
});

test('report failure is announced, and report retry is safe', async ({page}) => {
  await startAssessment(page, 'liver');
  await fillAllSteps(page, 'liver');
  await page.getByRole('checkbox').check();
  await page.getByRole('button', {name: 'Run prediction'}).click();
  await expect(page.locator('.result-card')).toBeVisible();
  await configureFault('report-failure', 'liver');
  await page.getByRole('button', {name: 'Download PDF report'}).click();
  await expect(page.locator('.result-card p[role="alert"]')).toContainText('The report could not be generated. Please try again.');
  await configureFault(null);
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', {name: 'Download PDF report'}).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('liver_report.pdf');
  expect(readFileSync((await download.path())!).subarray(0, 4).toString('ascii')).toBe('%PDF');
});

test('wrong BFF method is 405 and unknown disease is an actual 404', async ({page}) => {
  const methodResponse = await page.request.get('/api/predictions/liver');
  expect(methodResponse.status()).toBe(405);
  const unknownResponse = await page.goto('/assessments/not-a-disease');
  expect(unknownResponse?.status()).toBe(404);
  await scanAxe(page);
});

test('Axe covers landing, catalog, methodology, limitations and assessment states', async ({page}) => {
  await page.goto('/'); await scanAxe(page);
  await page.goto('/assessments'); await scanAxe(page);
  await page.goto('/methodology'); await scanAxe(page);
  await page.goto('/limitations'); await scanAxe(page);
  await page.goto('/assessments/parkinsons'); await scanAxe(page);
  await page.getByRole('button', {name: /Begin entering/}).click(); await scanAxe(page);
  await fillAllSteps(page, 'parkinsons'); await scanAxe(page);
  await page.getByRole('checkbox').check(); await page.getByRole('button', {name: 'Run prediction'}).click();
  await expect(page.locator('.result-card')).toBeVisible(); await scanAxe(page);
  const missing = await page.goto('/assessments/not-a-disease');
  expect(missing?.status()).toBe(404); await scanAxe(page);
});

test('keyboard-only liver assessment completes; focused controls show a visible focus ring', async ({page}) => {
  await page.goto('/');
  const start = page.getByRole('link', {name: 'Start an assessment'});
  await tabUntil(page, start);
  const outline = await start.evaluate((element) => getComputedStyle(element).outlineStyle);
  expect(outline).not.toBe('none');
  await page.keyboard.press('Enter');
  const firstModule = page.getByRole('link', {name: 'Start', exact: true}).first();
  await tabUntil(page, firstModule);
  await page.keyboard.press('Enter');
  await startAssessment(page, 'liver', {keyboardOnly: true, navigate: false});
  await fillAllSteps(page, 'liver', true);
  await tabUntil(page, page.getByRole('checkbox'));
  await page.keyboard.press('Space');
  await tabUntil(page, page.getByRole('button', {name: 'Run prediction'}));
  await page.keyboard.press('Enter');
  await expect(page.locator('.result-card')).toBeVisible();
  await tabUntil(page, page.getByRole('button', {name: 'Download PDF report'}));
  await page.keyboard.press('Enter');
  const download = await page.waitForEvent('download');
  expect(download.suggestedFilename()).toBe('liver_report.pdf');
});

test('Parkinson’s evidence stays visible and avoids screening or diagnosis claims', async ({page}) => {
  await page.setViewportSize({width: 390, height: 844});
  await page.goto('/assessments/parkinsons');
  const body = await page.locator('main').innerText();
  for (const evidence of ['195 sustained-vowel recordings', '32 people', '8 controls', '7 people', 'ROC-AUC 0.586', 'specificity 0.0', 'limited result is not clinical validation', 'Pre-computed biomarkers only']) {
    expect(body).toContain(evidence);
  }
  const clinicalStatements = body.split(/[.!?]/).filter((sentence) => /\b(screening|diagnos(?:is|e))\b/i.test(sentence));
  expect(clinicalStatements.every((sentence) => /\bnot\b|\bcannot\b/i.test(sentence))).toBeTruthy();
  await expectNoHorizontalOverflow(page);
  await expect(page.getByText('Experimental', {exact: true}).first()).toBeVisible();
});

test('desktop and mobile layout fit without clipped controls or result content', async ({page}) => {
  for (const viewport of [{width: 1440, height: 900}, {width: 390, height: 844}]) {
    await page.setViewportSize(viewport);
    for (const disease of diseases) {
      await startAssessment(page, disease);
      await expectNoHorizontalOverflow(page);
      for (const field of diseaseContracts[disease].features.filter((item) => item.group === diseaseContracts[disease].features[0].group)) {
        const rect = await page.locator(`#${disease}-${field.display_order}`).boundingBox();
        expect(rect).toBeTruthy();
        expect(rect!.x + rect!.width).toBeLessThanOrEqual(viewport.width);
      }
    }
  }
});
