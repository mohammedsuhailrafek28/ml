import {expect, test, type Page} from '@playwright/test';
import {readFileSync} from 'node:fs';
import golden from '../../tests/fixtures/model_release_golden.json';
import {diseaseContracts} from '../features/assessments/generated/contracts';

type Disease = keyof typeof diseaseContracts;
const vectors = new Map(golden.vectors.map((vector) => [vector.disease_identifier, vector]));

async function completeAssessment(page: Page, disease: Disease) {
  const vector = vectors.get(disease);
  if (!vector) throw new Error(`Missing public golden fixture for ${disease}`);
  await page.goto(`/assessments/${disease}`);
  await page.getByRole('button', {name: /Begin entering/}).click();
  const groups = [...new Set(diseaseContracts[disease].features.map((field) => field.group))];
  for (const [index, group] of groups.entries()) {
    await expect(page.getByRole('heading', {level: 1, name: group})).toBeVisible();
    for (const field of diseaseContracts[disease].features.filter((item) => item.group === group)) {
      const control = page.locator(`#${disease}-${field.display_order}`);
      const value = vector.measurements[field.name];
      if (field.control === 'select') await control.selectOption(String(value));
      else await control.fill(String(value));
    }
    await page.getByRole('button', {name: index === groups.length - 1 ? 'Review entries' : 'Continue'}).click();
  }
  await page.getByRole('checkbox').check();
  await page.getByRole('button', {name: 'Run prediction'}).click();
  await expect(page.getByRole('heading', {name: new RegExp(`${disease === 'parkinsons' ? "Parkinson's" : disease} model threshold result`, 'i')})).toBeVisible();
  await expect(page.locator('.result-card')).toContainText(`Threshold class ${vector.expected_prediction}`);
  await expect(page.locator('.result-card')).toContainText(vector.expected_model_score.toFixed(4));
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', {name: 'Download PDF report'}).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe(`${disease}_report.pdf`);
  const file = await download.path();
  expect(file).toBeTruthy();
  const bytes = readFileSync(file!);
  expect(bytes.subarray(0, 4).toString('ascii')).toBe('%PDF');
  expect(bytes.length).toBeGreaterThan(4);
}

for (const disease of Object.keys(diseaseContracts) as Disease[]) {
  test(`container ${disease} assessment and report`, async ({page}) => {
    const urls: string[] = [];
    page.on('request', (request) => urls.push(request.url()));
    await completeAssessment(page, disease);
    const vector = vectors.get(disease)!;
    const fieldNames = diseaseContracts[disease].features.map((field) => field.name);
    for (const rawUrl of urls) {
      const url = new URL(rawUrl);
      // Next.js may add its internal React Server Components cache token.
      // User-entered measurements must never appear in the URL.
      for (const parameter of url.searchParams.keys()) {
        expect(parameter).toBe('_rsc');
      }
      expect(url.hash).toBe('');
      for (const field of fieldNames) expect(url.searchParams.has(field)).toBeFalsy();
      expect(rawUrl).not.toContain(JSON.stringify(vector.measurements));
    }
    expect(urls.every((url) => new URL(url).origin === new URL(process.env.E2E_BASE_URL!).origin)).toBeTruthy();
    const state = await page.evaluate(() => ({
      local: Object.keys(localStorage),
      session: Object.keys(sessionStorage),
      html: document.documentElement.innerHTML,
      scripts: Array.from(document.scripts, (script) => script.textContent || '').join('\n'),
    }));
    expect(state.local).toEqual([]);
    expect(state.session).toEqual([]);
    expect(state.html).not.toContain('http://api:8000');
    expect(state.scripts).not.toContain('http://api:8000');
    expect(state.html).not.toContain('service_api_key');
    expect(state.scripts).not.toContain('BACKEND_API_KEY');
  });
}
