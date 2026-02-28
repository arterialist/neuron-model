// @ts-check
const { test, expect } = require('@playwright/test');

const BASE_URL = process.env.PIPELINE_URL || 'http://localhost:8765';

test.describe('Pipeline E2E Browser', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL);
  });

  test('dashboard loads with stats and job list', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Pipeline');
    await expect(page.locator('#stat-total')).toBeVisible();
    await expect(page.locator('#stat-running')).toBeVisible();
    await expect(page.locator('#stat-completed')).toBeVisible();
    await expect(page.locator('#stat-failed')).toBeVisible();
    await expect(page.locator('#jobs-container')).toBeVisible();
  });

  test('Upload Config opens YAML submit modal', async ({ page }) => {
    const yamlContent = `job_name: "e2e_test"
network:
  source: "build"
  build_config:
    dataset: "mnist"
    layers:
      - type: "dense"
        size: 10
        connectivity: 0.5
activity_recording:
  dataset: "mnist"
  ticks_per_image: 2
  images_per_label: 1
data_preparation:
  feature_types: ["avg_S"]
  train_split: 0.8
training:
  epochs: 1
  batch_size: 4
evaluation:
  samples: 2
  window_size: 2
  dataset_name: "mnist"
`;
    const path = require('path');
    const fs = require('fs');
    const tmpDir = path.join(process.cwd(), 'pipeline', 'tests', 'tmp');
    fs.mkdirSync(tmpDir, { recursive: true });
    const tmpFile = path.join(tmpDir, 'e2e_upload_test.yaml');
    fs.writeFileSync(tmpFile, yamlContent);
    try {
      const fileInput = page.locator('#config-upload');
      await fileInput.setInputFiles(tmpFile);
      await expect(page.locator('#yaml-submit-modal')).toBeVisible({ timeout: 5000 });
      await expect(page.locator('#yaml-submit-textarea')).toHaveValue(/job_name/);
    } finally {
      fs.unlinkSync(tmpFile);
    }
  });

  test('YAML submit creates job', async ({ page }) => {
    const initialCount = await page.locator('.job-row').count();
    const yaml = `job_name: "e2e_yaml_submit"
network:
  source: "build"
  build_config:
    dataset: "mnist"
    layers:
      - type: "dense"
        size: 10
        connectivity: 0.5
activity_recording:
  dataset: "mnist"
  ticks_per_image: 2
  images_per_label: 1
data_preparation:
  feature_types: ["avg_S"]
  train_split: 0.8
training:
  epochs: 1
  batch_size: 4
evaluation:
  samples: 2
  window_size: 2
  dataset_name: "mnist"
`;
    const fileInput = page.locator('#config-upload');
    await fileInput.setInputFiles({
      name: 'e2e.yaml',
      mimeType: 'application/x-yaml',
      buffer: Buffer.from(yaml),
    });
    await expect(page.locator('#yaml-submit-modal')).toBeVisible();
    await page.locator('#yaml-submit-textarea').fill(yaml);
    await page.locator('#yaml-submit-btn').click();
    await expect(page.locator('#yaml-submit-modal')).not.toBeVisible();
    await expect(page.locator('.job-row')).toHaveCount(initialCount + 1, { timeout: 10000 });
    await expect(page.getByText('e2e_yaml_submit').first()).toBeVisible();
  });

  test('New Job opens wizard', async ({ page }) => {
    await page.locator('button:has-text("New Job")').click();
    await expect(page.locator('#submit-modal')).toBeVisible();
    await expect(page.locator('#step-1')).toBeVisible();
    await expect(page.locator('#wiz-job-name')).toBeVisible();
  });

  test('Wizard produces valid config and submits', async ({ page }) => {
    await page.locator('button:has-text("New Job")').click();
    await expect(page.locator('#submit-modal')).toBeVisible();
    await page.locator('#wiz-job-name').fill('e2e_wizard_job');
    await page.locator('#wiz-next-btn').click();
    await expect(page.locator('#step-2')).toBeVisible();
    await page.locator('input[name="net-source"][value="build"]').check();
    await page.locator('#wiz-build-dataset').selectOption('mnist');
    await page.locator('button:has-text("Add Layer")').click();
    await page.locator('#wiz-layers-list .layer-type').selectOption('dense');
    await page.locator('#wiz-layers-list .layer-size').fill('10');
    await page.locator('#wiz-layers-list .layer-conn').fill('0.5');
    await page.locator('#wiz-next-btn').click();
    await expect(page.locator('#step-3')).toBeVisible();
    await page.locator('#wiz-rec-images').fill('1');
    await page.locator('#wiz-next-btn').click();
    await expect(page.locator('#step-4')).toBeVisible();
    await page.locator('#wiz-next-btn').click();
    await expect(page.locator('#step-5')).toBeVisible();
    await page.locator('#wiz-train-epochs').fill('1');
    await page.locator('#wiz-next-btn').click();
    await page.locator('#wiz-eval-samples').fill('2');
    await page.locator('#wiz-next-btn').click();
    await expect(page.locator('#step-7')).toBeVisible();
    await page.locator('#wiz-next-btn').click();
    await expect(page.locator('#step-8')).toBeVisible();
    await page.locator('button:has-text("Switch to Raw YAML")').click();
    await expect(page.locator('#wiz-review-raw')).toBeVisible();
    await expect(page.locator('#wiz-yaml-output')).toHaveValue(/job_name/, { timeout: 5000 });
    await expect(page.locator('#wiz-yaml-output')).toHaveValue(/e2e_wizard_job/);
    await page.locator('#submit-modal button:has-text("Submit Job")').click();
    await expect(page.locator('#submit-modal')).not.toBeVisible({ timeout: 5000 });
  });

  test('Job detail page shows step chain and controls', async ({ page }) => {
    const yaml = `job_name: "e2e_detail_test"
network:
  source: "build"
  build_config:
    dataset: "mnist"
    layers: [{ type: "dense", size: 10, connectivity: 0.5 }]
activity_recording: { dataset: "mnist", ticks_per_image: 2, images_per_label: 1 }
data_preparation: { feature_types: ["avg_S"], train_split: 0.8 }
training: { epochs: 1, batch_size: 4 }
evaluation: { samples: 2, window_size: 2, dataset_name: "mnist" }
`;
    const res = await page.request.post(`${BASE_URL}/api/jobs`, {
      data: { config_yaml: yaml },
    });
    const { job_id } = await res.json();
    await page.goto(`${BASE_URL}/jobs/${job_id}`);
    await expect(page.locator('.header-info')).toBeVisible();
    await expect(page.locator('.step-item')).toHaveCount(6, { timeout: 10000 });
    await page.waitForSelector('.status-completed, .status-failed', { timeout: 120000 });
  });

  test('Download all artifacts when job completes', async ({ page }) => {
    const yaml = `job_name: "e2e_download_test"
network:
  source: "build"
  build_config:
    dataset: "mnist"
    layers: [{ type: "dense", size: 10, connectivity: 0.5 }]
activity_recording: { dataset: "mnist", ticks_per_image: 2, images_per_label: 1 }
data_preparation: { feature_types: ["avg_S"], train_split: 0.8 }
training: { epochs: 1, batch_size: 4 }
evaluation: { samples: 2, window_size: 2, dataset_name: "mnist" }
`;
    const res = await page.request.post(`${BASE_URL}/api/jobs`, {
      data: { config_yaml: yaml },
    });
    const { job_id } = await res.json();
    await page.goto(`${BASE_URL}/jobs/${job_id}`);
    await page.waitForSelector('.status-completed, .status-failed', { timeout: 120000 });
    const downloadPromise = page.waitForEvent('download', { timeout: 5000 });
    await page.locator('a:has-text("Download All")').click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/\.tar\.gz$/);
  });

  test('Job detail page shows Pause/Resume/Cancel when running', async ({ page }) => {
    const yaml = `job_name: "e2e_controls"
network:
  source: "build"
  build_config:
    dataset: "mnist"
    layers: [{ type: "dense", size: 10, connectivity: 0.5 }]
activity_recording: { dataset: "mnist", ticks_per_image: 2, images_per_label: 1 }
data_preparation: { feature_types: ["avg_S"], train_split: 0.8 }
training: { epochs: 1, batch_size: 4 }
evaluation: { samples: 2, window_size: 2, dataset_name: "mnist" }
`;
    const res = await page.request.post(`${BASE_URL}/api/jobs`, {
      data: { config_yaml: yaml },
    });
    const { job_id } = await res.json();
    await page.goto(`${BASE_URL}/jobs/${job_id}`);
    await expect(page.locator('#job-status-badge')).toBeVisible({ timeout: 10000 });
    const pauseBtn = page.locator('button:has-text("Pause")');
    const cancelBtn = page.locator('button:has-text("Cancel")');
    if (await pauseBtn.isVisible()) {
      expect(await cancelBtn.isVisible()).toBe(true);
    }
    await page.waitForSelector('#job-status-badge.status-completed, #job-status-badge.status-failed, #job-status-badge.status-cancelled', { timeout: 120000 });
  });
});
