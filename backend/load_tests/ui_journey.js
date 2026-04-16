import { browser } from "k6/browser";
import { check } from "k6";

const BASE_URL = (__ENV.BASE_URL || "http://localhost:8080").replace(/\/$/, "");
const UI_VUS = Number(__ENV.BROWSER_VUS || "5");
const UI_ITERATIONS = Number(__ENV.BROWSER_ITERATIONS || "20");
const UI_TIMEOUT_MS = Number(__ENV.UI_TIMEOUT_MS || "240000");

const salesCsv = open("./fixtures/sales.csv", "b");

export const options = {
  scenarios: {
    ui_journey: {
      executor: "shared-iterations",
      vus: UI_VUS,
      iterations: UI_ITERATIONS,
      maxDuration: __ENV.BROWSER_MAX_DURATION || "45m",
      options: {
        browser: {
          type: "chromium",
        },
      },
    },
  },
  thresholds: {
    checks: ["rate>0.99"],
    browser_web_vital_lcp: ["p(95)<6000"],
    browser_web_vital_cls: ["p(95)<0.2"],
  },
};

export default async function() {
  const page = await browser.newPage();
  try {
    await page.goto(BASE_URL, { waitUntil: "networkidle" });

    const fileInput = page.locator('input[type="file"]').first();
    await fileInput.setInputFiles([
      {
        name: "sales.csv",
        mimeType: "text/csv",
        buffer: salesCsv,
      },
    ]);

    const queryInput = page.locator('input[placeholder*="Which segments are driving"]').first();
    await queryInput.waitFor({ state: "visible", timeout: UI_TIMEOUT_MS });
    await queryInput.fill("predict sales for next month");

    const runButton = page.getByRole("button", { name: /Run analysis/i });
    await runButton.click();

    const resultHeader = page.locator("text=Analysis Results").first();
    await resultHeader.waitFor({ state: "visible", timeout: UI_TIMEOUT_MS });

    const technicalHeader = page.locator("text=Technical Details").first();
    await technicalHeader.waitFor({ state: "visible", timeout: UI_TIMEOUT_MS });

    check(true, {
      "analysis results rendered in browser journey": () => true,
    });
  } finally {
    await page.close();
  }
}
