import {test, expect, type Page} from '@playwright/test';

type StreamControls = {emitChat: (event: Record<string, unknown>) => void; closeChat: () => void};
async function controlledStream(page: Page) {
  await page.addInitScript(() => {
    const original = window.fetch.bind(window);
    window.fetch = (input, init) => {
      const url = input instanceof Request ? input.url : String(input);
      if (!url.endsWith('/api/chat/stream')) return original(input, init);
      const body = new ReadableStream({start(controller) {
        const controls = window as unknown as StreamControls;
        controls.emitChat = event => controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`));
        controls.closeChat = () => controller.close();
        init?.signal?.addEventListener('abort', () => controller.error(new DOMException('Stopped', 'AbortError')));
      }});
      return Promise.resolve(new Response(body, {headers: {'Content-Type': 'text/event-stream'}}));
    };
  });
}
async function emit(page: Page, event: Record<string, unknown>) {
  await page.evaluate(value => (window as unknown as StreamControls).emitChat(value), event);
}
async function finish(page: Page) {
  await page.evaluate(() => (window as unknown as StreamControls).closeChat());
}
function result(answer: string) {
  return {event: 'result', result: {answer, kind: 'answer', answerable: true, cache: 'miss', sources: [], latency_ms: 1234,
    verification: {passed: true, citation_errors: 0, measurement_errors: 0, safety_errors: 0}}};
}
async function start(page: Page, question = 'What is the policy?') {
  await page.goto('/');
  await page.getByLabel('Your question').fill(question);
  await page.getByRole('button', {name: 'Send question'}).click();
  await expect(page.getByRole('status')).toContainText('Checking the details…');
}

test('progress follows live events while text waits for release, then renders safe Markdown', async ({page}) => {
  await page.setViewportSize({width: 1440, height: 1000});
  await controlledStream(page); await start(page);
  await expect(page.locator('.answer-skeleton > span')).toHaveCount(3);
  await emit(page, {event:'token', text:'UNVERIFIED SENTINEL'});
  await expect(page.getByText('UNVERIFIED SENTINEL')).toHaveCount(0);
  await expect(page.locator('.answer-content')).toHaveCount(0);
  await emit(page, {event:'route', department:'cabin_onboard', ms:4});
  await expect(page.getByRole('status')).toContainText('Routing to cabin_onboard…');
  await emit(page, {event:'tool_start', tool:'retrieve_policy', call_id:0, ms:5});
  await expect(page.getByRole('status')).toContainText('Searching policy documents…');
  await emit(page, {event:'tool_end', tool:'retrieve_policy', call_id:0, ok:true, chunk_count:3, ms:500});
  await expect(page.getByRole('status')).toContainText('Reviewing 3 retrieved passages…');
  await emit(page, {event:'synth_start', ms:501});
  await expect(page.getByRole('status')).toContainText('Preparing an answer from the evidence…');
  await page.screenshot({path:'../tmp/screens/jetblue-waiting.png', fullPage:true, animations:'disabled'});
  await emit(page, {event:'verify', stage:'synthesis', passed:true, ms:1000});
  await expect(page.getByRole('status')).toContainText('Verifying citations…');
  await expect(page.locator('.answer-content')).toHaveCount(0);
  await emit(page, {event:'verify', stage:'final', passed:true, ms:1001});
  await expect(page.getByRole('status')).toContainText('Completing final checks…');
  await emit(page, result('Here are the requirements:\n\n- Keep the **bag tag**.\n- Keep itemised receipts.\n\nAsk staff about the next steps.\n\n<img src="x" onerror="window.bad=true">'));
  const last = page.locator('.answer-visual .reveal-character').last();
  await last.waitFor({state:'attached'});
  expect(Number(await last.evaluate(el => getComputedStyle(el).opacity))).toBeLessThan(1);
  await expect(last).toHaveCSS('opacity', '1', {timeout:1500});
  await finish(page);
  await expect(page.getByRole('status')).toHaveCount(0);
  await expect(page.locator('.answer-visual ul > li')).toHaveCount(2);
  await expect(page.locator('.answer-visual li + li')).toHaveCSS('margin-top', '8px');
  await expect(page.locator('.answer-visual strong')).toHaveText('bag tag');
  await expect(page.locator('.answer-visual img')).toHaveCount(0);
  expect(await page.evaluate(() => (window as Window & {bad?:boolean}).bad)).toBeUndefined();
  await expect(page.locator('.answer-visual')).toContainText('Keep the bag tag.');
  await expect(page.locator('.answer-accessible')).toHaveCount(0);
  await expect(page.locator('.user-message')).toBeVisible();
  await expect(page.locator('.brand-pitch strong')).toHaveCSS('font-weight', '700');
  await page.keyboard.press('Tab');
  const toggle=page.locator('.chat-top .trace-toggle'); await toggle.focus();
  await expect(toggle).toHaveCSS('outline-color', 'rgb(0, 180, 216)');
  await expect(toggle).toHaveCSS('outline-width', '2px');
  await page.screenshot({path:'../tmp/screens/jetblue-markdown-answer.png', fullPage:true, animations:'disabled'});
});

test('reduced motion reveals the full answer immediately', async ({page}) => {
  await page.emulateMedia({reducedMotion:'reduce'});
  await controlledStream(page); await start(page);
  await emit(page, result('A fully released answer with several words.'));
  await expect(page.locator('.answer-visual .reveal-character').last()).toHaveCSS('opacity', '1');
  await expect(page.locator('.answer-visual')).toHaveCSS('animation-name', 'none');
  await finish(page);
});

test('a long answer keeps the current user bubble above it in the conversation viewport', async ({page}) => {
  await page.setViewportSize({width:1200,height:800});
  await page.emulateMedia({reducedMotion:'reduce'});
  await controlledStream(page); await start(page, 'Please explain the policy.');
  const answer=Array.from({length:18}, (_,i)=>`- Policy detail ${i+1}: keep your receipts and ask the airport team about the applicable conditions.`).join('\n');
  await emit(page, result(answer)); await finish(page);
  await expect(page.locator('.answer-visual li')).toHaveCount(18);
  const question=await page.locator('.user-turn').boundingBox(), log=await page.locator('.messages').boundingBox();
  expect(question!.y).toBeGreaterThanOrEqual(log!.y);
  expect(question!.y+question!.height).toBeLessThanOrEqual(log!.y+log!.height);
  await page.screenshot({path:'../tmp/screens/jetblue-long-answer.png',fullPage:true,animations:'disabled'});
});

test('a failed gate never reveals received token text or starts an answer animation', async ({page}) => {
  await controlledStream(page); await start(page);
  await emit(page, {event:'token', text:'THIS MUST NEVER APPEAR'});
  await emit(page, {event:'verify', stage:'synthesis', passed:false, ms:800});
  await expect(page.getByRole('status')).toContainText('A verification check needs review…');
  await emit(page, {event:'result', result:{answer:'No unverified answer was released.',kind:'verification_failed',answerable:false,cache:'bypass',sources:[],latency_ms:801}});
  await finish(page);
  await expect(page.locator('.verification-failure')).toContainText('Couldn’t verify this answer');
  await expect(page.locator('.answer-content')).toHaveCount(0);
  await expect(page.getByText('THIS MUST NEVER APPEAR')).toHaveCount(0);
});
