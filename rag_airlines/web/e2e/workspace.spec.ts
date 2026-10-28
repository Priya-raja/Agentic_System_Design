import {test,expect} from '@playwright/test';
test('all pages, downloads, historical evals and verified SSE',async({page})=>{
 const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
 await page.setViewportSize({width:1440,height:1000});await page.goto('/');
 await expect(page.getByRole('heading',{name:'A little help. A lot more clarity.'})).toBeVisible();
 await page.route('**/api/chat/stream',route=>route.fulfill({headers:{'Content-Type':'text/event-stream'},body:[
 {event:'safety',action:'OK',ms:1},{event:'plan',calls:[{tool:'retrieve_policy',depends_on:[]}],ms:4},
 {event:'route',department:'baggage',ms:3},{event:'tool_start',tool:'retrieve_policy',call_id:0,ms:5},{event:'tool_end',tool:'retrieve_policy',call_id:0,ok:true,chunk_count:3,ms:15},
 {event:'verify',stage:'evidence',passed:true,ms:16},{event:'synth_start',ms:17},
 {event:'verify',stage:'synthesis',passed:true,ms:25},{event:'verify',stage:'final',passed:true,ms:26},
 {event:'token',text:'Verified test answer.'},{event:'result',result:{answer:'Verified test answer.',sources:[{document_id:'POL-BAG-2026',source_file:'POL-BAG-2026.pdf',page:1,authority:'primary',chunk_id:'B1',excerpt:'Verified test answer. Another paragraph in the policy.'}],kind:'answer',answerable:true,cache:'miss',latency_ms:27,verification:{passed:true,citation_errors:0,measurement_errors:0,safety_errors:0}}},
 {event:'done'}].map(event=>'data: '+JSON.stringify(event)+'\n\n').join('')}));
 await expect(page.getByRole('heading',{name:'Ask JetBlue a question'})).toBeVisible();
 await expect(page.getByRole('complementary',{name:'Request trace'})).toBeVisible();
 await page.screenshot({path:'../tmp/screens/chat-empty-desktop.png',fullPage:true,animations:'disabled'});
 await page.getByLabel('Your question').fill('How much baggage?');await page.getByRole('button',{name:'Send question'}).click();
 await expect(page.locator('.answer-visual')).toBeVisible();await expect(page.getByRole('complementary',{name:'Request trace'})).toHaveCount(0);
 await expect(page.locator('.answer-visual')).toHaveCount(1);await expect(page.locator('.answer-accessible')).toHaveCount(0);
 await expect(page.getByText('Verified · 1 primary source · 0.0s')).toBeVisible();
 await page.getByRole('button',{name:'View source 1: POL-BAG-2026'}).click();
 await expect(page.getByRole('heading',{name:'The evidence, in full.'})).toBeVisible();
 await expect(page.locator('.source-passage')).toHaveText('Verified test answer. Another paragraph in the policy.');
 await expect(page.locator('.source-passage mark')).toHaveText('Verified test answer.');
 await page.screenshot({path:'../tmp/screens/chat-source-desktop.png',fullPage:true,animations:'disabled'});
 await page.getByRole('button',{name:'Back to request trace'}).click();
 await expect(page.getByText('Final check',{exact:false})).toBeVisible();
 await expect(page.getByText('└─ 3 chunks retrieved')).toBeVisible();
 await expect(page.getByText('Route → baggage')).toBeVisible();
 await page.screenshot({path:'../tmp/screens/chat-desktop.png',fullPage:true,animations:'disabled'});
 await page.getByRole('button',{name:'New conversation'}).click();
 await expect(page.getByRole('heading',{name:'Ask JetBlue a question'})).toBeVisible();
 await expect(page.locator('.answer-visual').filter({hasText:'Verified test answer.'})).toHaveCount(0);
 await page.getByRole('button',{name:'Baggage allowance on Economy Flex in 2026'}).click();
 await expect(page.locator('.answer-visual').filter({hasText:'Verified test answer.'})).toBeVisible();
 await page.getByRole('link',{name:'Bookings & flights'}).click();
 await expect(page.getByText('AN0001K · AN201')).toBeVisible();
 const download=page.waitForEvent('download');await page.getByRole('link',{name:'Download booking PDF'}).click();expect((await download).suggestedFilename()).toContain('AN0001K');
 await page.getByRole('button',{name:'Check flight'}).click();await expect(page.getByText('11:15',{exact:true})).toBeVisible();
 await page.getByRole('link',{name:'Refunds & claims'}).click();
 const review=page.getByRole('button',{name:'Review simulation'});
 if(await review.count()) {await review.click();await expect(page.getByRole('dialog')).toBeVisible();await page.getByRole('button',{name:'Not now'}).click();await review.click();await page.getByRole('button',{name:'Confirm simulation'}).click();}
 await expect(page.getByRole('link',{name:'Download receipt'})).toBeVisible();
 await page.getByRole('link',{name:'Evaluations',exact:true}).click();await expect(page.getByText(/Historical evaluation|Bounded-agent evaluation sample/)).toBeVisible();await expect(page.getByText('Context precision',{exact:true})).toBeVisible();
 await page.getByLabel('Show failing cases').check();await page.screenshot({path:'../tmp/screens/evals-desktop.png',fullPage:true,animations:'disabled'});
 await page.getByRole('link',{name:'Architecture',exact:true}).click();await expect(page.getByRole('img',{name:'Safety, planning, typed tool execution, evidence verification, synthesis, final verification, then cache'})).toBeVisible();
 expect(errors).toEqual([]);
});
test('mobile layout and failure states',async({page})=>{
 await page.setViewportSize({width:390,height:844});await page.goto('/');await page.screenshot({path:'../tmp/screens/chat-mobile.png',fullPage:true,animations:'disabled'});
 expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
 await expect(page.getByRole('dialog')).toHaveCount(0);
 await page.getByRole('button',{name:'Show trace',exact:true}).click();
 await expect(page.getByRole('dialog')).toBeVisible();
 await page.screenshot({path:'../tmp/screens/trace-mobile.png',fullPage:true,animations:'disabled'});
 await page.keyboard.press('Escape');await expect(page.getByRole('dialog')).toHaveCount(0);
 await expect(page.getByRole('button',{name:'Show trace',exact:true})).toBeFocused();
 await page.route('**/api/chat/stream',route=>route.fulfill({status:503,json:{detail:'Test outage: please retry.'}}));
 await page.getByLabel('Your question').fill('Baggage?');await page.getByRole('button',{name:'Send question'}).click();await expect(page.getByText('Test outage: please retry.',{exact:true})).toBeVisible();
});

test('verification failures remain distinct and show real check counts',async({page})=>{
 await page.goto('/');
 await page.route('**/api/chat/stream',route=>route.fulfill({headers:{'Content-Type':'text/event-stream'},body:[
  {event:'verify',stage:'synthesis',passed:false,ms:10},
  {event:'result',result:{answer:'No unverified answer was released.',kind:'verification_failed',answerable:false,cache:'bypass',sources:[],latency_ms:11,verification:{passed:false,citation_errors:2,measurement_errors:1,safety_errors:0}}},
  {event:'done'}].map(event=>'data: '+JSON.stringify(event)+'\n\n').join('')}));
 await page.getByLabel('Your question').fill('How much baggage?');await page.getByRole('button',{name:'Send question'}).click();
 await expect(page.locator('.verification-failure')).toContainText('Couldn’t verify this answer');
 await expect(page.locator('.verification-failure')).toContainText('2 citation errors · 1 quantity error · 0 safety errors');
 await expect(page.locator('.verification-statement.verified')).toHaveCount(0);
 await expect(page.getByText('Answer checkFAIL')).toBeVisible();
 await page.screenshot({path:'../tmp/screens/chat-verification-failure.png',fullPage:true,animations:'disabled'});
});

test('single deterministic retrieval is shown on a laptop',async({page})=>{
 await page.setViewportSize({width:1200,height:800});await page.goto('/');
 expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(1200);
 await page.route('**/api/chat/stream',route=>route.fulfill({headers:{'Content-Type':'text/event-stream'},body:[
  {event:'safety',action:'OK',ms:1},
  {event:'route',classifier:'deterministic',scope:'public_information',department:'upgrades',ms:2},
  {event:'cache',kind:'exact',hit:false,ms:4},
  {event:'tool_start',tool:'retrieve_policy',call_id:0,depends_on:[],ms:5},
  {event:'tool_end',tool:'retrieve_policy',call_id:0,ok:true,chunk_count:3,ms:800},
  {event:'verify',stage:'evidence',passed:true,ms:810},
  {event:'verify',stage:'final',passed:true,ms:2000},
  {event:'result',result:{answer:'Policy answer.',kind:'answer',answerable:true,cache:'miss',sources:[],latency_ms:2001}},
  {event:'done'}].map(event=>'data: '+JSON.stringify(event)+'\n\n').join('')}));
 await page.getByLabel('Your question').fill('Can I upgrade my fare to Business?');await page.getByRole('button',{name:'Send question'}).click();
 await expect(page.locator('.answer-visual').filter({hasText:'Policy answer.'})).toBeVisible();
 await page.locator('.chat-top').getByRole('button',{name:'Show trace'}).click();
 await expect(page.getByRole('heading',{name:'EXECUTION LANE'})).toBeVisible();
 const lanes=page.locator('.execution-lane');await expect(lanes).toHaveCount(1);
 await expect(lanes.nth(0)).toContainText('3 chunks retrieved');await expect(lanes.nth(0).locator('time').last()).toHaveText('800');
 await page.screenshot({path:'../tmp/screens/chat-laptop-single-retrieval.png',fullPage:true,animations:'disabled'});
 await expect(page.locator('.chat-top').getByRole('button',{name:'Hide trace'})).toBeVisible();
 await expect(page.getByRole('heading',{name:'EXECUTION LANE'})).toBeVisible();
});

test('JetBlue cache hits show truthful ghost lanes and dotted timings',async({page})=>{
 await page.setViewportSize({width:1440,height:1000});await page.goto('/');
 await expect(page.getByRole('link',{name:'JetBlue demo',exact:true})).toBeVisible();
 const brand=page.locator('.brand > span');
 await expect(brand).toHaveCSS('background-color','rgba(0, 0, 0, 0)');
 const original={version:1,recorded_at:'2026-10-27T06:30:00Z',latency_ms:2000,events:[
  {event:'route',department:'upgrades, baggage',scope:'public_information',ms:4},
  {event:'plan',calls:[{tool:'retrieve_policy',depends_on:[]},{tool:'retrieve_policy',depends_on:[]}],ms:5},
  {event:'tool_start',tool:'retrieve_policy',call_id:0,depends_on:[],ms:6},
  {event:'tool_start',tool:'retrieve_policy',call_id:1,depends_on:[],ms:7},
  {event:'tool_end',tool:'retrieve_policy',call_id:0,ok:true,chunk_count:3,ms:1100},
  {event:'tool_end',tool:'retrieve_policy',call_id:1,ok:true,chunk_count:2,ms:800},
  {event:'verify',stage:'evidence',passed:true,ms:1101},
  {event:'verify',stage:'final',passed:true,ms:2000},
 ]};
 await page.route('**/api/chat/stream',route=>route.fulfill({headers:{'Content-Type':'text/event-stream'},body:[
  {event:'safety',action:'OK',ms:0},{event:'cache',kind:'exact',hit:true,ms:4},
  {event:'verify',stage:'cache',passed:true,ms:9},
  {event:'result',result:{answer:'Cached combined policy answer.',kind:'answer',answerable:true,cache:'exact',sources:[],latency_ms:10,original_execution:original}},
  {event:'done'}].map(event=>'data: '+JSON.stringify(event)+'\n\n').join('')}));
 await page.getByLabel('Your question').fill('Can I upgrade and bring extra baggage?');await page.getByRole('button',{name:'Send question'}).click();
 await expect(page.locator('.user-label')).toHaveText('YOU');
 await page.locator('.chat-top').getByRole('button',{name:'Show trace'}).click();
 const history=page.getByRole('region',{name:'Original cached execution'});
 await expect(history.getByText('Retrieved from cache',{exact:true})).toBeVisible();
 await expect(history.getByText(/Tools shown below did not run on this request/)).toBeVisible();
 await expect(history.locator('.execution-lane')).toHaveCount(2);
 await expect(history.locator('.ghost-trace')).toHaveCSS('opacity','0.5');
 await expect(page.locator('.trace-panel > .trace-phase .execution-lane')).toHaveCount(0);
 await expect(page.locator('.trace-panel > .trace-phase .timing-leader').first()).toHaveCSS('border-bottom-style','dotted');
 await expect(history.locator('time').filter({hasText:/^1100$/})).toBeVisible();
 await page.screenshot({path:'../tmp/screens/jetblue-cache-ghost.png',fullPage:true,animations:'disabled'});
});

test('a refusal explains retrieval candidates without displaying them as citations',async({page})=>{
 await page.goto('/');
 await expect(page.getByRole('button',{name:'What expenses can I claim for delayed baggage?'})).toBeVisible();
 await page.route('**/api/chat/stream',route=>route.fulfill({headers:{'Content-Type':'text/event-stream'},body:[
  {event:'tool_start',tool:'retrieve_policy',call_id:0,ms:5},
  {event:'tool_end',tool:'retrieve_policy',call_id:0,ok:true,chunk_count:4,ms:800},
  {event:'result',result:{answer:'The indexed documents cover baggage claims, not cancelled-ticket refunds.',kind:'refusal',answerable:false,cache:'miss',sources:Array.from({length:4},(_,i)=>({document_id:'POL-CLAIM-2026',chunk_id:`CLAIM-${i}`,authority:'primary',page:1})),latency_ms:1500}},
  {event:'done'}].map(event=>'data: '+JSON.stringify(event)+'\n\n').join('')}));
 await page.getByLabel('Your question').fill('Refund policy for a cancelled flight');await page.getByRole('button',{name:'Send question'}).click();
 await expect(page.locator('.abstention')).toContainText('The indexed documents cover baggage claims');
 await expect(page.locator('.abstention-note')).toContainText('They are not supporting citations.');
 await expect(page.locator('.source-links')).toHaveCount(0);
 await expect(page.locator('.inline-citations')).toHaveCount(0);
 await expect(page.getByText('└─ 4 chunks retrieved')).toBeVisible();
 await expect(page.locator('.trace-summary')).toContainText('4 retrieved chunks · 0 cited sources');
 await expect(page.getByText(/Answer withheld/)).toBeVisible();
 await page.screenshot({path:'../tmp/screens/jetblue-refusal.png',fullPage:true,animations:'disabled'});
});

test('an early safety response does not claim that evidence was retrieved',async({page})=>{
 await page.goto('/');
 await page.route('**/api/chat/stream',route=>route.fulfill({headers:{'Content-Type':'text/event-stream'},body:[
  {event:'safety',action:'BLOCK',ms:1},
  {event:'result',result:{answer:'Please use the dedicated secure workflow.',kind:'secure',answerable:false,cache:'bypass',sources:[],latency_ms:2}},
  {event:'done'}].map(event=>'data: '+JSON.stringify(event)+'\n\n').join('')}));
 await page.getByLabel('Your question').fill('Check my booking');await page.getByRole('button',{name:'Send question'}).click();
 await expect(page.locator('.abstention')).toContainText('dedicated secure workflow');
 await expect(page.locator('.abstention-note')).toHaveCount(0);
});
