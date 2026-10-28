import {test, expect} from '@playwright/test';

test('human review survives reload and requires an explicit approve or reject', async ({page}) => {
  const id = 'a'.repeat(32);
  const claim = {claim_id:'REF-DEMO-001', claim_status:'approved', refund_status:'ready', approved_amount_aed:'120.00', can_proceed:true, reason:'Reviewed fictional claim.'};
  let status = 'pending', decisionCalls = 0;
  await page.route('**/api/overview', route => route.fulfill({json:{claims:[claim], bookings:[], demo:true, csrf_token:'fixture'}}));
  await page.route('**/api/refunds/REF-DEMO-001/review', route => route.fulfill({json:{workflow_id:id, claim_id:claim.claim_id, status, review:claim}}));
  await page.route('**/api/refund-reviews/'+id, route => route.fulfill({json:{workflow_id:id, claim_id:claim.claim_id, status, review:claim}}));
  await page.route('**/api/refund-reviews/'+id+'/decision', async route => {
    decisionCalls++;
    const approved = route.request().postDataJSON().approved;
    status = approved ? 'completed' : 'rejected';
    await route.fulfill({json:{workflow_id:id, claim_id:claim.claim_id, status, review:claim, result:{message:approved?'Simulation recorded.':'Review rejected. No refund was recorded.'}}});
  });
  await page.goto('/refunds');
  await page.getByRole('button', {name:'Review simulation', exact:true}).click();
  await expect(page.getByRole('dialog')).toContainText('Paused for your decision');
  expect(decisionCalls).toBe(0);
  await page.getByRole('button', {name:'Not now'}).click();
  await page.reload();
  await page.getByRole('button', {name:'Resume review', exact:true}).click();
  await expect(page.getByRole('dialog')).toContainText('AED 120.00');
  expect(decisionCalls).toBe(0);
  await page.screenshot({path:'../tmp/screens/langgraph-review.png', fullPage:true, animations:'disabled'});
  await page.getByRole('button', {name:'Reject simulation'}).click();
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(page.getByRole('status')).toContainText('Review rejected');
  expect(decisionCalls).toBe(1);
});
