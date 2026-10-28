'use client';

import {useEffect, useState} from 'react';
import {Receipt, ArrowRight, Download, Check, ShieldCheck, X, PauseCircle} from 'lucide-react';
import {Button} from '@/components/ui/button';
import {ConfirmDialog} from '@/components/ui/dialog';
import {api} from '@/lib/utils';

type Claim = {claim_id:string; claim_status:string; refund_status:string; approved_amount_aed:string|null; can_proceed:boolean; reason:string};
type Review = {workflow_id:string; claim_id:string; status:'pending'|'completed'|'rejected'; review:Claim|null; result?:{message?:string}};
const storageKey = 'aeronova.refund-reviews';

export default function Refunds() {
  const [claims, setClaims] = useState<Claim[]>([]);
  const [token, setToken] = useState('');
  const [selected, setSelected] = useState<Review|null>(null);
  const [pending, setPending] = useState<Record<string,string>>({});
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');

  function savePending(value:Record<string,string>) {
    setPending(value);
    // Only opaque workflow handles are stored here; the server checks ownership.
    try { localStorage.setItem(storageKey, JSON.stringify(value)); } catch { /* Optional convenience. */ }
  }
  async function load() {
    try {
      const data = await api<{claims:Claim[];csrf_token:string}>('/overview');
      setClaims(data.claims); setToken(data.csrf_token);
    } catch(error) { setNotice((error as Error).message); }
  }
  useEffect(() => {
    void load();
    try {
      const stored = JSON.parse(localStorage.getItem(storageKey) || '{}');
      if (stored && typeof stored === 'object' && !Array.isArray(stored)) {
        setPending(Object.fromEntries(Object.entries(stored).filter((entry):entry is [string,string] => typeof entry[1] === 'string' && /^[a-f0-9]{32}$/.test(entry[1]))));
      }
    } catch { /* A malformed local handle never grants access. */ }
  }, []);

  async function review(claim:Claim) {
    setBusy(true); setNotice('');
    try {
      const value = pending[claim.claim_id]
        ? await api<Review>('/refund-reviews/'+pending[claim.claim_id])
        : await api<Review>('/refunds/'+claim.claim_id+'/review', {method:'POST', headers:{'x-aeronova-token':token}});
      if (value.status === 'pending') {
        savePending({...pending, [claim.claim_id]:value.workflow_id}); setSelected(value);
      } else {
        const remaining = {...pending}; delete remaining[claim.claim_id]; savePending(remaining);
        setNotice(value.result?.message || 'This review is complete.'); await load();
      }
    } catch(error) { setNotice((error as Error).message); }
    finally { setBusy(false); }
  }
  async function decide(approved:boolean) {
    if (!selected) return;
    setBusy(true);
    try {
      const value = await api<Review>('/refund-reviews/'+selected.workflow_id+'/decision', {
        method:'POST', headers:{'Content-Type':'application/json', 'x-aeronova-token':token}, body:JSON.stringify({approved}),
      });
      const remaining = {...pending}; delete remaining[selected.claim_id]; savePending(remaining);
      setSelected(null); setNotice(value.result?.message || 'Review decision saved.'); await load();
    } catch(error) { setNotice((error as Error).message); }
    finally { setBusy(false); }
  }

  return <>
    <div className="page-title"><span className="eyebrow">LET’S GET IT SORTED</span><h1>A clear path <em>to resolution.</em></h1><p>Review the amount. Make the decision. Keep the receipt.</p></div>
    <div className="notice"><ShieldCheck size={19}/>Demo only. Customer identity is server-bound; this is not real authentication. No money is transferred.</div>
    {notice && <div className="notice" role="status">{notice}</div>}
    {!claims.length && <div className="panel empty">There are no claims for the current profile. Enable a bound synthetic demo customer on the API.</div>}
    {claims.map(claim => <section className="panel claim" key={claim.claim_id}>
      <div className="panel-heading"><span className="eyebrow">{claim.claim_id}</span><span className="pill">{pending[claim.claim_id] && claim.can_proceed ? 'Awaiting your decision' : claim.refund_status.replaceAll('_',' ')}</span></div>
      <div className="claim-summary"><div><Receipt size={23}/><h2>Baggage expense reimbursement</h2><p>{claim.reason}</p></div><div><small>APPROVED AMOUNT</small><strong>{claim.approved_amount_aed ? 'AED '+claim.approved_amount_aed : 'Under review'}</strong></div></div>
      <div className="claim-actions"><Button variant="outline" onClick={load}>Check status</Button>
        {claim.refund_status === 'simulated_processed'
          ? <a className="button button-primary" href={'/api/refunds/'+claim.claim_id+'/receipt'}><Download size={16}/>Download receipt</a>
          : <Button disabled={!claim.can_proceed || busy || !token} onClick={()=>void review(claim)}>{pending[claim.claim_id] ? 'Resume review' : 'Review simulation'}<ArrowRight size={16}/></Button>}
      </div>
    </section>)}
    <ConfirmDialog open={!!selected} onOpenChange={open=>{if(!busy&&!open)setSelected(null);}} title="Confirm your refund simulation">
      <div className="notice"><PauseCircle size={18}/>Paused for your decision. You can return to this review later.</div>
      <div className="confirm-total"><span>Approved amount</span><strong>AED {selected?.review?.approved_amount_aed}</strong></div>
      <p className="muted">Original payment method only. The claim is checked again after approval. Duplicate processing is prevented.</p>
      {notice && <p role="alert">{notice}</p>}
      <Button disabled={busy} onClick={()=>void decide(true)}><Check size={16}/>{busy ? 'Saving decision…' : 'Confirm simulation'}</Button>
      <Button variant="outline" disabled={busy} onClick={()=>void decide(false)}><X size={16}/>Reject simulation</Button>
      <Button variant="ghost" disabled={busy} onClick={()=>setSelected(null)}>Not now</Button>
    </ConfirmDialog>
  </>;
}
