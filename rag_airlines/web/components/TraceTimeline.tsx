import {Check, X, LoaderCircle, ArrowUpRight, ArrowLeft, History} from 'lucide-react';

export type TraceEvent = {
  event: string; ms?: number; passed?: boolean; ok?: boolean; error?: string;
  stage?: string; kind?: string; hit?: boolean; tool?: string; call_id?: number;
  classifier?: string; scope?: string; department?: string; action?: string; chunk_count?: number;
  model_tier?: string; model?: string;
  depends_on?: number[]; calls?: {tool: string; depends_on: number[]}[];
  message?: string; text?: string; result?: Result;
};
export type Source = {
  document_id: string; source_file?: string; page: number | null; authority: string;
  chunk_id: string; excerpt?: string;
};
export type OriginalExecution = {version: number; recorded_at?: string; latency_ms: number; events: TraceEvent[]};
export type Result = {
  answer: string; kind: string; cache: string; latency_ms: number; answerable: boolean;
  request_id?: string; trace_id?: string; phoenix_url?: string;
  refusal_reason?: string; sources: Source[]; original_execution?: OriginalExecution;
  verification?: {passed: boolean; citation_errors: number; measurement_errors: number; format_errors?: number; safety_errors: number};
};
export const seconds = (ms: number) => `${(ms / 1000).toFixed(1)}s`;

// This is a reading aid, not a semantic entailment claim. Only source text is highlighted.
function Passage({source, answer}: {source: Source; answer: string}) {
  if (!source.excerpt) return <p className="source-unavailable">The excerpt is unavailable for this response. Ask again to retrieve the source.</p>;
  const excerpt = source.excerpt.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, '').trim();
  const terms = new Set(answer.toLowerCase().match(/\b[a-z]{4,}\b/g) || []);
  const sentences = excerpt.match(/[^.!?\n]+(?:[.!?]+|\n|$)/g) || [excerpt];
  let best = '', score = 0;
  for (const sentence of sentences) {
    const overlap = new Set((sentence.toLowerCase().match(/\b[a-z]{4,}\b/g) || []).filter(word => terms.has(word))).size;
    if (overlap > score) {score = overlap; best = sentence;}
  }
  const index = best ? excerpt.indexOf(best) : -1;
  return <><blockquote className="source-passage">{index < 0 ? excerpt : <>
    {excerpt.slice(0, index)}<mark>{best}</mark>{excerpt.slice(index + best.length)}
  </>}</blockquote>{index >= 0 && <p className="source-reading-note">Highlighted by word overlap to help you find the relevant passage.</p>}</>;
}

function TracePhases({events, busy}: {events: TraceEvent[]; busy: boolean}) {
  const lanes = events.filter(event => event.event === 'tool_start');
  const phases = events.filter(event => ['safety', 'route', 'model_route', 'plan'].includes(event.event)
    || (event.event === 'cache' && event.kind !== 'write'));
  const checks = events.filter(event => event.event === 'verify');
  function row(label: string, event: TraceEvent, key: string | number, failed = false) {
    return <div className={`phase-row ${failed ? 'phase-failed' : ''}`} key={key}>
      <span>{label}</span><i className="timing-leader" aria-hidden="true"/><time>{Math.round(event.ms || 0)}</time>
    </div>;
  }
  return <>
      <section className="trace-phase" aria-label="Preparation">
        {phases.map((event, index) => {
          if (event.event === 'plan') {
            const counts = new Map<string, number>();
            event.calls?.forEach(call => counts.set(call.tool, (counts.get(call.tool) || 0) + 1));
            return <div key={index}>{row('Plan', event, 'plan')}<p className="plan-tools">{Array.from(counts).map(([tool, count]) => `${count}× ${tool}`).join(' + ') || 'No tools required'}</p></div>;
          }
          const label = event.event === 'safety' ? `Safety${event.action && event.action !== 'OK' ? ` · ${event.action}` : ''}`
            : event.event === 'cache' ? `${event.kind === 'semantic' ? 'Semantic cache' : 'Cache'} · ${event.hit ? 'HIT' : 'MISS'}`
            : event.event === 'model_route' ? `Model → ${event.model || event.model_tier || 'standard'}`
              : event.event === 'plan' ? 'Plan'
                : `Route → ${event.department || event.scope || event.classifier || 'policy'}`;
          return row(label, event, index, event.action === 'BLOCK' || event.action === 'REVIEW');
        })}
      </section>
      {lanes.length > 0 && <section className="trace-phase" aria-label="Tool execution">
        <h4 className="phase-heading">{lanes.filter(lane => !lane.depends_on?.length).length > 1 ? 'PARALLEL LANES' : lanes.length > 1 ? 'EXECUTION LANES' : 'EXECUTION LANE'}</h4>
        {lanes.map(start => {
          const finish = events.find(event => event.event === 'tool_end' && event.call_id === start.call_id);
          return <div className="execution-lane" key={start.call_id}>
            <div className="lane-label"><span>LANE {(start.call_id || 0) + 1}</span><code>{start.tool}</code></div>
            {!!start.depends_on?.length && <p className="lane-dependencies">After {start.depends_on.map(id => `lane ${id + 1}`).join(', ')}</p>}
            {row('├─ started', start, 'start')}
            {finish ? row(`└─ ${finish.ok === false ? finish.error || 'Failed' : finish.chunk_count !== undefined ? `${finish.chunk_count} chunks retrieved` : 'Completed'}`, finish, 'finish', finish.ok === false)
              : <p className="lane-pending">└─ {busy ? 'Running…' : 'Stopped before completion'}</p>}
          </div>;
        })}
      </section>}
      {checks.length > 0 && <section className="trace-phase" aria-label="Verification">
        <h4 className="phase-heading">VERIFICATION</h4>
        {checks.map((event, index) => <div className={`phase-row verification-row ${event.passed ? 'check-passed' : 'phase-failed'}`} key={index}>
          <span>{event.passed ? <Check size={13}/> : <X size={13}/>}
            {event.stage === 'evidence' ? 'Evidence check' : event.stage === 'cache' ? 'Cached answer check' : event.stage === 'final' ? 'Final check' : 'Answer check'}
            <small>{event.passed ? 'PASS' : 'FAIL'}</small></span><i className="timing-leader" aria-hidden="true"/><time>{Math.round(event.ms || 0)}</time>
        </div>)}
      </section>}
  </>;
}

export function TraceTimeline({events, busy, source, answer, onBack, onClose}: {
  events: TraceEvent[]; busy: boolean; source?: Source; answer: string; onBack: () => void; onClose: () => void;
}) {
  const result = events.find(event => event.event === 'result')?.result;
  const original = result?.original_execution;
  const cacheHit = result && ['exact', 'semantic'].includes(result.cache);
  const retrieved = events.filter(event => event.event === 'tool_end').reduce((count, event) => count + (event.chunk_count || 0), 0);
  const recorded = original?.recorded_at ? new Date(original.recorded_at) : undefined;
  return <aside className="trace-panel" aria-label="Request trace" id="request-trace">
    <div className="trace-title">
      <span className="eyebrow">UNDER THE SURFACE</span>
      <button className="icon-button" onClick={onClose} aria-label="Close trace"><X size={17}/></button>
      <h3>{source ? 'The evidence, in full.' : 'A clear chain of evidence.'}</h3>
      <p>{source ? 'The actual passage supplied to the answer model.' : 'One request. Every decision in view.'}</p>
    </div>
    {source ? <div className="source-inspector">
      <button className="text-button" onClick={onBack}><ArrowLeft size={13}/>Back to request trace</button>
      <div className="source-heading"><strong>{source.document_id}</strong><span>{source.authority || 'Source'}{source.page ? ` · page ${source.page}` : ''}</span></div>
      <Passage source={source} answer={answer}/>
      <dl className="source-details"><dt>Document</dt><dd>{source.source_file || source.document_id}</dd><dt>Chunk</dt><dd>{source.chunk_id}</dd></dl>
      <p className="source-reading-note">Citations identify retrieved evidence. Automated checks validate citation IDs, quantities and safety; they do not prove every claim.</p>
    </div> : <>
      <div className="phase-table-heading"><span>{cacheHit ? 'This request' : 'Phase'}</span><span>elapsed ms</span></div>
      {!events.length && <div className="trace-placeholder"><span className="trace-branch">├─</span><p>Ask a question to follow its path.<small>Safety → intent route → retrieval → verification</small></p></div>}
      <TracePhases events={events} busy={busy}/>
      {events.filter(event => event.event === 'error').map((event, index) => <p role="alert" className="trace-error" key={index}>{event.message || 'Request failed.'}</p>)}
      {busy && <div className="trace-running"><LoaderCircle size={14} className="spin"/>Following your request…</div>}
      {result && <p className="trace-summary">{retrieved > 0 ? `${retrieved} retrieved chunks · ` : ''}{result.answerable ? result.sources.length : 0} cited sources · {seconds(result.latency_ms)} · {result.cache === 'miss' ? 'cache miss' : result.cache === 'bypass' ? 'cache bypassed' : `${result.cache} cache hit`}</p>}
      {cacheHit && <section className="cached-execution" aria-label="Original cached execution">
        <div className="cache-origin-heading"><History size={15}/><div><strong>Retrieved from cache</strong><span>Original execution</span></div></div>
        {original?.events.length ? <>
          <p className="cache-origin-note">Historical trace · {seconds(original.latency_ms)}{recorded && !Number.isNaN(recorded.getTime()) ? ` · ${recorded.toLocaleDateString('en-GB', {day:'numeric',month:'short',year:'numeric'})}` : ''}. Tools shown below did not run on this request.</p>
          <div className="ghost-trace"><div className="phase-table-heading"><span>Original phases</span><span>original elapsed ms</span></div><TracePhases events={original.events.filter(event => !['safety', 'cache', 'synth_start'].includes(event.event))} busy={false}/></div>
        </> : <p className="cache-origin-note">The original execution was not recorded for this older cache entry. No historical steps have been reconstructed.</p>}
      </section>}
      <div className="trace-foot"><span>MAX 4 TOOLS · MAX 2 MODEL CALLS</span>
        <p>Times are elapsed from the start of the labelled request. Independent lanes run concurrently. Retrieved chunks are candidates; only cited sources support an answer.</p>
        {result?.phoenix_url && /^https?:\/\//.test(result.phoenix_url) && <a href={result.phoenix_url} target="_blank" rel="noopener noreferrer">View in Phoenix <ArrowUpRight size={13}/></a>}
        <a href="/architecture">Explore the architecture <ArrowUpRight size={13}/></a>
      </div>
    </>}
  </aside>;
}
