import type {TraceEvent} from './TraceTimeline';

function progressText(events: TraceEvent[]): string {
  for (let index = events.length - 1; index >= 0; index--) {
    const event = events[index];
    switch (event.event) {
      case 'route': return event.department ? `Routing to ${event.department}…` : 'Finding the right service…';
      case 'plan': return 'Preparing the policy search…';
      case 'tool_start': return event.tool === 'flight_status' ? 'Checking the flight snapshot…' : 'Searching policy documents…';
      case 'tool_end': {
        if (event.ok === false) return 'A lookup needs attention…';
        const finished = new Set(events.filter(item => item.event === 'tool_end').map(item => item.call_id));
        const searching = events.some(item => item.event === 'tool_start' && !finished.has(item.call_id));
        if (searching) return 'Finishing the remaining searches…';
        if (event.chunk_count === 0) return 'Checking whether the policies cover your question…';
        return event.chunk_count ? `Reviewing ${event.chunk_count} retrieved passages…` : 'Reviewing the lookup results…';
      }
      case 'synth_start': return 'Preparing an answer from the evidence…';
      case 'verify':
        if (event.passed === false) return 'A verification check needs review…';
        if (event.stage === 'cache') return 'Verifying the cached answer…';
        if (event.stage === 'evidence') return 'Checking the source evidence…';
        if (event.stage === 'final') return 'Completing final checks…';
        return 'Verifying citations…';
      case 'cache':
        if (event.kind === 'write') continue;
        return event.hit ? 'Checking a saved answer…' : 'Looking for matching evidence…';
      case 'safety': return event.action && event.action !== 'OK' ? 'Reviewing your request…' : 'Checking the details…';
      case 'error': return 'A service is unavailable…';
    }
  }
  return 'Checking the details…';
}

export function AnswerProgress({events}: {events: TraceEvent[]}) {
  return <div className="answer-progress" role="status" aria-atomic="true">
    <p><span className="progress-light" aria-hidden="true"/>{progressText(events)}</p>
    <div className="answer-skeleton" aria-hidden="true"><span/><span/><span/></div>
  </div>;
}
