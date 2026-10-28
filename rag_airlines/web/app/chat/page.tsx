'use client';

import {useEffect, useLayoutEffect, useRef, useState} from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import {ArrowRight, ArrowUpRight, Check, Plus, Sparkles, AlertCircle, PanelRight, LockKeyhole} from 'lucide-react';
import {Button} from '@/components/ui/button';
import {TraceTimeline, seconds, type TraceEvent, type Result, type Source} from '@/components/TraceTimeline';
import {api} from '@/lib/utils';
import {VerifiedAnswer} from '@/components/VerifiedAnswer';
import {AnswerProgress} from '@/components/AnswerProgress';

type Message = {question: string; answer: string; result?: Result; error?: string; trace: TraceEvent[]};
const examples = [
  'Baggage allowance on Economy Flex in 2026',
  'Can I upgrade my fare to Business?',
  'What expenses can I claim for delayed baggage?',
  'How does EU Regulation 261/2004 address flight delays?',
  "What's in the onboard menu?",
];

export default function Chat() {
  const [token, setToken] = useState(''), [draft, setDraft] = useState('');
  const [messages, setMessages] = useState<Message[]>([]), [events, setEvents] = useState<TraceEvent[]>([]);
  const [busy, setBusy] = useState(false), [connection, setConnection] = useState('');
  const [traceOpen, setTraceOpen] = useState(true), [mobile, setMobile] = useState(false);
  const [selectedSource, setSelectedSource] = useState<Source>(), [sourceAnswer, setSourceAnswer] = useState('');
  const abort = useRef<AbortController | null>(null), firstAnswer = useRef(false);
  const log = useRef<HTMLDivElement>(null), traceButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    api<{csrf_token: string}>('/overview').then(data => setToken(data.csrf_token))
      .catch(() => setConnection('The AeroNova service is temporarily unavailable. Please try again shortly.'));
    return () => abort.current?.abort();
  }, []);
  useEffect(() => {
    const query = window.matchMedia('(max-width: 1000px)');
    setMobile(query.matches);
    if (query.matches) setTraceOpen(false);
    let wasMobile = query.matches;
    const resize = () => {
      if (query.matches === wasMobile) return;
      wasMobile = query.matches; setMobile(query.matches); setTraceOpen(false);
    };
    query.addEventListener('change', resize);
    return () => query.removeEventListener('change', resize);
  }, []);
  const latestResult = messages[messages.length - 1]?.result;
  const latestError = messages[messages.length - 1]?.error;
  useLayoutEffect(() => {
    const container = log.current;
    const turn = container?.querySelector<HTMLElement>('.conversation:last-child');
    if (container && turn) container.scrollTo({top: Math.max(0, turn.offsetTop - 24), behavior: 'instant'});
  }, [messages.length, latestResult, latestError]);

  function inspect(message: Message, source?: Source) {
    setEvents(message.trace); setSelectedSource(source); setSourceAnswer(message.answer); setTraceOpen(true);
  }
  function reset() {
    setMessages([]); setEvents([]); setDraft(''); setSelectedSource(undefined);
    firstAnswer.current = false; setTraceOpen(!mobile);
  }
  async function send(question = draft) {
    if (!question.trim() || busy) return;
    const previous = messages[messages.length - 1];
    const isEmailAboutRefundFollowup = /^\s*(?:please\s+)?(?:write|draft|compose)\s+(?:(?:me|an?)\s+)*email\s+(?:about|for)\s+it[.!?]*\s*$/i.test(question);
    const requestQuestion = isEmailAboutRefundFollowup && previous?.result?.kind === 'handoff' && /\brefund\b/i.test(previous.question)
      ? 'Write an email to AeroNova about a ticket refund'
      : question;
    setDraft(''); setBusy(true); setEvents([]); setSelectedSource(undefined);
    if (!firstAnswer.current && !mobile) setTraceOpen(true);
    setMessages(previous => [...previous, {question, answer: '', trace: []}]);
    const collected: TraceEvent[] = [];
    abort.current = new AbortController();
    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST', headers: {'Content-Type': 'application/json', 'x-aeronova-token': token},
        body: JSON.stringify({question: requestQuestion}), signal: abort.current.signal,
      });
      if (!response.ok) {
        const failure = await response.json();
        throw new Error(failure.detail || failure.error || 'Assistant unavailable');
      }
      const reader = response.body!.getReader(), decoder = new TextDecoder();
      let buffer = '';
      function consume(frame: string) {
        const data = frame.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
        if (!data) return;
        const event = JSON.parse(data) as TraceEvent;
        collected.push(event); setEvents([...collected]);
        if (event.event === 'result' && event.result?.answerable && !firstAnswer.current) {
          firstAnswer.current = true; setTraceOpen(false);
        }
        setMessages(previous => previous.map((message, index) => index === previous.length - 1 ? {
          ...message, trace: [...collected],
          // The result event is the public release boundary. Ignore token events.
          answer: event.event === 'result' ? event.result!.answer : message.answer,
          result: event.event === 'result' ? event.result : message.result,
        } : message));
      }
      for (;;) {
        const {value, done} = await reader.read();
        buffer += decoder.decode(value, {stream: !done}).replace(/\r\n/g, '\n');
        const frames = buffer.split('\n\n'); buffer = frames.pop() || '';
        for (const frame of frames) consume(frame);
        if (done) {if (buffer.trim()) consume(buffer); break;}
      }
      const result = collected.find(event => event.event === 'result')?.result;
      if (!result) throw new Error('The stream ended before a verified result was received. Please retry.');
    } catch (error) {
      const message = (error as Error).name === 'AbortError' ? 'Request stopped. No unfinished answer was saved.' : (error as Error).message;
      setMessages(previous => previous.map((item, index) => index === previous.length - 1 ? {...item, error: message} : item));
    } finally {setBusy(false);}
  }

  const trace = <Dialog.Content className="trace-drawer" aria-describedby="trace-description"
    onOpenAutoFocus={event => {if (!mobile) event.preventDefault();}}
    onCloseAutoFocus={event => {event.preventDefault(); traceButton.current?.focus();}}
    onInteractOutside={event => {if (!mobile) event.preventDefault();}}>
    <Dialog.Title className="sr-only">Request trace and evidence</Dialog.Title>
    <Dialog.Description className="sr-only" id="trace-description">The request phases, tool execution lanes, verification checks and cited source passages.</Dialog.Description>
    <TraceTimeline events={events} busy={busy} source={selectedSource} answer={sourceAnswer}
      onBack={() => setSelectedSource(undefined)} onClose={() => setTraceOpen(false)}/>
  </Dialog.Content>;

  return <Dialog.Root open={traceOpen} onOpenChange={setTraceOpen} modal={mobile}>
    <div className="page-title chat-page-title"><span className="eyebrow">YOUR TRAVEL COMPANION</span>
      <h1>A little help. <em>A lot more clarity.</em></h1>
      <p>Ask your question. See the evidence. Get on with your journey.</p>
    </div>
    {connection && <div className="notice error" role="alert"><AlertCircle size={16}/>{connection}</div>}
    <div className={`chat-grid ${traceOpen && !mobile ? 'trace-is-open' : ''}`}>
      <section className="chat-column" aria-label="AeroNova demo conversation">
        <div className="chat-top">
          <span className="assistant-icon"><Sparkles size={18}/></span>
          <div><strong>AeroNova assistant</strong><small>Fictional policy demo</small></div>
          <div className="chat-actions">
            <Button variant="ghost" disabled={busy} onClick={reset} aria-label="New conversation" title="New conversation"><Plus size={15}/><span>New conversation</span></Button>
            <button className={`trace-toggle ${busy ? 'is-working' : ''}`} ref={traceButton} aria-expanded={traceOpen}
              aria-controls={traceOpen ? 'request-trace' : undefined} onClick={() => {setSelectedSource(undefined); setTraceOpen(!traceOpen);}}>
              <PanelRight size={15}/><span>{traceOpen ? 'Hide trace' : 'Show trace'}</span>
            </button>
          </div>
        </div>
        <div className="messages" ref={log} aria-live="polite" aria-relevant="additions text">
          {!messages.length && <div className="welcome">
            <span aria-hidden="true">◐</span><h2>Ask AeroNova a question</h2>
            <p>Every answer is cited against a source document.</p>
            <div className="suggestions"><span className="eyebrow">TRY ONE</span>
              {examples.map(question => <button disabled={!token} key={question} onClick={() => void send(question)}>{question}<ArrowUpRight size={15}/></button>)}
            </div>
            <p className="welcome-promise">Grounded <span>·</span> Cited <span>·</span> Verifiable</p>
          </div>}
          {messages.map((message, index) => {
            const result = message.result, verified = result?.verification?.passed && result.answerable;
            const consideredChunks = [...message.trace, ...(result?.original_execution?.events || [])]
              .some(event => event.event === 'tool_end' && (event.chunk_count || 0) > 0);
            const citationGroups = Array.from((result?.sources || []).reduce((groups, source, sourceIndex) => {
              const key = source.document_id || source.source_file || source.chunk_id;
              const group = groups.get(key) || {documentId:key, authority:source.authority, citations:[] as number[], sources:[] as Source[]};
              group.citations.push(sourceIndex + 1); group.sources.push(source); groups.set(key, group); return groups;
            }, new Map<string,{documentId:string;authority:string;citations:number[];sources:Source[]}>()).values());
            const primary = citationGroups.filter(group => group.authority === 'primary').length;
            return <article className="conversation" key={index}>
              <div className="user-turn"><span className="eyebrow user-label">YOU</span><div className="user-message">{message.question}</div></div>
              <div className={`assistant-message ${message.error ? 'failed' : ''}`}>
                <span className="eyebrow">AERONOVA</span>
                {message.error ? <p role="alert">{message.error}</p> : result?.kind === 'handoff' ? <div className="handoff-card"><LockKeyhole size={19}/><div><strong>Contact AeroNova support</strong><p>{result.refusal_reason || message.answer}</p></div></div> : result?.kind === 'clarification' ? <div className="clarification-card"><AlertCircle size={19}/><div><strong>I need a little more detail</strong><p>{result.refusal_reason || message.answer}</p><span>Which would you like to know about?</span><div className="clarification-actions"><button type="button" onClick={() => void send('Lost or damaged baggage claim policy')}>Lost or damaged baggage</button><button type="button" onClick={() => void send('Public ticket refund policy for a cancelled flight in the demo corpus')}>Cancelled flight ticket refund</button></div></div></div> : result?.kind === 'verification_failed' ? <div className="verification-failure" role="alert">
                  <AlertCircle size={18}/><div><strong>Couldn’t verify this answer</strong><p>{message.answer}</p>
                    {result.verification && <p>{result.verification.citation_errors} citation error{result.verification.citation_errors === 1 ? '' : 's'} · {result.verification.measurement_errors} quantity error{result.verification.measurement_errors === 1 ? '' : 's'} · {result.verification.format_errors || 0} format error{result.verification.format_errors === 1 ? '' : 's'} · {result.verification.safety_errors} safety error{result.verification.safety_errors === 1 ? '' : 's'}</p>}
                  </div>
                </div> : result && !result.answerable ? <div className="abstention"><AlertCircle size={16}/><div><p>{result.refusal_reason || message.answer}</p>{consideredChunks && <p className="abstention-note">Retrieved chunks were considered, but did not contain enough information to answer. They are not supporting citations.</p>}</div></div>
                  : result ? <div className="released-answer"><VerifiedAnswer text={message.answer}/>
                    {citationGroups.length > 0 && result?.answerable && <span className="inline-citations">{citationGroups.map((group, groupIndex) => <sup key={group.documentId}>
                      <button aria-label={`View source ${group.citations.join(', ')}: ${group.documentId}`} onClick={() => inspect(message, group.sources[0])}>[{group.citations.join(', ')}]</button>
                    </sup>)}</span>}
                  </div> : <AnswerProgress events={message.trace}/>}
                {result && <div className="answer-footer">
                  <div className={`verification-statement ${verified ? 'verified' : ''}`}>
                    {verified && <Check size={15}/>}
                    <span>{verified ? `Verified · ${primary ? `${primary} primary source${primary === 1 ? '' : 's'}` : `${result.sources.length} source${result.sources.length === 1 ? '' : 's'}`} · ${seconds(result.latency_ms)}` : `${result.answerable ? 'Completed' : 'Answer withheld'} · ${seconds(result.latency_ms)}`}</span>
                  </div>
                  {verified && <p className="verification-scope">Citation & quantity checks passed</p>}
                  <button className="text-button answer-trace-toggle" onClick={() => inspect(message)}><PanelRight size={13}/>Show trace</button>
                </div>}
                {citationGroups.length > 0 && result?.answerable && <div className="source-links" aria-label="Cited sources">
                  {citationGroups.map(group => <details className="citation-group" key={group.documentId}>
                    <summary><span>[{group.citations.join(', ')}]</span>{group.documentId}<small>{group.authority} · {group.sources.length} cited passage{group.sources.length === 1 ? '' : 's'}</small></summary>
                    <div className="source-chunks">{group.sources.map((source, index) => <button key={source.chunk_id} onClick={() => inspect(message, source)}>
                      Passage {index + 1} · {source.chunk_id}{source.page ? ` · p. ${source.page}` : ''}<ArrowUpRight size={12}/>
                    </button>)}</div>
                  </details>)}
                </div>}
              </div>
            </article>;
          })}
        </div>
        <div className="composer-area">
          <form className="composer" onSubmit={event => {event.preventDefault(); void send();}}>
            <label className="sr-only" htmlFor="question">Your question</label>
            <input id="question" maxLength={2000} value={draft} onChange={event => setDraft(event.target.value)} placeholder="Ask about your journey…" disabled={busy}/>
            <button aria-label={busy ? 'Stop request' : 'Send question'} type={busy ? 'button' : 'submit'} disabled={!token || (!busy && !draft.trim())} onClick={busy ? () => abort.current?.abort() : undefined}>
              {busy ? <span className="stop-icon"/> : <ArrowRight size={20}/>}
            </button>
          </form>
          <p className="privacy-note"><LockKeyhole size={11}/>Keep private identifiers out of chat. Use the booking and refund workflows.</p>
        </div>
      </section>
      {!mobile && trace}
    </div>
    {mobile && <Dialog.Portal><Dialog.Overlay className="trace-overlay"/>{trace}</Dialog.Portal>}
  </Dialog.Root>;
}
