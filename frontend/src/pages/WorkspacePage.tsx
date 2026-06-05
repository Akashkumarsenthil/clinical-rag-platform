import { useState, useRef, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Document, Page, pdfjs } from 'react-pdf';
import { MessageSquare, Send, X, ChevronDown, ChevronUp, FileText } from 'lucide-react';
import { getDocument, getDocumentFileUrl, chatWithDocument } from '../lib/api';
import type { ChatResponse, SourceDoc } from '../lib/api';
import { StatusBadge } from '../components/StatusBadge';
import { Skeleton } from '../components/Skeleton';
import { cn } from '../lib/utils';

import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceDoc[];
  confidence?: number;
  latency_ms?: number;
}

export function WorkspacePage() {
  const { docId } = useParams<{ docId: string }>();
  const [chatOpen, setChatOpen] = useState(false);
  const [numPages, setNumPages] = useState<number>(0);
  const [metaCollapsed, setMetaCollapsed] = useState(false);

  const { data: doc, isLoading } = useQuery({
    queryKey: ['document', docId],
    queryFn: () => getDocument(docId!),
    enabled: !!docId,
  });

  if (isLoading) {
    return (
      <div className="flex h-full">
        <div className="w-72 border-r border-border-light p-4 space-y-3">
          <Skeleton className="h-6 w-32" />
          {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-4 w-full" />)}
        </div>
        <div className="flex-1 p-4"><Skeleton className="h-full" /></div>
      </div>
    );
  }

  if (!doc) return <div className="p-8 text-text-muted">Document not found</div>;

  const meta = doc.metadata as Record<string, string | number | null> | null;
  const metaEntries = meta
    ? Object.entries(meta).filter(([k, v]) => v != null && k !== 'raw_json')
    : [];

  return (
    <div className="flex h-full relative">
      {/* Left panel: metadata */}
      <aside className={cn('border-r border-border-light flex flex-col flex-shrink-0 transition-all', metaCollapsed ? 'w-10' : 'w-72')}>
        {metaCollapsed ? (
          <button onClick={() => setMetaCollapsed(false)} className="p-2.5 hover:bg-bg-tertiary" aria-label="Expand metadata panel">
            <ChevronDown className="w-4 h-4 text-text-muted rotate-[-90deg]" />
          </button>
        ) : (
          <>
            <div className="px-4 py-3 border-b border-border-light flex items-center justify-between">
              <h2 className="text-[12px] font-semibold text-text-secondary uppercase tracking-wider">Metadata</h2>
              <button onClick={() => setMetaCollapsed(true)} className="p-1 hover:bg-bg-tertiary rounded" aria-label="Collapse metadata panel">
                <ChevronUp className="w-3.5 h-3.5 text-text-muted rotate-[-90deg]" />
              </button>
            </div>
            <div className="flex-1 overflow-auto px-4 py-3">
              <div className="flex items-center gap-2 mb-3">
                <FileText className="w-4 h-4 text-text-muted" />
                <p className="text-[13px] font-medium text-text-primary truncate">{doc.filename}</p>
              </div>
              <StatusBadge status={doc.status} />
              <p className="text-[10px] text-text-muted font-mono mt-2 mb-4">{doc.doc_id}</p>

              {metaEntries.length > 0 ? (
                <dl className="space-y-2.5">
                  {metaEntries.map(([key, value]) => (
                    <div key={key}>
                      <dt className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">{key.replace(/_/g, ' ')}</dt>
                      <dd className="text-[13px] text-text-primary mt-0.5 font-mono tabular-nums">
                        {String(value)}
                      </dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <p className="text-[12px] text-text-muted italic">No metadata extracted</p>
              )}

              <div className="mt-4 pt-3 border-t border-border-light">
                <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">Chunks</p>
                <p className="text-[13px] text-text-primary font-mono tabular-nums">{doc.chunk_count}</p>
              </div>
            </div>
          </>
        )}
      </aside>

      {/* Center: summary + PDF viewer */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Summary bar */}
        {doc.summary && (
          <div className="px-5 py-3 bg-accent-light/50 border-b border-accent-subtle flex-shrink-0">
            <h3 className="text-[11px] font-semibold text-accent uppercase tracking-wider mb-1">Clinical Summary</h3>
            <p className="text-[13px] text-text-primary leading-relaxed">{doc.summary}</p>
          </div>
        )}

        {/* PDF viewer */}
        <div className="flex-1 overflow-auto bg-bg-tertiary p-4 flex justify-center">
          <div className="max-w-3xl w-full">
            <Document
              file={getDocumentFileUrl(doc.doc_id)}
              onLoadSuccess={({ numPages: n }) => setNumPages(n)}
              loading={<Skeleton className="h-[800px] w-full" />}
              error={<div className="text-center py-12 text-text-muted">Failed to load PDF</div>}
            >
              {Array.from({ length: numPages }, (_, i) => (
                <Page
                  key={i}
                  pageNumber={i + 1}
                  width={700}
                  className="mb-4 shadow-sm rounded-sm overflow-hidden"
                  loading={<Skeleton className="h-[900px] w-full mb-4" />}
                />
              ))}
            </Document>
          </div>
        </div>
      </div>

      {/* Chat widget */}
      <ChatWidget docId={doc.doc_id} open={chatOpen} onToggle={() => setChatOpen(!chatOpen)} />
    </div>
  );
}

/* ── Chat widget ──────────────────────────────────────────────────── */

function ChatWidget({ docId, open, onToggle }: { docId: string; open: boolean; onToggle: () => void }) {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  const chat = useMutation({
    mutationFn: (question: string) => chatWithDocument(docId, question),
    onSuccess: (data: ChatResponse) => {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        confidence: data.confidence,
        latency_ms: data.latency_ms,
      }]);
    },
    onError: (err: Error) => {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}` }]);
    },
  });

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  const send = () => {
    const q = input.trim();
    if (!q || chat.isPending) return;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: q }]);
    chat.mutate(q);
  };

  if (!open) {
    return (
      <button
        onClick={onToggle}
        className="absolute bottom-5 right-5 w-12 h-12 bg-accent hover:bg-accent-hover text-white rounded-full shadow-lg flex items-center justify-center transition-colors"
        aria-label="Open document chat"
      >
        <MessageSquare className="w-5 h-5" />
      </button>
    );
  }

  return (
    <div className="absolute bottom-5 right-5 w-96 h-[500px] bg-bg-secondary border border-border rounded-xl shadow-2xl flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border-light flex items-center justify-between flex-shrink-0">
        <div>
          <h3 className="text-[13px] font-semibold text-text-primary">Document Chat</h3>
          <p className="text-[10px] text-text-muted font-mono">Scoped to {docId.slice(0, 12)}...</p>
        </div>
        <button onClick={onToggle} className="p-1 rounded hover:bg-bg-tertiary" aria-label="Close chat">
          <X className="w-4 h-4 text-text-muted" />
        </button>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-auto px-4 py-3 space-y-3">
        {messages.length === 0 && (
          <div className="text-center py-8 text-text-muted">
            <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-20" />
            <p className="text-[12px]">Ask a question about this document</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={cn('flex', msg.role === 'user' ? 'justify-end' : 'justify-start')}>
            <div className={cn(
              'max-w-[85%] px-3 py-2 rounded-lg text-[13px] leading-relaxed',
              msg.role === 'user'
                ? 'bg-accent text-white rounded-br-sm'
                : 'bg-bg-tertiary text-text-primary rounded-bl-sm',
            )}>
              {msg.content}
              {msg.sources && msg.sources.length > 0 && (
                <SourceList sources={msg.sources} />
              )}
              {msg.confidence != null && (
                <div className="mt-1.5 text-[10px] opacity-70">
                  Confidence: {(msg.confidence * 100).toFixed(0)}% | {((msg.latency_ms ?? 0) / 1000).toFixed(1)}s
                </div>
              )}
            </div>
          </div>
        ))}
        {chat.isPending && (
          <div className="flex justify-start">
            <div className="bg-bg-tertiary px-3 py-2 rounded-lg rounded-bl-sm">
              <div className="flex gap-1">
                <span className="w-1.5 h-1.5 bg-text-muted rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-1.5 h-1.5 bg-text-muted rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-1.5 h-1.5 bg-text-muted rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="px-3 py-2.5 border-t border-border-light flex gap-2 flex-shrink-0">
        <input
          type="text"
          placeholder="Ask about this document..."
          className="flex-1 px-3 py-2 bg-bg-primary border border-border rounded-lg text-[13px] focus:outline-none focus:border-accent"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
          disabled={chat.isPending}
        />
        <button
          onClick={send}
          disabled={!input.trim() || chat.isPending}
          className="px-3 py-2 bg-accent hover:bg-accent-hover disabled:bg-border text-white rounded-lg transition-colors"
          aria-label="Send message"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

/* ── Source citations ─────────────────────────────────────────────── */

function SourceList({ sources }: { sources: SourceDoc[] }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="text-[10px] font-medium underline opacity-70 hover:opacity-100"
      >
        {expanded ? 'Hide' : 'Show'} {sources.length} source{sources.length !== 1 ? 's' : ''}
      </button>
      {expanded && (
        <div className="mt-1.5 space-y-1.5">
          {sources.map((s, i) => (
            <div key={i} className="text-[10px] bg-white/10 rounded px-2 py-1.5 leading-snug">
              <span className="font-semibold">
                [{i + 1}] p.{String(s.metadata.page_number ?? '?')}
              </span>
              <span className="ml-1 opacity-70">
                (score: {s.score.toFixed(2)})
              </span>
              <p className="mt-0.5 opacity-80">{s.content.slice(0, 150)}...</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
