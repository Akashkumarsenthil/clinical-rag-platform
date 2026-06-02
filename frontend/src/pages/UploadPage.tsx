import { useCallback, useRef, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Upload, FileUp, ChevronRight, Database, X } from 'lucide-react';
import { listDocuments, uploadDocument, getDocumentStatus, getDocumentVectors } from '../lib/api';
import type { DocumentListItem } from '../lib/api';
import { StatusBadge } from '../components/StatusBadge';
import { ProgressBar } from '../components/ProgressBar';
import { TableSkeleton } from '../components/Skeleton';
import { cn, formatDate } from '../lib/utils';

export function UploadPage() {
  const qc = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [inspecting, setInspecting] = useState<string | null>(null);

  const { data: docs, isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: listDocuments,
    refetchInterval: 4000,
  });

  const upload = useMutation({
    mutationFn: uploadDocument,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['documents'] }),
  });

  const handleFiles = useCallback((files: FileList | null) => {
    if (!files) return;
    Array.from(files).forEach(f => {
      if (f.type === 'application/pdf') upload.mutate(f);
    });
  }, [upload]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const processing = docs?.filter(d => !['COMPLETED', 'FAILED'].includes(d.status)) ?? [];

  return (
    <div className="flex h-full">
      {/* Left column: uploader + doc list */}
      <div className="w-[420px] border-r border-border-light flex flex-col flex-shrink-0">
        {/* Drop zone */}
        <div
          className={cn(
            'mx-4 mt-4 mb-3 border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors',
            dragOver
              ? 'border-accent bg-accent-light'
              : 'border-border hover:border-accent/50',
          )}
          onClick={() => fileRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        >
          <FileUp className="w-8 h-8 mx-auto text-text-muted mb-2" />
          <p className="text-sm font-medium text-text-secondary">
            Drop PDFs here or click to browse
          </p>
          <p className="text-[11px] text-text-muted mt-1">
            Medical documents are processed in 3 stages
          </p>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf"
            multiple
            className="hidden"
            onChange={e => handleFiles(e.target.files)}
          />
        </div>

        {upload.isError && (
          <div className="mx-4 mb-3 px-3 py-2 bg-status-failed-bg text-status-failed text-[12px] rounded-md">
            Upload failed: {(upload.error as Error).message}
          </div>
        )}

        {/* Progress cards for active uploads */}
        {processing.map(doc => (
          <ProgressCard key={doc.doc_id} doc={doc} />
        ))}

        {/* Document list */}
        <div className="flex-1 overflow-auto px-4 pb-4">
          <h3 className="text-[11px] font-semibold text-text-muted uppercase tracking-wider mb-2">
            Documents ({docs?.length ?? 0})
          </h3>

          {isLoading ? <TableSkeleton rows={4} /> : (
            <div className="space-y-1">
              {docs?.map(doc => (
                <div
                  key={doc.doc_id}
                  className="flex items-center gap-3 px-3 py-2.5 rounded-md hover:bg-bg-tertiary group transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <p className="text-[13px] font-medium text-text-primary truncate">
                      {doc.filename}
                    </p>
                    <p className="text-[11px] text-text-muted font-mono mt-0.5">
                      {doc.doc_id.slice(0, 8)}... | {doc.chunk_count} chunks | {formatDate(doc.uploaded_at)}
                    </p>
                  </div>
                  <StatusBadge status={doc.status} />
                  {doc.status === 'COMPLETED' && (
                    <button
                      onClick={() => setInspecting(inspecting === doc.doc_id ? null : doc.doc_id)}
                      className="p-1 rounded hover:bg-accent-light text-text-muted hover:text-accent transition-colors"
                      title="Inspect vectors"
                      aria-label={`Inspect vectors for ${doc.filename}`}
                    >
                      <Database className="w-3.5 h-3.5" />
                    </button>
                  )}
                  <ChevronRight className="w-3.5 h-3.5 text-text-muted opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              ))}

              {docs?.length === 0 && (
                <div className="text-center py-12 text-text-muted">
                  <Upload className="w-10 h-10 mx-auto mb-3 opacity-30" />
                  <p className="text-sm">No documents yet</p>
                  <p className="text-[11px] mt-1">Upload a PDF to get started</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Right panel: vector inspector */}
      <div className="flex-1 overflow-auto">
        {inspecting ? (
          <VectorInspector docId={inspecting} onClose={() => setInspecting(null)} />
        ) : (
          <div className="flex items-center justify-center h-full text-text-muted">
            <div className="text-center">
              <Database className="w-12 h-12 mx-auto mb-3 opacity-20" />
              <p className="text-sm">Select a document to inspect vectors</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Progress card (polls /status while active) ───────────────────── */

function ProgressCard({ doc }: { doc: DocumentListItem }) {
  const { data: progress } = useQuery({
    queryKey: ['doc-progress', doc.doc_id],
    queryFn: () => getDocumentStatus(doc.doc_id),
    refetchInterval: 1500,
    enabled: !['COMPLETED', 'FAILED'].includes(doc.status),
  });

  if (!progress) return null;

  return (
    <div className="mx-4 mb-2 px-3 py-2.5 bg-bg-secondary border border-border-light rounded-lg">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[12px] font-medium text-text-primary truncate max-w-[200px]">
          {doc.filename}
        </span>
        <StatusBadge status={doc.status} />
      </div>
      <ProgressBar progress={progress} />
    </div>
  );
}

/* ── Vector inspector drawer ──────────────────────────────────────── */

function VectorInspector({ docId, onClose }: { docId: string; onClose: () => void }) {
  const { data: vectors, isLoading } = useQuery({
    queryKey: ['vectors', docId],
    queryFn: () => getDocumentVectors(docId),
  });

  return (
    <div className="p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-sm font-semibold text-text-primary">Vector Store Inspector</h2>
          <p className="text-[11px] text-text-muted font-mono mt-0.5">{docId}</p>
        </div>
        <button onClick={onClose} className="p-1.5 rounded-md hover:bg-bg-tertiary" aria-label="Close inspector">
          <X className="w-4 h-4 text-text-muted" />
        </button>
      </div>

      {isLoading ? <TableSkeleton rows={8} /> : (
        <div className="border border-border-light rounded-lg overflow-hidden">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="bg-bg-tertiary border-b border-border-light">
                <th className="text-left px-3 py-2 font-semibold text-text-secondary">Point ID</th>
                <th className="text-left px-3 py-2 font-semibold text-text-secondary">Chunk</th>
                <th className="text-left px-3 py-2 font-semibold text-text-secondary">Dim</th>
                <th className="text-left px-3 py-2 font-semibold text-text-secondary">Vector Preview</th>
              </tr>
            </thead>
            <tbody>
              {vectors?.map((v, i) => (
                <tr key={v.point_id} className={cn('border-b border-border-light last:border-0', i % 2 === 0 ? 'bg-bg-secondary' : 'bg-bg-primary')}>
                  <td className="px-3 py-2 font-mono text-text-muted">{v.point_id.slice(0, 12)}...</td>
                  <td className="px-3 py-2 tabular-nums">{v.chunk_index}</td>
                  <td className="px-3 py-2 tabular-nums">{v.vector_dim}</td>
                  <td className="px-3 py-2 font-mono text-[10px] text-text-muted">
                    [{v.vector_preview.map(n => n.toFixed(4)).join(', ')}...]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {vectors?.length === 0 && (
            <div className="p-8 text-center text-text-muted text-sm">
              No vectors found for this document
            </div>
          )}
        </div>
      )}

      {vectors && vectors.length > 0 && (
        <p className="mt-3 text-[11px] text-text-muted">
          {vectors.length} points in Qdrant | {vectors[0]?.vector_dim}-dimensional vectors (all-MiniLM-L6-v2)
        </p>
      )}
    </div>
  );
}
