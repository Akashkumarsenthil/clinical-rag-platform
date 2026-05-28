import type { DocumentProgress } from '../lib/api';

const STAGE_LABELS: Record<string, string> = {
  extracting_metadata: 'Extracting metadata',
  generating_summary: 'Generating summary',
  embedding_chunks: 'Embedding chunks',
  completed: 'Completed',
  failed: 'Failed',
};

export function ProgressBar({ progress }: { progress: DocumentProgress }) {
  const pct = Math.min(Math.max(progress.percent, 0), 100);
  const label = STAGE_LABELS[progress.stage] ?? progress.stage;
  const detail = progress.stage === 'embedding_chunks' && progress.total_chunks > 0
    ? `${progress.chunks_embedded}/${progress.total_chunks} chunks`
    : '';

  return (
    <div className="w-full space-y-1">
      <div className="flex items-center justify-between text-[11px] text-text-secondary">
        <span className="font-medium">{label}</span>
        <span className="tabular-nums">
          {detail ? `${detail} — ` : ''}{pct.toFixed(0)}%
        </span>
      </div>
      <div className="h-1.5 w-full bg-bg-tertiary rounded-full overflow-hidden">
        <div
          className="h-full bg-accent rounded-full transition-all duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
