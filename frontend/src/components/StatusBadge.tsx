import { cn } from '../lib/utils';

const STATUS_STYLES: Record<string, string> = {
  QUEUED: 'bg-status-queued-bg text-status-queued border-status-queued/20',
  EXTRACTING: 'bg-status-processing-bg text-status-processing border-status-processing/20',
  SUMMARIZING: 'bg-status-processing-bg text-status-processing border-status-processing/20',
  EMBEDDING: 'bg-status-processing-bg text-status-processing border-status-processing/20',
  COMPLETED: 'bg-status-completed-bg text-status-completed border-status-completed/20',
  FAILED: 'bg-status-failed-bg text-status-failed border-status-failed/20',
};

export function StatusBadge({ status }: { status: string }) {
  const style = STATUS_STYLES[status] ?? STATUS_STYLES.QUEUED;
  return (
    <span className={cn(
      'inline-flex items-center px-2 py-0.5 text-[11px] font-medium rounded border',
      'leading-tight tracking-wide uppercase',
      style,
    )}>
      {status}
    </span>
  );
}
