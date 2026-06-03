import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { Search, FileText } from 'lucide-react';
import { searchDocuments } from '../lib/api';
import { StatusBadge } from '../components/StatusBadge';
import { TableSkeleton } from '../components/Skeleton';
import { cn } from '../lib/utils';

export function SearchPage() {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [showAdvanced, setShowAdvanced] = useState(false);

  const searchParams = {
    ...(query ? { q: query } : {}),
    ...Object.fromEntries(Object.entries(filters).filter(([, v]) => v)),
  };
  const hasSearch = Object.keys(searchParams).length > 0;

  const { data: results, isLoading, isFetching } = useQuery({
    queryKey: ['search', searchParams],
    queryFn: () => searchDocuments(searchParams),
    enabled: hasSearch,
  });

  const updateFilter = (key: string, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="p-5 max-w-5xl mx-auto">
      {/* Search bar */}
      <div className="mb-5">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            type="text"
            placeholder="Search by patient name, MRN, DOB, document type..."
            className={cn(
              'w-full pl-10 pr-4 py-2.5 bg-bg-secondary border border-border rounded-lg',
              'text-[13px] placeholder:text-text-muted focus:outline-none focus:border-accent',
              'transition-colors',
            )}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && e.currentTarget.blur()}
          />
          {isFetching && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
          )}
        </div>

        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="mt-2 text-[11px] text-accent hover:text-accent-hover font-medium"
        >
          {showAdvanced ? 'Hide' : 'Show'} advanced filters
        </button>

        {showAdvanced && (
          <div className="mt-3 grid grid-cols-4 gap-3">
            {[
              { key: 'mrn', label: 'MRN', placeholder: 'e.g. 123456' },
              { key: 'first_name', label: 'First Name', placeholder: 'e.g. John' },
              { key: 'last_name', label: 'Last Name', placeholder: 'e.g. Smith' },
              { key: 'dob', label: 'Date of Birth', placeholder: 'e.g. Mar 8 2001' },
              { key: 'document_type', label: 'Document Type', placeholder: 'e.g. Discharge Summary' },
              { key: 'age', label: 'Age', placeholder: 'e.g. 45' },
              { key: 'date_from', label: 'Encounter From', placeholder: 'YYYY-MM-DD' },
              { key: 'date_to', label: 'Encounter To', placeholder: 'YYYY-MM-DD' },
            ].map(({ key, label, placeholder }) => (
              <div key={key}>
                <label className="block text-[11px] font-medium text-text-secondary mb-1">{label}</label>
                <input
                  type="text"
                  placeholder={placeholder}
                  className="w-full px-2.5 py-1.5 bg-bg-secondary border border-border rounded text-[12px] focus:outline-none focus:border-accent"
                  value={filters[key] ?? ''}
                  onChange={e => updateFilter(key, e.target.value)}
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Results */}
      {!hasSearch ? (
        <div className="text-center py-20 text-text-muted">
          <Search className="w-12 h-12 mx-auto mb-3 opacity-20" />
          <p className="text-sm">Enter a search term or use filters</p>
          <p className="text-[11px] mt-1">Search across patient names, MRNs, dates, and document types</p>
        </div>
      ) : isLoading ? (
        <TableSkeleton rows={6} />
      ) : results && results.length > 0 ? (
        <div className="border border-border-light rounded-lg overflow-hidden">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="bg-bg-tertiary border-b border-border-light">
                <th className="text-left px-4 py-2.5 font-semibold text-text-secondary text-[12px]">Document</th>
                <th className="text-left px-4 py-2.5 font-semibold text-text-secondary text-[12px]">Patient</th>
                <th className="text-left px-4 py-2.5 font-semibold text-text-secondary text-[12px]">MRN</th>
                <th className="text-left px-4 py-2.5 font-semibold text-text-secondary text-[12px]">DOB</th>
                <th className="text-left px-4 py-2.5 font-semibold text-text-secondary text-[12px]">Type</th>
                <th className="text-left px-4 py-2.5 font-semibold text-text-secondary text-[12px]">Status</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => (
                <tr
                  key={r.doc_id}
                  className={cn(
                    'border-b border-border-light last:border-0 cursor-pointer hover:bg-accent-light/50 transition-colors',
                    i % 2 === 0 ? 'bg-bg-secondary' : 'bg-bg-primary',
                  )}
                  onClick={() => navigate(`/documents/${r.doc_id}`)}
                  tabIndex={0}
                  role="button"
                  aria-label={`Open ${r.filename}`}
                  onKeyDown={e => e.key === 'Enter' && navigate(`/documents/${r.doc_id}`)}
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-text-muted flex-shrink-0" />
                      <div>
                        <p className="font-medium truncate max-w-[200px]">{r.filename}</p>
                        <p className="text-[10px] text-text-muted font-mono">{r.doc_id.slice(0, 12)}...</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-text-secondary">{r.patient_name ?? '—'}</td>
                  <td className="px-4 py-3 font-mono text-text-secondary tabular-nums">{r.mrn ?? '—'}</td>
                  <td className="px-4 py-3 text-text-secondary tabular-nums">{r.dob ?? '—'}</td>
                  <td className="px-4 py-3 text-text-secondary">{r.document_type ?? '—'}</td>
                  <td className="px-4 py-3"><StatusBadge status={r.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-center py-12 text-text-muted">
          <p className="text-sm">No results found</p>
          <p className="text-[11px] mt-1">Try different search terms or filters</p>
        </div>
      )}

      {results && results.length > 0 && (
        <p className="mt-3 text-[11px] text-text-muted">
          {results.length} document{results.length !== 1 ? 's' : ''} found
        </p>
      )}
    </div>
  );
}
