import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { Upload, Search, FileText } from 'lucide-react';
import { cn } from '../lib/utils';

const NAV_ITEMS = [
  { to: '/upload', label: 'Upload', icon: Upload },
  { to: '/search', label: 'Search', icon: Search },
] as const;

export function AppShell() {
  const location = useLocation();

  const breadcrumb = (() => {
    const p = location.pathname;
    if (p.startsWith('/documents/')) return ['Documents', p.split('/')[2]?.slice(0, 8) + '...'];
    if (p === '/upload') return ['Upload & Process'];
    if (p === '/search') return ['Search'];
    return ['Home'];
  })();

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 bg-bg-sidebar flex flex-col flex-shrink-0">
        <div className="px-4 py-5 border-b border-white/5">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center">
              <FileText className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-text-sidebar-active leading-tight">
                Clinical RAG
              </h1>
              <p className="text-[10px] text-text-sidebar leading-tight mt-0.5">
                Document Workspace
              </p>
            </div>
          </div>
        </div>

        <nav className="flex-1 px-3 py-3 space-y-0.5">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) => cn(
                'flex items-center gap-2.5 px-3 py-2 rounded-md text-[13px] font-medium transition-colors',
                isActive
                  ? 'bg-bg-sidebar-active text-text-sidebar-active'
                  : 'text-text-sidebar hover:bg-bg-sidebar-hover hover:text-text-sidebar-active',
              )}
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="px-4 py-3 border-t border-white/5 text-[10px] text-text-sidebar">
          v0.2.0 — Hybrid RAG
        </div>
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Breadcrumb */}
        <header className="h-11 px-5 flex items-center border-b border-border-light bg-bg-secondary flex-shrink-0">
          <nav className="flex items-center gap-1.5 text-[12px]">
            {breadcrumb.map((crumb, i) => (
              <span key={i} className="flex items-center gap-1.5">
                {i > 0 && <span className="text-text-muted">/</span>}
                <span className={i === breadcrumb.length - 1 ? 'text-text-primary font-medium' : 'text-text-muted'}>
                  {crumb}
                </span>
              </span>
            ))}
          </nav>
        </header>

        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
