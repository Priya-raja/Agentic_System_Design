'use client';

import Link from 'next/link';
import {usePathname} from 'next/navigation';
import {Plane, MessageCircle, Ticket, Receipt, BarChart3, Workflow, ArrowUpRight} from 'lucide-react';

const pages = [
  ['/', 'Assistant', MessageCircle],
  ['/bookings', 'Bookings & flights', Ticket],
  ['/refunds', 'Refunds & claims', Receipt],
  ['/evals', 'Evaluations', BarChart3],
  ['/architecture', 'Architecture', Workflow],
] as const;

export function Shell({children}: {children: React.ReactNode}) {
  const path = usePathname();
  const assistant = path === '/' || path === '/chat';
  return <div className={`shell compact-shell ${assistant ? 'assistant-shell' : ''}`}>
    <aside className="sidebar">
      <Link className="brand" href="/" aria-label="AeroNova home" title="AeroNova home">
        <span><Plane size={21}/></span>
      </Link>
      <nav aria-label="Main navigation">
        {pages.map(([href, name, Icon]) => <Link
          className={path === href || (href === '/' && assistant) ? 'active' : ''}
          aria-current={path === href || (href === '/' && assistant) ? 'page' : undefined}
          aria-label={name} title={name} key={href} href={href}>
          <Icon size={19}/><span className="nav-tooltip" aria-hidden="true">{name}</span>
        </Link>)}
      </nav>
      <span className="rail-caption" aria-label="AeroNova demo, fictional policies">DEMO</span>
    </aside>
    <div className="workspace">
      <header>
        <div className="workspace-brand"><Link href="/" className="aeronova-wordmark" aria-label="AeroNova home">Aero<span>Nova</span></Link>
          <span className="breadcrumb-separator">/</span>
          <span>{pages.find(([href]) => href === path)?.[1] || 'Assistant'}</span>
        </div>
        <span className="security-label">AERONOVA DEMO · FICTIONAL POLICIES</span>
      </header>
      <main>{children}</main>
      <footer>
        <p className="brand-pitch"><strong>Every answer. Accounted for.</strong>
          <span>A bounded agent that cites its sources. <Link href="/architecture">See how it works <ArrowUpRight size={12}/></Link></span>
        </p>
        <span>Fictional airline. Demo policies, bookings and payments.</span>
      </footer>
    </div>
  </div>;
}
