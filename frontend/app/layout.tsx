import './globals.css';
import Link from 'next/link';
import type {Metadata} from 'next';
import ReleaseFooter from './release-footer';

export const metadata: Metadata = {
  title: 'Medical AI Suite - Educational Model Explorer',
  description: 'Explore educational, research-only disease model outputs and their limitations.',
};

export default function Layout({children}:{children:React.ReactNode}) {
  return <html lang="en"><body>
    {/* THESIS: An evidence index makes persisted research models inspectable without dashboard noise. OWN-WORLD: warm off-white, white surfaces, ink navy, restrained teal, amber only for Experimental. STORY: reviewers choose a module, inspect inputs, then interpret a fixed threshold result with its limitations. FIRST VIEWPORT: concise introduction left, five scannable model rows right, research-use notice below. FORM: Evidence index, approved composition B, seed 68f657b0. FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, and DESIGN.md */}
    <header className="site-header">
      <Link href="/" className="brand"><span className="brand-mark" aria-hidden="true">M</span>Medical AI Suite</Link>
      <nav aria-label="Primary navigation"><Link href="/assessments" className="nav-link">Assessments</Link><Link href="/methodology" className="nav-link">Methodology</Link><Link href="/limitations" className="nav-link">Limitations</Link></nav>
    </header>
    {children}
    <ReleaseFooter />
  </body></html>;
}
