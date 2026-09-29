import './globals.css'; import Link from 'next/link';
import type {Metadata} from 'next';
import ReleaseFooter from './release-footer';

export const metadata: Metadata = {
  title: 'Medical AI Suite — Educational Model Explorer',
  description: 'Explore educational, research-only disease model outputs and their limitations.',
};

export default function Layout({children}:{children:React.ReactNode}){return <html lang="en"><body><header><Link href="/" className="brand">Medical AI Suite</Link><nav><Link href="/assessments">Assessments</Link><Link href="/methodology">Methodology</Link><Link href="/limitations">Limitations</Link></nav></header>{children}<ReleaseFooter /></body></html>}
