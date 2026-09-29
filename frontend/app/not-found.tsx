import Link from 'next/link';

export default function NotFound() {
  return <main className="page reading-page"><div className="reading-head"><p className="eyebrow">Page not found</p><h1>This route is not part of the release.</h1><p>Return to the assessment catalogue to choose one of the five persisted research modules.</p><Link className="button" href="/assessments">View assessments</Link></div></main>;
}
