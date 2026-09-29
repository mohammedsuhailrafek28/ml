import Link from 'next/link';

const diseases = [
  ['liver', 'Liver', 'Indian Liver Patient Dataset'],
  ['diabetes', 'Diabetes', 'Pima Indians Diabetes'],
  ['heart', 'Heart', 'UCI Cleveland Heart Disease'],
  ['kidney', 'Kidney', 'UCI Chronic Kidney Disease'],
  ['parkinsons', "Parkinson's", "UCI Parkinson's"],
] as const;

export default function Home() {
  return <main className="page">
    <section className="hero-index" aria-labelledby="home-title">
      <div className="hero-copy">
        <p className="eyebrow">Research model explorer</p>
        <h1 id="home-title">Inspect educational model scores</h1>
        <p>Five persisted models. Transparent thresholds. Research use only.</p>
        <div className="hero-actions"><Link className="button" href="/assessments">Start an assessment <span aria-hidden="true">-&gt;</span></Link><Link className="arrow-link" href="/methodology">View methodology</Link></div>
        <p className="disclaimer"><strong>For education and research only.</strong> Not for diagnosis or clinical screening.</p>
      </div>
      <aside className="research-note" aria-labelledby="research-note-title">
        <h2 id="research-note-title">A note for researchers</h2>
        <div className="note-row"><span className="note-icon" aria-hidden="true">01</span><div><strong>Persisted pipelines</strong><span>Five trained models with fixed preprocessing and evaluation code.</span></div></div>
        <div className="note-row"><span className="note-icon" aria-hidden="true">02</span><div><strong>Fixed decision thresholds</strong><span>Predefined boundaries for consistent, reproducible assessment.</span></div></div>
        <div className="note-row"><span className="note-icon" aria-hidden="true">03</span><div><strong>Explicit limitations</strong><span>Dataset context, assumptions, and evidence stay visible.</span></div></div>
      </aside>
    </section>
    <section className="index-section" aria-labelledby="models-title">
      <div className="section-heading"><h2 id="models-title">Explore the models</h2><p>Five disease areas - open datasets - research only</p></div>
      <div className="model-index">{diseases.map(([slug, name, dataset], index) => <Link className={'model-row' + (slug === 'parkinsons' ? ' model-row--experimental' : '')} href={'/assessments/' + slug} key={slug}>
        <span className="model-number" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span><span><span className="model-name">{name}</span><span className="model-dataset">{dataset}</span></span>
        <span className={'status-badge' + (slug === 'parkinsons' ? ' status-badge--experimental' : '')}>{slug === 'parkinsons' ? 'Experimental' : 'Educational / research'}</span><span className="model-arrow" aria-hidden="true">-&gt;</span>
      </Link>)}</div>
    </section>
    <section className="output-guide" aria-labelledby="guide-title"><h2 id="guide-title" className="sr-only">Understand the output</h2>
      <div className="guide-item"><h3>Model score</h3><p>The estimator&apos;s uncalibrated numeric output for the submitted measurements.</p></div>
      <div className="guide-item"><h3>Decision threshold</h3><p>A fixed boundary converts the score into a threshold class for research use.</p></div>
      <div className="guide-item"><h3>Research limitations</h3><p>Dataset characteristics and small cohorts limit what the result can support.</p></div>
    </section>
  </main>;
}
