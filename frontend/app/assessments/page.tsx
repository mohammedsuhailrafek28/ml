import Link from 'next/link';

const diseases = [
  ['liver', 'Liver', 'Indian Liver Patient Dataset', 'Review the dataset, model, and measurements before starting.'],
  ['diabetes', 'Diabetes', 'Pima Indians Diabetes', 'Review the dataset, model, and measurements before starting.'],
  ['heart', 'Heart', 'UCI Cleveland Heart Disease', 'Review the dataset, model, and measurements before starting.'],
  ['kidney', 'Kidney', 'UCI Chronic Kidney Disease', 'Review the dataset, model, and measurements before starting.'],
  ['parkinsons', "Parkinson's", "UCI Parkinson's", 'Pre-computed voice biomarkers only; limited subject-disjoint evidence.'],
] as const;

export default function Assessments() {
  return <main className="page catalog-page">
    <div className="catalog-intro"><div><p className="eyebrow">Assessment catalogue</p><h1>Choose a disease module</h1></div><p>Each module documents its dataset, persisted pipeline, inputs, threshold semantics, and known limitations before you begin.</p></div>
    <section className="catalog-list" aria-label="Disease modules">{diseases.map(([slug, name, dataset, description], index) => <article className={'model-row' + (slug === 'parkinsons' ? ' model-row--experimental' : '')} key={slug}>
      <span className="model-number" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span><div><h2 className="model-name">{name}</h2><p className="model-dataset">{dataset}</p><p className="model-dataset">{description}</p></div><span className={'status-badge' + (slug === 'parkinsons' ? ' status-badge--experimental' : '')}>{slug === 'parkinsons' ? 'Experimental' : 'Educational / research'}</span><Link className="button" href={'/assessments/' + slug}>Start</Link>
    </article>)}</section>
  </main>;
}
