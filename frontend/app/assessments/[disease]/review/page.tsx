import Link from 'next/link';
export function generateStaticParams(){return ['liver','heart','diabetes','kidney','parkinsons'].map(disease=>({disease}))}
export default async function Review({params}:{params:Promise<{disease:string}>}){const p=await params; return <main className="page"><p className="eyebrow">REVIEW</p><h1>{p.disease} measurements</h1><div className="card"><p>Use the Python API for the verified disease-specific schema and validation.</p><Link className="button" href={`/assessments/${p.disease}/results`}>Continue to result preview</Link></div></main>}
