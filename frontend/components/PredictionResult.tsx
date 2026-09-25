import type {PredictionResponse} from '../lib/api-types';

type Props = {
  result: PredictionResponse;
  onDownload: () => void;
};

export function ReleaseBadge({status}:{status:string}) {
  const experimental=status==='experimental';
  return <span className={`status-badge${experimental?' status-badge--experimental':''}`}>
    {experimental?'Experimental':'Educational / research'}
  </span>;
}

export default function PredictionResult({result,onDownload}:Props) {
  const relation=result.threshold_result==='at_or_above'?'At or above':'Below';
  return <div className="card result-card">
    <ReleaseBadge status={result.release_status}/>
    <h2>{relation} the model&apos;s decision threshold</h2>
    <dl className="result-facts">
      <div><dt>Model classification</dt><dd>{result.prediction===1?'Threshold class 1':'Threshold class 0'}</dd></div>
      <div><dt>Uncalibrated model score</dt><dd>{result.model_score.toFixed(4)}</dd></div>
      <div><dt>Decision threshold</dt><dd>{result.decision_threshold.toFixed(4)}</dd></div>
      <div><dt>Threshold result</dt><dd>{relation}</dd></div>
      <div><dt>Score type</dt><dd>Uncalibrated model score — not disease probability</dd></div>
      <div><dt>Model identifier</dt><dd><code>{result.model_identifier}</code></dd></div>
    </dl>
    <h3>Intended use</h3>
    <p>{result.intended_use}</p>
    <h3>Known limitations</h3>
    <ul>{result.limitations.map((item,index)=><li key={index}>{item}</li>)}</ul>
    <p className="disclaimer"><strong>Important:</strong> {result.disclaimer}</p>
    <p>If symptoms or test results concern you, discuss them with a qualified healthcare professional.</p>
    <button className="button" onClick={onDownload}>Download PDF report</button>
  </div>;
}
