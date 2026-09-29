'use client';
import {useState} from 'react';
import type {PredictionResponse} from '../lib/api-types';
import {generateDiseaseReport} from '../lib/api';

type Props = {
  result: PredictionResponse;
  disease: string;
  measurements: Record<string,unknown>;
};

export function ReleaseBadge({status}:{status:string}) {
  const experimental=status==='experimental';
  return <span className={`status-badge${experimental?' status-badge--experimental':''}`}>
    {experimental?'Experimental':'Educational / research'}
  </span>;
}

export default function PredictionResult({result,disease,measurements}:Props) {
  const [reportLoading,setReportLoading]=useState(false);
  const [reportError,setReportError]=useState('');
  const relation=result.threshold_result==='at_or_above'?'At or above':'Below';
  async function downloadReport(){
    if(reportLoading)return;
    setReportLoading(true);setReportError('');
    let objectUrl:string|undefined;
    try{
      const blob=await generateDiseaseReport(disease,measurements);
      objectUrl=URL.createObjectURL(blob);
      const link=document.createElement('a');
      link.href=objectUrl;link.download=`${disease}_report.pdf`;link.click();
    }catch{
      setReportError('The report could not be generated. Please try again.');
    }finally{
      if(objectUrl){const completedUrl=objectUrl;setTimeout(()=>URL.revokeObjectURL(completedUrl),0)}
      setReportLoading(false);
    }
  }
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
    <button className="button" disabled={reportLoading} onClick={downloadReport}>
      {reportLoading?'Generating report…':'Download PDF report'}
    </button>
    {reportError&&<p role="alert">{reportError}</p>}
  </div>;
}
