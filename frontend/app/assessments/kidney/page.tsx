'use client';
import {useState} from 'react';
import {motion,useReducedMotion} from 'motion/react';
import {predictDisease} from '../../../lib/api';
import PredictionResult from '../../../components/PredictionResult';
import type {PredictionResponse} from '../../../lib/api-types';

// The production CKD model uses 14 of the dataset's 24 fields (a fold-safe
// feature-selection experiment dropped the rest without hurting CV ROC-AUC).
// Mirrors src/preprocessing/kidney_schema.py. Every field is optional: tick
// "Not available" and the fitted pipeline imputes it.
type Num='age'|'bp'|'sg'|'al'|'bgr'|'bu'|'sc'|'hemo'|'pcv'|'rc';
type Cat='htn'|'dm'|'appet'|'ane';
type V=Record<Num,number|null|''>&Record<Cat,string|null|''>;
const NUMS:Num[]=['age','bp','sg','al','bgr','bu','sc','hemo','pcv','rc'];
const CATS:Cat[]=['htn','dm','appet','ane'];
const init:V=Object.fromEntries([...NUMS,...CATS].map(k=>[k,''])) as V;

const NUM:Record<Num,{label:string;unit:string;min:number;max:number;step:number;hint?:string}>={
  age:{label:'Age',unit:'years',min:1,max:110,step:1},
  bp:{label:'Diastolic blood pressure',unit:'mm Hg',min:30,max:200,step:1},
  sg:{label:'Urine specific gravity',unit:'ratio',min:1.0,max:1.05,step:0.005,hint:'typically 1.005–1.025'},
  al:{label:'Urine albumin',unit:'0–5 scale',min:0,max:5,step:1},
  bgr:{label:'Random blood glucose',unit:'mg/dL',min:20,max:600,step:1},
  bu:{label:'Blood urea',unit:'mg/dL',min:1,max:450,step:1},
  sc:{label:'Serum creatinine',unit:'mg/dL',min:0.1,max:90,step:0.1},
  hemo:{label:'Haemoglobin',unit:'g/dL',min:3,max:20,step:0.1},
  pcv:{label:'Packed cell volume',unit:'%',min:9,max:60,step:1},
  rc:{label:'Red blood cell count',unit:'millions/cmm',min:2,max:8,step:0.1},
};
const CAT:Record<Cat,{label:string;options:[string,string][]}>={
  htn:{label:'Hypertension',options:[['yes','Yes'],['no','No']]},
  dm:{label:'Diabetes mellitus',options:[['yes','Yes'],['no','No']]},
  appet:{label:'Appetite',options:[['good','Good'],['poor','Poor']]},
  ane:{label:'Anaemia',options:[['yes','Yes'],['no','No']]},
};
const groups=[
  ['Patient basics',['age','bp']],
  ['Urine tests',['sg','al']],
  ['Blood chemistry',['bgr','bu','sc']],
  ['Blood counts',['hemo','pcv','rc']],
  ['Medical history and findings',['htn','dm','appet','ane']],
] as const;
const labelOf=(k:string)=> (NUM as any)[k]?.label ?? (CAT as any)[k]?.label ?? k;

export default function KidneyAssessment(){
  const [v,setV]=useState<V>(init);
  const [step,setStep]=useState(0);
  const [stage,setStage]=useState('intro');
  const [ack,setAck]=useState(false);
  const [result,setResult]=useState<PredictionResponse|null>(null);
  const [error,setError]=useState('');
  const reduced=useReducedMotion();
  const set=(k:string,x:any)=>setV({...v,[k]:x});

  const valid=()=>{
    for(const k of groups[step][1]){
      const x=(v as any)[k];
      if(x===null) continue;                       // marked not available
      if(x===''||x===undefined) return `${labelOf(k)} is required, or mark it "Not available".`;
      const num=(NUM as any)[k];
      if(num&&typeof x==='number'&&(x<num.min||x>num.max)) return `${num.label} must be between ${num.min} and ${num.max} ${num.unit}.`;
      const cat=(CAT as any)[k];
      if(cat&&!cat.options.some(([val]:[string,string])=>val===x)) return `Select a valid ${cat.label.toLowerCase()}.`;
    }
    return '';
  };
  const payload=()=>Object.fromEntries(Object.entries(v).map(([k,x])=>[k,x===''?null:x]));

  async function submit(){
    setStage('processing');
    try{const r=await predictDisease('kidney',payload());setResult(r);setStage('results');}
    catch(e:any){
      let msg='We could not run this assessment. Check that the API is running.';
      try{const d=JSON.parse(e?.message);const m=d?.error?.message||d?.detail;if(m)msg=String(m);}catch{}
      setError(msg);setStage('review');
    }
  }
  if(stage==='intro')return <main className="page"><p className="eyebrow">KIDNEY ASSESSMENT</p><h1>Explore a kidney-dataset model output</h1><div className="card">
    <p>This educational module uses the UCI <strong>Chronic Kidney Disease</strong> dataset (400 records, 2015). A fold-safe feature-selection experiment reduced the 24 recorded fields to a compact <strong>14-field</strong> set with no loss of cross-validated performance, and a persisted scikit-learn <strong>logistic regression</strong> pipeline was chosen on a held-out development split. Every field below is optional — mark it &quot;Not available&quot; and the pipeline will impute it. It produces an uncalibrated dataset-associated score and is not a diagnosis.</p>
    <p><small>This dataset is close to separable on legitimate clinical markers, so held-out scores are very high. That reflects this small curated dataset, not clinical-grade CKD detection.</small></p>
    <button className="button" onClick={()=>setStage('form')}>Begin entering measurements</button>
  </div></main>;

  if(stage==='processing')return <main className="page"><motion.div className="card" initial={reduced?false:{opacity:0,y:10}} animate={{opacity:1,y:0}}>
    <h1>Preparing your result</h1><p>Validating measurements</p><p>Applying the fitted preprocessing pipeline</p><p>Running the kidney model</p>
  </motion.div></main>;

  if(stage==='results'&&result){
    return <main className="page"><motion.div initial={reduced?false:{opacity:0,y:10}} animate={{opacity:1,y:0}}>
      <p className="eyebrow">RESULT</p><h1>Kidney model threshold result</h1>
      <PredictionResult result={result} disease="kidney" measurements={payload()}/>
    </motion.div></main>;
  }

  if(stage==='review')return <main className="page"><h1>Review kidney measurements</h1><div className="card">
    {Object.entries(v).map(([k,x])=>{
      const cat=(CAT as any)[k];const num=(NUM as any)[k];
      let shown:string;
      if(x===null) shown='Not available — model imputation will be used';
      else if(x==='') shown='Not entered';
      else if(cat) shown=cat.options.find(([val]:[string,string])=>val===x)?.[1]??String(x);
      else shown=`${x}${num?' '+num.unit:''}`;
      return <p key={k}><strong>{labelOf(k)}:</strong> {shown}</p>;
    })}
    <label><input type="checkbox" checked={ack} onChange={e=>setAck(e.target.checked)}/> I understand that this is an educational machine-learning prediction and not a medical diagnosis.</label><br/>
    <button className="button" onClick={()=>setStage('form')}>Edit</button>
    <button className="button" disabled={!ack} onClick={submit}>Run prediction</button>
    {error&&<p role="alert">{error}</p>}
  </div></main>;

  const [title,keys]=groups[step];
  return <main className="page"><p className="eyebrow">STEP {step+1} OF {groups.length}</p><h1>{title}</h1><div className="card">
    {keys.map(k=>{
      const num=(NUM as any)[k];const cat=(CAT as any)[k];
      const isNull=(v as any)[k]===null;
      return <label key={k} style={{display:'block',margin:'18px 0'}}>
        {labelOf(k)}{num&&<span> ({num.unit}, {num.min}–{num.max})</span>}{num?.hint&&<span> — {num.hint}</span>}
        {cat
          ? <select aria-label={cat.label} disabled={isNull} value={isNull?'':((v as any)[k]||'')} onChange={e=>set(k,e.target.value)}>
              <option value="">Select…</option>
              {cat.options.map(([val,text]:[string,string])=><option key={val} value={val}>{text}</option>)}
            </select>
          : <input aria-label={num.label} type="number" inputMode="decimal"
              step={num.step} min={num.min} max={num.max} disabled={isNull}
              value={isNull?'':((v as any)[k])}
              onChange={e=>set(k,e.target.value===''?'':Number(e.target.value))}/>}
        <span> <input type="checkbox" checked={isNull} onChange={e=>set(k,e.target.checked?null:'')}/> Not available</span>
      </label>;
    })}
    <button className="button" onClick={()=>{setError('');step?setStep(step-1):null;}}>Back</button>
    {step<groups.length-1
      ? <button className="button" onClick={()=>{const e=valid();if(e){setError(e);return}setError('');setStep(step+1);}}>Continue</button>
      : <button className="button" onClick={()=>{const e=valid();if(e){setError(e);return}setError('');setStage('review');}}>Review</button>}
    {error&&<p role="alert">{error}</p>}
  </div></main>;
}
