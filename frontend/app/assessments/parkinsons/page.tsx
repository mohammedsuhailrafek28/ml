'use client';
import {useState} from 'react';
import {motion,useReducedMotion} from 'motion/react';
import {predictDisease} from '../../../lib/api';
import PredictionResult, {ReleaseBadge} from '../../../components/PredictionResult';
import type {PredictionResponse} from '../../../lib/api-types';

// The production model uses 15 non-redundant pre-computed voice biomarkers
// (a group-aware feature-redundancy experiment dropped the collinear
// RAP/PPQ/DDP jitter and APQ3/APQ5/DDA shimmer measures). Mirrors
// src/preprocessing/parkinsons_schema.py. These values come from voice-analysis
// software (e.g. Praat) — this module does NOT record or analyse raw audio.
type K=
  |'MDVP:Fo(Hz)'|'MDVP:Fhi(Hz)'|'MDVP:Flo(Hz)'
  |'MDVP:Jitter(%)'|'MDVP:Jitter(Abs)'
  |'MDVP:Shimmer(dB)'|'MDVP:APQ'
  |'NHR'|'HNR'
  |'RPDE'|'DFA'|'spread1'|'spread2'|'D2'|'PPE';
type V=Record<K,number|''>;
const F:Record<K,{label:string;unit:string;min:number;max:number;step:number;hint:string}>={
  'MDVP:Fo(Hz)':{label:'Average vocal fundamental frequency',unit:'Hz',min:50,max:300,step:0.001,hint:'mean pitch of the sustained vowel'},
  'MDVP:Fhi(Hz)':{label:'Maximum vocal fundamental frequency',unit:'Hz',min:60,max:650,step:0.001,hint:'highest pitch reached'},
  'MDVP:Flo(Hz)':{label:'Minimum vocal fundamental frequency',unit:'Hz',min:40,max:300,step:0.001,hint:'lowest pitch reached'},
  'MDVP:Jitter(%)':{label:'Jitter (frequency variation)',unit:'%',min:0,max:0.1,step:0.00001,hint:'cycle-to-cycle pitch instability'},
  'MDVP:Jitter(Abs)':{label:'Absolute jitter',unit:'s',min:0,max:0.001,step:0.000001,hint:'pitch instability in seconds'},
  'MDVP:Shimmer(dB)':{label:'Shimmer (amplitude variation, dB)',unit:'dB',min:0,max:3,step:0.001,hint:'cycle-to-cycle loudness instability'},
  'MDVP:APQ':{label:'Amplitude perturbation quotient (11-point)',unit:'ratio',min:0,max:0.3,step:0.00001,hint:'smoothed loudness instability'},
  'NHR':{label:'Noise-to-harmonics ratio',unit:'ratio',min:0,max:1,step:0.00001,hint:'noise energy relative to harmonic energy'},
  'HNR':{label:'Harmonics-to-noise ratio',unit:'dB',min:0,max:45,step:0.001,hint:'harmonic energy relative to noise'},
  'RPDE':{label:'Recurrence period density entropy',unit:'ratio',min:0,max:1,step:0.000001,hint:'nonlinear measure of pitch-period irregularity'},
  'DFA':{label:'Detrended fluctuation analysis',unit:'ratio',min:0.4,max:1,step:0.000001,hint:'fractal scaling of the signal'},
  'spread1':{label:'Nonlinear fundamental-frequency variation 1',unit:'',min:-10,max:0,step:0.000001,hint:'usually negative'},
  'spread2':{label:'Nonlinear fundamental-frequency variation 2',unit:'',min:0,max:1,step:0.000001,hint:''},
  'D2':{label:'Correlation dimension',unit:'',min:0.5,max:5,step:0.000001,hint:'nonlinear signal complexity'},
  'PPE':{label:'Pitch period entropy',unit:'ratio',min:0,max:1,step:0.000001,hint:'irregularity of pitch on a log scale'},
};
const groups=[
  ['Fundamental frequency',['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)']],
  ['Jitter (frequency variation)',['MDVP:Jitter(%)','MDVP:Jitter(Abs)']],
  ['Shimmer (amplitude variation)',['MDVP:Shimmer(dB)','MDVP:APQ']],
  ['Noise and harmonic measures',['NHR','HNR']],
  ['Nonlinear voice measures',['RPDE','DFA','spread1','spread2','D2','PPE']],
] as const;
const KEYS=groups.flatMap(g=>g[1]) as K[];
const init:V=Object.fromEntries(KEYS.map(k=>[k,''])) as V;
const labelOf=(k:string)=>F[k as K]?.label ?? k;

export default function ParkinsonsAssessment(){
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
      const d=F[k as K];const x=v[k as K];
      if(x===''||x===undefined) return `${d.label} is required.`;
      if(typeof x==='number'&&(x<d.min||x>d.max)) return `${d.label} must be between ${d.min} and ${d.max}${d.unit?' '+d.unit:''}.`;
    }
    return '';
  };
  const payload=()=>Object.fromEntries(Object.entries(v).map(([k,x])=>[k,x===''?null:x]));

  async function submit(){
    setStage('processing');
    try{const r=await predictDisease('parkinsons',payload());setResult(r);setStage('results');}
    catch(e:any){
      let msg='We could not run this assessment. Check that the API is running.';
      try{const d=JSON.parse(e?.message);const m=d?.error?.message||d?.detail;if(m)msg=String(m);}catch{}
      setError(msg);setStage('review');
    }
  }
  if(stage==='intro')return <main className="page"><p className="eyebrow">PARKINSON&apos;S ASSESSMENT</p><ReleaseBadge status="experimental"/><h1>Explore a Parkinson&apos;s-associated voice pattern</h1><div className="card">
    <p>This educational module uses the UCI <strong>Parkinson&apos;s voice dataset</strong> (195 sustained-vowel recordings from 32 people, 2008) and a persisted scikit-learn <strong>logistic regression</strong> pipeline. Because the dataset has ~6 recordings per person, training, tuning and evaluation are all <strong>subject-aware</strong> — the same person never appears in both training and test data.</p>
    <p><strong>This module accepts pre-computed voice biomarkers.</strong> The 15 values below are produced by voice-analysis software (e.g. Praat) from a sustained-vowel recording. Medical AI Suite does <em>not</em> record or process raw audio.</p>
    <p><strong>Experimental evidence:</strong> 195 recordings from 32 subjects, including only 8 controls. The subject-disjoint holdout contains 7 people and produced ROC-AUC 0.586 with specificity 0.0 at the fixed 0.5 threshold. This limited result is not clinical validation.</p>
    <button className="button" onClick={()=>setStage('form')}>Begin entering biomarkers</button>
  </div></main>;

  if(stage==='processing')return <main className="page"><motion.div className="card" initial={reduced?false:{opacity:0,y:10}} animate={{opacity:1,y:0}}>
    <h1>Preparing your result</h1><p>Validating biomarkers</p><p>Applying the fitted preprocessing pipeline</p><p>Running the Parkinson&apos;s model</p>
  </motion.div></main>;

  if(stage==='results'&&result){
    return <main className="page"><motion.div initial={reduced?false:{opacity:0,y:10}} animate={{opacity:1,y:0}}>
      <p className="eyebrow">RESULT</p><h1>Parkinson&apos;s model threshold result</h1>
      <PredictionResult result={result} disease="parkinsons" measurements={payload()}/>
    </motion.div></main>;
  }

  if(stage==='review')return <main className="page"><ReleaseBadge status="experimental"/><h1>Review voice biomarkers</h1><div className="card">
    {Object.entries(v).map(([k,x])=>{
      const d=F[k as K];
      return <p key={k}><strong>{d.label}:</strong> {x===''?'Not entered':`${x}${d.unit?' '+d.unit:''}`}</p>;
    })}
    <label><input type="checkbox" checked={ack} onChange={e=>setAck(e.target.checked)}/> I understand this is an experimental educational demonstration with weak validation evidence. It accepts pre-computed biomarkers, not raw audio; its 7-subject holdout had ROC-AUC 0.586 and specificity 0.0; and its output is not a diagnosis or screening result.</label><br/>
    <button className="button" onClick={()=>setStage('form')}>Edit</button>
    <button className="button" disabled={!ack} onClick={submit}>Run prediction</button>
    {error&&<p role="alert">{error}</p>}
  </div></main>;

  const [title,keys]=groups[step];
  return <main className="page"><p className="eyebrow">STEP {step+1} OF {groups.length}</p><ReleaseBadge status="experimental"/><h1>{title}</h1><div className="card">
    <p><small>Pre-computed voice biomarkers from voice-analysis software.</small></p>
    {keys.map(k=>{
      const d=F[k as K];
      return <label key={k} style={{display:'block',margin:'18px 0'}}>
        {d.label} <span>({d.unit||'unitless'}, {d.min}–{d.max})</span>{d.hint&&<span> — {d.hint}</span>}
        <br/><small>metric: <code>{k}</code></small>
        <input aria-label={d.label} type="number" inputMode="decimal"
          step={d.step} min={d.min} max={d.max}
          value={v[k as K]}
          onChange={e=>set(k,e.target.value===''?'':Number(e.target.value))}/>
      </label>;
    })}
    <button className="button" onClick={()=>{setError('');step?setStep(step-1):null;}}>Back</button>
    {step<groups.length-1
      ? <button className="button" onClick={()=>{const e=valid();if(e){setError(e);return}setError('');setStep(step+1);}}>Continue</button>
      : <button className="button" onClick={()=>{const e=valid();if(e){setError(e);return}setError('');setStage('review');}}>Review</button>}
    {error&&<p role="alert">{error}</p>}
  </div></main>;
}
