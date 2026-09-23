'use client';
import {useState} from 'react';
import {motion,useReducedMotion} from 'motion/react';
import {predictDisease,generateDiseaseReport} from '../../../lib/api';

// Mirrors src/preprocessing/diabetes_schema.py (DIABETES_RANGES / labels / units).
type K='pregnancies'|'glucose'|'blood_pressure'|'skin_thickness'|'insulin'|'bmi'|'diabetes_pedigree'|'age';
type V=Record<K,number|''>;
const init:V={pregnancies:'',glucose:'',blood_pressure:'',skin_thickness:'',insulin:'',bmi:'',diabetes_pedigree:'',age:''};
// label, unit, [min,max], integer?, zeroMeansMissing?
const F:Record<K,{label:string;unit:string;min:number;max:number;int?:boolean;zeroMissing?:boolean;hint?:string}>={
  pregnancies:{label:'Number of pregnancies',unit:'count',min:0,max:20,int:true},
  glucose:{label:'Plasma glucose (2-hour OGTT)',unit:'mg/dL',min:0,max:400,int:true,zeroMissing:true},
  blood_pressure:{label:'Diastolic blood pressure',unit:'mm Hg',min:0,max:200,int:true,zeroMissing:true},
  skin_thickness:{label:'Triceps skin fold thickness',unit:'mm',min:0,max:110,int:true,zeroMissing:true},
  insulin:{label:'2-hour serum insulin',unit:'mu U/ml',min:0,max:1200,int:true,zeroMissing:true},
  bmi:{label:'Body mass index',unit:'kg/m²',min:0,max:100,zeroMissing:true},
  diabetes_pedigree:{label:'Diabetes pedigree function',unit:'score',min:0,max:3},
  age:{label:'Age',unit:'years',min:18,max:120,int:true},
};
const groups=[
  ['Patient profile',['pregnancies','age']],
  ['Glucose and blood pressure',['glucose','blood_pressure']],
  ['Body and insulin measurements',['skin_thickness','insulin','bmi']],
  ['Family-risk measurement',['diabetes_pedigree']],
] as const;
const labelOf=(k:string)=>F[k as K]?.label ?? k;
const pct=(x:number)=>Math.round(x*100);

export default function DiabetesAssessment(){
  const [v,setV]=useState<V>(init);
  const [step,setStep]=useState(0);
  const [stage,setStage]=useState('intro');
  const [ack,setAck]=useState(false);
  const [result,setResult]=useState<any>();
  const [error,setError]=useState('');
  const reduced=useReducedMotion();
  const set=(k:string,x:any)=>setV({...v,[k]:x});

  const valid=()=>{
    for(const k of groups[step][1]){
      const d=F[k as K];const x=v[k as K];
      if(x===''||x===undefined) return `${d.label} is required.`;
      if(typeof x==='number'&&(x<d.min||x>d.max)) return `${d.label} must be between ${d.min} and ${d.max} ${d.unit}.`;
      if(typeof x==='number'&&d.int&&!Number.isInteger(x)) return `${d.label} must be a whole number.`;
    }
    return '';
  };
  const payload=()=>Object.fromEntries(Object.entries(v).map(([k,x])=>[k,x===''?null:x]));

  async function submit(){
    setStage('processing');
    try{const r=await predictDisease('diabetes',payload());setResult(r);setStage('results');}
    catch(e:any){
      let msg='We could not run this assessment. Check that the API is running.';
      try{const d=JSON.parse(e?.message);const m=d?.error?.message||d?.detail;if(m)msg=String(m);}catch{}
      setError(msg);setStage('review');
    }
  }
  async function pdf(){
    const r=await generateDiseaseReport('diabetes',payload());
    if(r.ok){const a=document.createElement('a');a.href=URL.createObjectURL(await r.blob());a.download='diabetes-assessment-report.pdf';a.click();}
  }

  if(stage==='intro')return <main className="page"><p className="eyebrow">DIABETES ASSESSMENT</p><h1>Explore a diabetes risk pattern</h1><div className="card">
    <p>This educational module uses the <strong>Pima Indians Diabetes Database</strong> (768 records of Pima women aged 21+) and a persisted scikit-learn <strong>logistic regression</strong> pipeline chosen by cross-validation on a held-out development split. Impossible zero readings for glucose, blood pressure, skin fold, insulin and BMI are treated as &quot;not measured&quot; and imputed inside the pipeline. It estimates a risk <em>pattern</em> and is not a diagnosis.</p>
    <button className="button" onClick={()=>setStage('form')}>Begin entering measurements</button>
  </div></main>;

  if(stage==='processing')return <main className="page"><motion.div className="card" initial={reduced?false:{opacity:0,y:10}} animate={{opacity:1,y:0}}>
    <h1>Preparing your result</h1><p>Validating measurements</p><p>Applying the fitted preprocessing pipeline</p><p>Running the diabetes model</p>
  </motion.div></main>;

  if(stage==='results'){
    const m=result.modelMetrics||{};
    return <main className="page"><motion.div initial={reduced?false:{opacity:0,y:10}} animate={{opacity:1,y:0}}>
      <p className="eyebrow">RESULT</p><h1>{result.label}</h1>
      <div className="card">
        <h2>{pct(result.probability||0)}% estimated probability of a diabetes risk pattern</h2>
        <p>Decision threshold: {result.threshold}. At or above this the model reports an elevated-risk pattern.</p>
        <p>Model: <strong>{result.selectedModel}</strong>
          {m.roc_auc!=null&&<> · holdout ROC-AUC {m.roc_auc.toFixed(2)} · recall {m.recall.toFixed(2)} · specificity {m.specificity.toFixed(2)}</>}
        </p>
        {result.topFactors?.length>0&&<>
          <h3>Factors this model weighs most (permutation importance on held-out data)</h3>
          <ul>{result.topFactors.map((f:any)=><li key={f.feature}>{labelOf(f.feature)}</li>)}</ul>
          <p><small>These describe model behaviour on the Pima sample, not a cause of disease.</small></p>
        </>}
        <p>{result.disclaimer}</p>
        <button className="button" onClick={pdf}>Download PDF report</button>
      </div>
    </motion.div></main>;
  }

  if(stage==='review')return <main className="page"><h1>Review diabetes measurements</h1><div className="card">
    {Object.entries(v).map(([k,x])=>{
      const d=F[k as K];
      const shown = x===''?'Not provided' : (d?.zeroMissing&&x===0 ? '0 — treated as "not measured" and imputed' : `${x} ${d?.unit??''}`);
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
      const d=F[k as K];
      return <label key={k} style={{display:'block',margin:'18px 0'}}>
        {d.label} <span>({d.unit}, {d.min}–{d.max})</span>{d.zeroMissing&&<span> — enter 0 if not measured</span>}
        <input aria-label={d.label} type="number"
          inputMode={d.int?'numeric':'decimal'} step={d.int?1:0.001} min={d.min} max={d.max}
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
