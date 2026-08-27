'use client';
import {useState} from 'react';
import {motion,useReducedMotion} from 'motion/react';
import {predictDisease,generateDiseaseReport} from '../../../lib/api';

// Mirrors src/preprocessing/heart_schema.py. Categorical values are sent to the
// API as the exact string tokens the trained one-hot encoder expects; the raw
// coded numbers (cp=3, thal=7 ...) are never shown to the user.
type Num='age'|'trestbps'|'chol'|'thalach'|'oldpeak';
type Cat='sex'|'cp'|'fbs'|'restecg'|'exang'|'slope'|'ca'|'thal';
type V=Record<Num,number|''>&Record<Cat,string|null>;
const init:V={age:'',trestbps:'',chol:'',thalach:'',oldpeak:'',sex:'',cp:'',fbs:'',restecg:'',exang:'',slope:'',ca:'',thal:''};

const NUM:Record<Num,{label:string;unit:string;min:number;max:number;int?:boolean;hint?:string}>={
  age:{label:'Age',unit:'years',min:18,max:100,int:true},
  trestbps:{label:'Resting blood pressure',unit:'mm Hg',min:80,max:220,int:true,hint:'On admission'},
  chol:{label:'Serum cholesterol',unit:'mg/dL',min:100,max:600,int:true},
  thalach:{label:'Maximum heart rate achieved',unit:'bpm',min:60,max:220,int:true,hint:'During exercise test'},
  oldpeak:{label:'ST depression (oldpeak)',unit:'mm',min:0,max:8,hint:'Exercise relative to rest'},
};
const CAT:Record<Cat,{label:string;options:[string,string][];optional?:boolean;hint?:string}>={
  sex:{label:'Sex',options:[['0','Female'],['1','Male']]},
  cp:{label:'Chest pain type',options:[['1','Typical angina'],['2','Atypical angina'],['3','Non-anginal pain'],['4','Asymptomatic']]},
  fbs:{label:'Fasting blood sugar',options:[['0','120 mg/dL or below'],['1','Above 120 mg/dL']]},
  restecg:{label:'Resting ECG result',options:[['0','Normal'],['1','ST-T wave abnormality'],['2','Left ventricular hypertrophy']]},
  exang:{label:'Exercise-induced angina',options:[['0','No'],['1','Yes']]},
  slope:{label:'Slope of peak exercise ST segment',options:[['1','Upsloping'],['2','Flat'],['3','Downsloping']]},
  ca:{label:'Major vessels coloured by fluoroscopy',options:[['0','0 vessels'],['1','1 vessel'],['2','2 vessels'],['3','3 vessels']],optional:true},
  thal:{label:'Thallium stress test',options:[['3','Normal'],['6','Fixed defect'],['7','Reversible defect']],optional:true},
};
const groups=[
  ['Patient information',['age','sex']],
  ['Chest pain and blood pressure',['cp','trestbps']],
  ['Blood work',['chol','fbs']],
  ['ECG and heart rate',['restecg','thalach']],
  ['Exercise stress test',['exang','oldpeak','slope']],
  ['Fluoroscopy and thallium scan',['ca','thal']],
] as const;

const labelOf=(k:string)=> (NUM as any)[k]?.label ?? (CAT as any)[k]?.label ?? k;
const pct=(x:number)=>Math.round(x*100);

export default function HeartAssessment(){
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
      const cat=(CAT as any)[k];
      if(cat){
        const x=(v as any)[k];
        if(x===null) continue;                       // marked unavailable
        if(!x) return `${cat.label} is required.`;
        if(!cat.options.some(([val]:[string,string])=>val===x)) return `Select a valid ${cat.label.toLowerCase()}.`;
        continue;
      }
      const d=(NUM as any)[k];
      const x=(v as any)[k];
      if(x===''||x===undefined) return `${d.label} is required.`;
      if(typeof x==='number'&&(x<d.min||x>d.max)) return `${d.label} must be between ${d.min} and ${d.max} ${d.unit}.`;
    }
    return '';
  };
  const payload=()=>Object.fromEntries(Object.entries(v).map(([k,x])=>[k,x===''?null:x]));

  async function submit(){
    setStage('processing');
    try{const r=await predictDisease('heart',payload());setResult(r);setStage('results');}
    catch(e:any){
      let msg='We could not run this assessment. Check that the API is running.';
      try{const d=JSON.parse(e?.message);const m=d?.error?.message||d?.detail;if(m)msg=String(m);}catch{}
      setError(msg);setStage('review');
    }
  }
  async function pdf(){
    const r=await generateDiseaseReport('heart',payload());
    if(r.ok){const a=document.createElement('a');a.href=URL.createObjectURL(await r.blob());a.download='heart-assessment-report.pdf';a.click();}
  }

  if(stage==='intro')return <main className="page"><p className="eyebrow">HEART ASSESSMENT</p><h1>Explore a heart disease risk pattern</h1><div className="card">
    <p>This educational module uses the UCI <strong>Cleveland Heart Disease</strong> database (303 patient records) and a persisted scikit-learn <strong>logistic regression</strong> pipeline chosen by cross-validation on a held-out development split. It estimates a risk <em>pattern</em> and is not a diagnosis.</p>
    <button className="button" onClick={()=>setStage('form')}>Begin entering measurements</button>
  </div></main>;

  if(stage==='processing')return <main className="page"><motion.div className="card" initial={reduced?false:{opacity:0,y:10}} animate={{opacity:1,y:0}}>
    <h1>Preparing your result</h1><p>Validating measurements</p><p>Applying the fitted preprocessing pipeline</p><p>Running the heart model</p>
  </motion.div></main>;

  if(stage==='results'){
    const m=result.modelMetrics||{};
    return <main className="page"><motion.div initial={reduced?false:{opacity:0,y:10}} animate={{opacity:1,y:0}}>
      <p className="eyebrow">RESULT</p><h1>{result.label}</h1>
      <div className="card">
        <h2>{pct(result.probability||0)}% estimated probability of a heart-disease risk pattern</h2>
        <p>Decision threshold: {result.threshold}. At or above this the model reports an elevated-risk pattern.</p>
        <p>Model: <strong>{result.selectedModel}</strong>
          {m.roc_auc!=null&&<> · holdout ROC-AUC {m.roc_auc.toFixed(2)} · recall {m.recall.toFixed(2)} · specificity {m.specificity.toFixed(2)}</>}
        </p>
        {result.topFactors?.length>0&&<>
          <h3>Factors this model weighs most (permutation importance on held-out data)</h3>
          <ul>{result.topFactors.map((f:any)=><li key={f.feature}>{labelOf(f.feature)}</li>)}</ul>
          <p><small>These describe model behaviour on the Cleveland sample, not a cause of disease.</small></p>
        </>}
        <p>{result.disclaimer}</p>
        <button className="button" onClick={pdf}>Download PDF report</button>
      </div>
    </motion.div></main>;
  }

  if(stage==='review')return <main className="page"><h1>Review heart measurements</h1><div className="card">
    {Object.entries(v).map(([k,x])=>{
      const cat=(CAT as any)[k];const num=(NUM as any)[k];
      let shown:string;
      if(x===null||x==='') shown='Not provided — model imputation will be used';
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
      const cat=(CAT as any)[k];const num=(NUM as any)[k];
      const isNull=(v as any)[k]===null;
      return <label key={k} style={{display:'block',margin:'18px 0'}}>
        {labelOf(k)}{num&&<span> ({num.unit}, {num.min}–{num.max})</span>}{num?.hint&&<span> — {num.hint}</span>}{cat?.hint&&<span> — {cat.hint}</span>}
        {cat
          ? <>
              <select aria-label={cat.label} disabled={isNull}
                value={isNull?'':((v as any)[k]||'')}
                onChange={e=>set(k,e.target.value)}>
                <option value="">Select…</option>
                {cat.options.map(([val,text]:[string,string])=><option key={val} value={val}>{text}</option>)}
              </select>
              {cat.optional&&<span> <input type="checkbox" checked={isNull} onChange={e=>set(k,e.target.checked?null:'')}/> Not available</span>}
            </>
          : <input aria-label={num.label} type="number"
              inputMode={num.int?'numeric':'decimal'} step={num.int?1:0.1} min={num.min} max={num.max}
              value={(v as any)[k]}
              onChange={e=>set(k,e.target.value===''?'':Number(e.target.value))}/>}
      </label>;
    })}
    <button className="button" onClick={()=>{setError('');step?setStep(step-1):null;}}>Back</button>
    {step<groups.length-1
      ? <button className="button" onClick={()=>{const e=valid();if(e){setError(e);return}setError('');setStep(step+1);}}>Continue</button>
      : <button className="button" onClick={()=>{const e=valid();if(e){setError(e);return}setError('');setStage('review');}}>Review</button>}
    {error&&<p role="alert">{error}</p>}
  </div></main>;
}
