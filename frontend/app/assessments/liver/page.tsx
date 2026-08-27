'use client';
import {useState} from 'react';
import {motion,useReducedMotion} from 'motion/react';
import {predictDisease,generateDiseaseReport} from '../../../lib/api';

type V={Age:number|'';Gender:string;TB:number|'';DB:number|'';Alkphos:number|'';Sgpt:number|'';Sgot:number|'';TP:number|'';ALB:number|'';'A/G Ratio':number|null|''};
const init:V={Age:'',Gender:'',TB:'',DB:'',Alkphos:'',Sgpt:'',Sgot:'',TP:'',ALB:'','A/G Ratio':''};
const groups=[['Patient information',['Age','Gender']],['Bilirubin measurements',['TB','DB']],['Liver enzymes',['Alkphos','Sgpt','Sgot']],['Protein measurements',['TP','ALB','A/G Ratio']]] as const;
// label, unit, [min,max], integer? — ranges mirror the FastAPI validator (LIVER_RANGES).
const F:Record<string,{label:string;unit:string;min:number;max:number;int?:boolean}>={
  Age:{label:'Age',unit:'years',min:1,max:120,int:true},
  TB:{label:'Total bilirubin',unit:'mg/dL',min:0,max:100},
  DB:{label:'Direct bilirubin',unit:'mg/dL',min:0,max:60},
  Alkphos:{label:'Alkaline phosphatase',unit:'IU/L',min:10,max:3000,int:true},
  Sgpt:{label:'Alanine aminotransferase (ALT/SGPT)',unit:'IU/L',min:1,max:3000,int:true},
  Sgot:{label:'Aspartate aminotransferase (AST/SGOT)',unit:'IU/L',min:1,max:6000,int:true},
  TP:{label:'Total proteins',unit:'g/dL',min:1,max:12},
  ALB:{label:'Albumin',unit:'g/dL',min:0.5,max:7},
  'A/G Ratio':{label:'Albumin / globulin ratio',unit:'ratio',min:0,max:5},
};
const labels:any={Gender:'Gender',...Object.fromEntries(Object.entries(F).map(([k,d])=>[k,d.label]))};
const pct=(x:number)=>Math.round(x*100);

export default function LiverAssessment(){
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
      const x=v[k as keyof V];
      if(k==='A/G Ratio'&&x===null) continue;               // explicitly marked unavailable
      if(x===''||x===undefined) return `${labels[k]} is required.`;
      if(k==='Gender'){ if(x!=='Male'&&x!=='Female') return 'Select a gender.'; continue; }
      const d=F[k as keyof typeof F];
      if(typeof x==='number'&&(x<d.min||x>d.max)) return `${d.label} must be between ${d.min} and ${d.max} ${d.unit}.`;
    }
    return '';
  };
  const payload=()=>Object.fromEntries(Object.entries(v).map(([k,x])=>[k,x===''?null:x]));

  async function submit(){
    setStage('processing');
    try{const r=await predictDisease('liver',payload());setResult(r);setStage('results');}
    catch(e:any){
      let msg='We could not run this assessment. Check the API.';
      try{const d=JSON.parse(e?.message);const m=d?.error?.message||d?.detail;if(m)msg=String(m);}catch{}
      setError(msg);setStage('review');
    }
  }
  async function pdf(){
    const r=await generateDiseaseReport('liver',payload());
    if(r.ok){const a=document.createElement('a');a.href=URL.createObjectURL(await r.blob());a.download='liver-assessment-report.pdf';a.click();}
  }

  if(stage==='intro')return <main className="page"><p className="eyebrow">LIVER ASSESSMENT</p><h1>Explore a liver disease risk pattern</h1><div className="card">
    <p>This educational module uses the UCI <strong>Indian Liver Patient Dataset</strong> (ILPD, 570 records after de-duplication) and a persisted scikit-learn <strong>logistic regression</strong> pipeline chosen by cross-validation. It estimates a risk <em>pattern</em> and is not a diagnosis.</p>
    <button className="button" onClick={()=>setStage('form')}>Begin entering measurements</button>
  </div></main>;

  if(stage==='processing')return <main className="page"><motion.div className="card" initial={reduced?false:{opacity:0,y:10}} animate={{opacity:1,y:0}}>
    <h1>Preparing your result</h1><p>Validating measurements</p><p>Applying the fitted preprocessing pipeline</p><p>Running the liver model</p>
  </motion.div></main>;

  if(stage==='results'){
    const m=result.modelMetrics||{};
    return <main className="page"><motion.div initial={reduced?false:{opacity:0,y:10}} animate={{opacity:1,y:0}}>
      <p className="eyebrow">RESULT</p><h1>{result.label}</h1>
      <div className="card">
        <h2>{pct(result.probability||0)}% estimated probability of a higher-risk pattern</h2>
        <p>Decision threshold: {result.threshold}. Above this the model reports a higher-risk pattern.</p>
        <p>Model: <strong>{result.selectedModel}</strong>
          {m.roc_auc!=null&&<> · holdout ROC-AUC {m.roc_auc.toFixed(2)} · recall {m.recall.toFixed(2)} · specificity {m.specificity.toFixed(2)}</>}
        </p>
        {result.topFactors?.length>0&&<>
          <h3>Factors this model weighs most (permutation importance on held-out data)</h3>
          <ul>{result.topFactors.map((f:any)=><li key={f.feature}>{labels[f.feature]||f.feature}</li>)}</ul>
          <p><small>These describe model behaviour on the ILPD sample, not a cause of disease.</small></p>
        </>}
        <p>{result.disclaimer}</p>
        <button className="button" onClick={pdf}>Download PDF report</button>
      </div>
    </motion.div></main>;
  }

  if(stage==='review')return <main className="page"><h1>Review liver measurements</h1><div className="card">
    {Object.entries(v).map(([k,x])=><p key={k}><strong>{labels[k]}:</strong> {x===null||x===''?'Not provided — model imputation will be used':`${x}${F[k as keyof typeof F]?' '+F[k as keyof typeof F].unit:''}`}</p>)}
    <label><input type="checkbox" checked={ack} onChange={e=>setAck(e.target.checked)}/> I understand that this is an educational machine-learning prediction and not a medical diagnosis.</label><br/>
    <button className="button" onClick={()=>setStage('form')}>Edit</button>
    <button className="button" disabled={!ack} onClick={submit}>Run prediction</button>
    {error&&<p role="alert">{error}</p>}
  </div></main>;

  const [title,keys]=groups[step];
  return <main className="page"><p className="eyebrow">STEP {step+1} OF 4</p><h1>{title}</h1><div className="card">
    {keys.map(k=>{
      const d=F[k as keyof typeof F];
      return <label key={k} style={{display:'block',margin:'18px 0'}}>
        {labels[k]}{d&&<span> ({d.unit}, {d.min}–{d.max})</span>}
        {k==='Gender'
          ? <select aria-label={labels[k]} value={v.Gender} onChange={e=>set('Gender',e.target.value)}><option value="">Select gender</option><option>Male</option><option>Female</option></select>
          : <>
              <input aria-label={labels[k]} type="number"
                inputMode={d?.int?'numeric':'decimal'} step={d?.int?1:0.01} min={d?.min} max={d?.max}
                disabled={k==='A/G Ratio'&&v['A/G Ratio']===null}
                value={k==='A/G Ratio'&&v['A/G Ratio']===null?'':(v as any)[k]}
                onChange={e=>set(k,e.target.value===''?'':Number(e.target.value))}/>
              {k==='A/G Ratio'&&<span> <input type="checkbox" checked={v['A/G Ratio']===null} onChange={e=>set('A/G Ratio',e.target.checked?null:'')}/> Value unavailable</span>}
            </>}
      </label>;
    })}
    <button className="button" onClick={()=>{const e=valid();if(e){setError(e);return}setError('');step?setStep(step-1):null;}}>Back</button>
    {step<groups.length-1
      ? <button className="button" onClick={()=>{const e=valid();if(e){setError(e);return}setError('');setStep(step+1);}}>Continue</button>
      : <button className="button" onClick={()=>{const e=valid();if(e){setError(e);return}setError('');setStage('review');}}>Review</button>}
    {error&&<p role="alert">{error}</p>}
  </div></main>;
}
