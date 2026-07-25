import {PredictionResponse} from './api-types'; const base=process.env.NEXT_PUBLIC_API_BASE_URL||'http://localhost:8000';
async function request<T>(url:string,init?:RequestInit):Promise<T>{const r=await fetch(base+url,{...init,headers:{'Content-Type':'application/json',...(init?.headers||{})}});if(!r.ok)throw new Error(await r.text());return r.json()}
export const predictDisease=(d:string,measurements:Record<string,unknown>)=>request<PredictionResponse>(`/api/v1/predictions/${d}`,{method:'POST',body:JSON.stringify({measurements})});
export const generateDiseaseReport=(d:string,m:Record<string,unknown>)=>fetch(`${base}/api/v1/reports/${d}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({measurements:m})});
