import type {PredictionResponse} from './api-types';

async function request<T>(url:string,init?:RequestInit):Promise<T>{
  const response=await fetch(url,{...init,cache:'no-store',headers:{'Content-Type':'application/json',...(init?.headers||{})}});
  if(!response.ok)throw new Error(await response.text());
  return response.json();
}

export const predictDisease=(disease:string,measurements:Record<string,unknown>)=>
  request<PredictionResponse>(`/api/predictions/${disease}`,{method:'POST',body:JSON.stringify({measurements})});

export async function generateDiseaseReport(disease:string,measurements:Record<string,unknown>):Promise<Blob>{
  const response=await fetch(`/api/reports/${disease}`,{
    method:'POST',
    cache:'no-store',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({measurements}),
  });
  if(!response.ok)throw new Error(await response.text());
  return response.blob();
}
