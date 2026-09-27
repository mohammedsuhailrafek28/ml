import assert from 'node:assert/strict';
import test from 'node:test';

import {forwardGet,forwardJson,isKnownDisease,unknownDisease} from '../lib/backend.mjs';

const originalFetch=globalThis.fetch;
const originalEnv={...process.env};

test.afterEach(()=>{
  globalThis.fetch=originalFetch;
  for(const key of Object.keys(process.env))if(!(key in originalEnv))delete process.env[key];
  Object.assign(process.env,originalEnv);
});

function configure(){
  process.env.NODE_ENV='test';
  process.env.BACKEND_INTERNAL_URL='http://internal-api:8000';
  process.env.BACKEND_API_KEY='unit-test-key-that-is-never-a-real-secret';
}

test('disease allowlist rejects unknown slugs',async()=>{
  assert.equal(isKnownDisease('liver'),true);
  assert.equal(isKnownDisease('pancreas'),false);
  const response=unknownDisease(new Request('http://web/api/predictions/pancreas'));
  assert.equal(response.status,404);
  assert.equal(response.headers.get('cache-control'),'no-store');
});

test('JSON boundary rejects invalid content type and oversized bodies',async()=>{
  configure();
  let response=await forwardJson(new Request('http://web/api/predictions/liver',{method:'POST',body:'{}'}),'/api/v1/predictions/liver','liver');
  assert.equal(response.status,415);
  process.env.BFF_MAX_REQUEST_BYTES='1024';
  response=await forwardJson(new Request('http://web/api/predictions/liver',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({measurements:{x:'a'.repeat(2000)}})}),'/api/v1/predictions/liver','liver');
  assert.equal(response.status,413);
  response=await forwardJson(new Request('http://web/api/predictions/liver',{method:'POST',headers:{'Content-Type':'application/json'},body:'not-json'}),'/api/v1/predictions/liver','liver');
  assert.equal(response.status,400);
});

test('production boundary fails closed without a sufficiently long key',async()=>{
  configure();
  process.env.NODE_ENV='production';
  process.env.BACKEND_API_KEY='short';
  let called=false;
  globalThis.fetch=async()=>{called=true;return Response.json({ok:true})};
  const response=await forwardGet(new Request('http://web/api/diseases'),'/api/v1/diseases',true);
  assert.equal(response.status,503);
  assert.equal(called,false);
});

test('service credential is added only to protected server requests',async()=>{
  configure();
  const seen=[];
  globalThis.fetch=async(url,init)=>{seen.push({url,headers:new Headers(init.headers)});return Response.json({ok:true})};
  await forwardGet(new Request('http://web/api/health'),'/api/v1/health',false);
  await forwardGet(new Request('http://web/api/diseases'),'/api/v1/diseases',true);
  assert.equal(seen[0].headers.has('x-api-key'),false);
  assert.equal(seen[1].headers.get('x-api-key'),process.env.BACKEND_API_KEY);
  assert.equal(JSON.stringify(await (await forwardGet(new Request('http://web/api/health'),'/api/v1/health')).json()).includes(process.env.BACKEND_API_KEY),false);
});

test('backend status and safe error body are preserved without configuration leakage',async()=>{
  configure();
  globalThis.fetch=async()=>new Response(JSON.stringify({error:{code:'validation_error'}}),{status:422,headers:{'Content-Type':'application/json','X-Request-ID':'backend-id'}});
  const request=new Request('http://web/api/predictions/liver',{method:'POST',headers:{'Content-Type':'application/json'},body:'{"measurements":{}}'});
  const response=await forwardJson(request,'/api/v1/predictions/liver','liver');
  assert.equal(response.status,422);
  assert.equal(response.headers.get('x-request-id'),'backend-id');
  const body=await response.text();
  assert.equal(body.includes(process.env.BACKEND_INTERNAL_URL),false);
  assert.equal(body.includes(process.env.BACKEND_API_KEY),false);
});

test('backend timeout becomes a no-store 504 response',async()=>{
  configure();
  process.env.BACKEND_TIMEOUT_MS='100';
  globalThis.fetch=(_url,init)=>new Promise((_resolve,reject)=>init.signal.addEventListener('abort',()=>reject(new DOMException('aborted','AbortError')),{once:true}));
  const request=new Request('http://web/api/predictions/liver',{method:'POST',headers:{'Content-Type':'application/json'},body:'{"measurements":{}}'});
  const response=await forwardJson(request,'/api/v1/predictions/liver','liver');
  assert.equal(response.status,504);
  assert.equal(response.headers.get('cache-control'),'no-store');
});

test('PDF response is streamed with safe headers',async()=>{
  configure();
  globalThis.fetch=async()=>new Response(new ReadableStream({start(controller){controller.enqueue(new TextEncoder().encode('%PDF-safe'));controller.close()}}),{status:200,headers:{'Content-Type':'application/pdf'}});
  const request=new Request('http://web/api/reports/liver',{method:'POST',headers:{'Content-Type':'application/json'},body:'{"measurements":{}}'});
  const response=await forwardJson(request,'/api/v1/reports/liver','liver',true);
  assert.equal(response.headers.get('content-type'),'application/pdf');
  assert.equal(response.headers.get('content-disposition'),'attachment; filename="liver_report.pdf"');
  assert.equal(response.headers.get('cache-control'),'no-store');
  assert.equal(await response.text(),'%PDF-safe');
});
