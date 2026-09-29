const DISEASES = new Set(['liver', 'diabetes', 'heart', 'kidney', 'parkinsons']);
const DEFAULT_BODY_LIMIT = 16 * 1024;
const DEFAULT_TIMEOUT_MS = 8_000;
const APP_VERSION = 'unknown';
import {readFileSync} from 'node:fs';

export function isKnownDisease(value) {
  return DISEASES.has(value);
}

function requestId(request) {
  const supplied=request.headers.get('x-request-id')?.trim()||'';
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(supplied)?supplied.toLowerCase():crypto.randomUUID();
}

function routeInfo(path) {
  const fixed=new Map([
    ['/api/v1/health',['/api/v1/health',null]],
    ['/api/v1/ready',['/api/v1/ready',null]],
    ['/api/v1/diseases',['/api/v1/diseases',null]],
  ]);
  if(fixed.has(path))return fixed.get(path);
  const match=/^\/api\/v1\/(diseases|predictions|reports)\/([^/]+)$/.exec(path);
  if(match&&DISEASES.has(match[2]))return [`/api/v1/${match[1]}/{disease}`,match[2]];
  return ['unmatched',null];
}

function safeRequestLog(request,path,id,status,start) {
  const [route,disease]=routeInfo(path);
  const category=status===401?'authentication':status===429?'rate_limit':status===422?'validation':status>=500?'server':status>=400?'client':'none';
  const event={timestamp:new Date().toISOString(),level:status>=500?'ERROR':status>=400?'WARNING':'INFO',event:'bff_http_request',request_id:id,http_method:['GET','POST','PUT','PATCH','DELETE','OPTIONS','HEAD'].includes(request.method)?request.method:'OTHER',route,status_code:status,duration_ms:Number((performance.now()-start).toFixed(2)),disease,error_category:category,app_version:process.env.APP_VERSION||APP_VERSION,environment:process.env.NODE_ENV||'development'};
  if(process.env.NODE_ENV==='production')console.info(JSON.stringify(event));
  else console.info(`bff_http_request request_id=${id} route=${route} status=${status} duration_ms=${event.duration_ms}`);
}

function boundedInteger(raw, fallback, minimum, maximum) {
  const parsed=Number.parseInt(raw||'',10);
  return Number.isFinite(parsed)?Math.min(maximum,Math.max(minimum,parsed)):fallback;
}

function errorResponse(status, code, message, id) {
  return Response.json(
    {error:{code,message,requestId:id},detail:message},
    {status,headers:{'Cache-Control':'no-store','X-Request-ID':id}},
  );
}

function backendConfig(protectedEndpoint) {
  const raw=process.env.BACKEND_INTERNAL_URL?.trim();
  if(!raw) throw new Error('backend_configuration');
  let url;
  try { url=new URL(raw); } catch { throw new Error('backend_configuration'); }
  if(!['http:','https:'].includes(url.protocol)) throw new Error('backend_configuration');
  if(url.username||url.password) throw new Error('backend_configuration');
  let key=process.env.BACKEND_API_KEY?.trim()||'';
  const keyFile=process.env.BACKEND_API_KEY_FILE?.trim();
  if(keyFile) {
    try { key=readFileSync(keyFile,'utf8').trim(); }
    catch { throw new Error('backend_configuration'); }
  }
  if(protectedEndpoint&&process.env.NODE_ENV==='production'&&key.length<32) {
    throw new Error('backend_configuration');
  }
  return {base:url.toString().replace(/\/$/,''),key};
}

async function callBackend(request, path, options) {
  const id=requestId(request);
  const started=performance.now();
  let status=500;
  const finish=(response)=>{status=response.status;return response;};
  try {
  let config;
  try { config=backendConfig(options.protectedEndpoint); }
  catch { return finish(errorResponse(503,'service_unavailable','Backend service is not configured',id)); }

  const headers=new Headers({'X-Request-ID':id});
  if(options.protectedEndpoint&&config.key) headers.set('X-API-Key',config.key);
  let body;
  if(options.method==='POST') {
    if(!(request.headers.get('content-type')||'').toLowerCase().startsWith('application/json')) {
      return finish(errorResponse(415,'unsupported_media_type','Content-Type must be application/json',id));
    }
    const limit=boundedInteger(process.env.BFF_MAX_REQUEST_BYTES,DEFAULT_BODY_LIMIT,1024,64*1024);
    const declared=Number.parseInt(request.headers.get('content-length')||'0',10);
    if(Number.isFinite(declared)&&declared>limit) {
      return finish(errorResponse(413,'request_too_large',`Request body exceeds ${limit} bytes`,id));
    }
    const bytes=await request.arrayBuffer();
    if(bytes.byteLength>limit) {
      return finish(errorResponse(413,'request_too_large',`Request body exceeds ${limit} bytes`,id));
    }
    try { JSON.parse(new TextDecoder().decode(bytes)); }
    catch { return finish(errorResponse(400,'bad_request','Request body must contain valid JSON',id)); }
    body=bytes;
    headers.set('Content-Type','application/json');
  }

  const controller=new AbortController();
  const timeoutMs=boundedInteger(process.env.BACKEND_TIMEOUT_MS,DEFAULT_TIMEOUT_MS,100,30_000);
  const timeout=setTimeout(()=>controller.abort(),timeoutMs);
  const cancel=()=>controller.abort();
  request.signal.addEventListener('abort',cancel,{once:true});
  try {
    const upstream=await fetch(`${config.base}${path}`,{
      method:options.method,
      headers,
      body,
      signal:controller.signal,
      cache:'no-store',
    });
    const responseHeaders=new Headers({
      'Cache-Control':'no-store',
      'X-Request-ID':upstream.headers.get('x-request-id')||id,
    });
    if(options.pdf&&upstream.ok) {
      if(!(upstream.headers.get('content-type')||'').toLowerCase().startsWith('application/pdf')) {
        return finish(errorResponse(502,'bad_gateway','Backend returned an invalid report response',id));
      }
      responseHeaders.set('Content-Type','application/pdf');
      responseHeaders.set('Content-Disposition',`attachment; filename="${options.disease}_report.pdf"`);
    } else {
      responseHeaders.set('Content-Type',upstream.headers.get('content-type')||'application/json');
    }
    return finish(new Response(upstream.body,{status:upstream.status,headers:responseHeaders}));
  } catch(error) {
    if(controller.signal.aborted&&!request.signal.aborted) {
      return finish(errorResponse(504,'gateway_timeout','Backend service timed out',id));
    }
    return finish(errorResponse(502,'bad_gateway','Backend service request failed',id));
  } finally {
    clearTimeout(timeout);
    request.signal.removeEventListener('abort',cancel);
  }
  } finally {
    safeRequestLog(request,path,id,status,started);
  }
}

export function forwardGet(request, path, protectedEndpoint=false) {
  return callBackend(request,path,{method:'GET',protectedEndpoint,pdf:false});
}

export function forwardJson(request, path, disease, pdf=false) {
  return callBackend(request,path,{method:'POST',protectedEndpoint:true,pdf,disease});
}

export function unknownDisease(request) {
  return errorResponse(404,'not_found','Unknown disease',requestId(request));
}
