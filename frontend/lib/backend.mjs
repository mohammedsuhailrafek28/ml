const DISEASES = new Set(['liver', 'diabetes', 'heart', 'kidney', 'parkinsons']);
const DEFAULT_BODY_LIMIT = 16 * 1024;
const DEFAULT_TIMEOUT_MS = 8_000;

export function isKnownDisease(value) {
  return DISEASES.has(value);
}

function requestId(request) {
  const supplied=request.headers.get('x-request-id')?.trim()||'';
  return /^[A-Za-z0-9._-]{1,128}$/.test(supplied)?supplied:crypto.randomUUID();
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
  const key=process.env.BACKEND_API_KEY?.trim()||'';
  if(protectedEndpoint&&process.env.NODE_ENV==='production'&&key.length<32) {
    throw new Error('backend_configuration');
  }
  return {base:url.toString().replace(/\/$/,''),key};
}

async function callBackend(request, path, options) {
  const id=requestId(request);
  let config;
  try { config=backendConfig(options.protectedEndpoint); }
  catch { return errorResponse(503,'service_unavailable','Backend service is not configured',id); }

  const headers=new Headers({'X-Request-ID':id});
  if(options.protectedEndpoint&&config.key) headers.set('X-API-Key',config.key);
  let body;
  if(options.method==='POST') {
    if(!(request.headers.get('content-type')||'').toLowerCase().startsWith('application/json')) {
      return errorResponse(415,'unsupported_media_type','Content-Type must be application/json',id);
    }
    const limit=boundedInteger(process.env.BFF_MAX_REQUEST_BYTES,DEFAULT_BODY_LIMIT,1024,64*1024);
    const declared=Number.parseInt(request.headers.get('content-length')||'0',10);
    if(Number.isFinite(declared)&&declared>limit) {
      return errorResponse(413,'request_too_large',`Request body exceeds ${limit} bytes`,id);
    }
    const bytes=await request.arrayBuffer();
    if(bytes.byteLength>limit) {
      return errorResponse(413,'request_too_large',`Request body exceeds ${limit} bytes`,id);
    }
    try { JSON.parse(new TextDecoder().decode(bytes)); }
    catch { return errorResponse(400,'bad_request','Request body must contain valid JSON',id); }
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
        return errorResponse(502,'bad_gateway','Backend returned an invalid report response',id);
      }
      responseHeaders.set('Content-Type','application/pdf');
      responseHeaders.set('Content-Disposition',`attachment; filename="${options.disease}_report.pdf"`);
    } else {
      responseHeaders.set('Content-Type',upstream.headers.get('content-type')||'application/json');
    }
    return new Response(upstream.body,{status:upstream.status,headers:responseHeaders});
  } catch(error) {
    if(controller.signal.aborted&&!request.signal.aborted) {
      return errorResponse(504,'gateway_timeout','Backend service timed out',id);
    }
    return errorResponse(502,'bad_gateway','Backend service request failed',id);
  } finally {
    clearTimeout(timeout);
    request.signal.removeEventListener('abort',cancel);
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
