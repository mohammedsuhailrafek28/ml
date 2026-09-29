import {forwardGet,isKnownDisease,unknownDisease} from '../../../../lib/backend.mjs';
export const runtime='nodejs'; export const dynamic='force-dynamic';
export async function GET(request:Request,{params}:{params:Promise<{disease:string}>}){const {disease}=await params;return isKnownDisease(disease)?forwardGet(request,`/api/v1/diseases/${disease}`,true):unknownDisease(request)}
