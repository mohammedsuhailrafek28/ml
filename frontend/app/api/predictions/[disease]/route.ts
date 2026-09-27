import {forwardJson,isKnownDisease,unknownDisease} from '../../../../lib/backend.mjs';
export const runtime='nodejs'; export const dynamic='force-dynamic';
export async function POST(request:Request,{params}:{params:Promise<{disease:string}>}){const {disease}=await params;return isKnownDisease(disease)?forwardJson(request,`/api/v1/predictions/${disease}`,disease):unknownDisease(request)}
