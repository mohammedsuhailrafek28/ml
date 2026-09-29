import {forwardGet} from '../../../lib/backend.mjs';
export const runtime='nodejs'; export const dynamic='force-dynamic';
export function GET(request:Request){return forwardGet(request,'/api/v1/ready')}
