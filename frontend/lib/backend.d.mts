export function isKnownDisease(value:string):boolean;
export function forwardGet(request:Request,path:string,protectedEndpoint?:boolean):Promise<Response>;
export function forwardJson(request:Request,path:string,disease:string,pdf?:boolean):Promise<Response>;
export function unknownDisease(request:Request):Response;
