import {cpSync, existsSync, mkdirSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const buildRoot = path.join(frontendRoot, '.next');
const standaloneRoot = path.join(buildRoot, 'standalone');
const standaloneNext = path.join(standaloneRoot, '.next');

if (!existsSync(path.join(standaloneRoot, 'server.js'))) {
  throw new Error('Next.js standalone build output is missing');
}
mkdirSync(standaloneNext, {recursive: true});
cpSync(path.join(buildRoot, 'static'), path.join(standaloneNext, 'static'), {recursive: true, force: true});
if (existsSync(path.join(frontendRoot, 'public'))) {
  cpSync(path.join(frontendRoot, 'public'), path.join(standaloneRoot, 'public'), {recursive: true, force: true});
}
console.log('Standalone static assets prepared');
