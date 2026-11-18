import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcDir = path.join(__dirname, 'src');
const distDir = path.join(__dirname, 'dist');
const assetsDir = path.join(distDir, 'assets');

fs.mkdirSync(assetsDir, { recursive: true });

// Read and validate source assets
const htmlSrc = fs.readFileSync(path.join(srcDir, 'index.html'), 'utf8');
const cssSrc = fs.readFileSync(path.join(srcDir, 'styles.css'), 'utf8');
const jsSrc = fs.readFileSync(path.join(srcDir, 'app.js'), 'utf8');

// Basic syntax check on JS bundle before writing production build
new Function(jsSrc);

// Write production assets
fs.writeFileSync(path.join(assetsDir, 'styles.css'), cssSrc, 'utf8');
fs.writeFileSync(path.join(assetsDir, 'app.js'), jsSrc, 'utf8');
fs.writeFileSync(path.join(distDir, 'index.html'), htmlSrc, 'utf8');

console.log('[ArgonClip Build] Production build succeeded -> frontend/dist/');
console.log(`  - index.html (${Buffer.byteLength(htmlSrc)} bytes)`);
console.log(`  - assets/styles.css (${Buffer.byteLength(cssSrc)} bytes)`);
console.log(`  - assets/app.js (${Buffer.byteLength(jsSrc)} bytes)`);
