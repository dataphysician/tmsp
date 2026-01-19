import { PROVIDERS, PROVIDER_MODELS, DEFAULT_MODELS } from './src/lib/constants';
console.log('PROVIDERS:', JSON.stringify(PROVIDERS, null, 2));
console.log('Has VERTEXAI:', 'VERTEXAI' in PROVIDERS);
console.log('vertexai models:', PROVIDER_MODELS.vertexai);
console.log('vertexai default:', DEFAULT_MODELS.vertexai);