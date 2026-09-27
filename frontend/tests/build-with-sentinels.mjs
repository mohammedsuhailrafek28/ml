process.env.BACKEND_INTERNAL_URL='http://phase4-internal-sentinel:8000';
process.env.BACKEND_API_KEY='phase4-not-a-real-secret-bundle-sentinel-123456';
process.argv=['node','next','build'];
await import('../node_modules/next/dist/bin/next');
