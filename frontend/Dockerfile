# Exact supported tag and registry-verified multi-platform manifest digest.
FROM node:20.19.0-alpine3.21@sha256:8bda036ddd59ea51a23bc1a1035d3b5c614e72c01366d989f4120e8adca196d4 AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20.19.0-alpine3.21@sha256:8bda036ddd59ea51a23bc1a1035d3b5c614e72c01366d989f4120e8adca196d4 AS build
WORKDIR /app
ARG APP_VERSION
ARG APP_SOURCE_REVISION=unknown
ARG OCI_CREATED=unknown
ENV NEXT_TELEMETRY_DISABLED=1 NODE_ENV=production APP_VERSION=$APP_VERSION APP_SOURCE_REVISION=$APP_SOURCE_REVISION
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20.19.0-alpine3.21@sha256:8bda036ddd59ea51a23bc1a1035d3b5c614e72c01366d989f4120e8adca196d4 AS runtime
WORKDIR /app
ARG APP_VERSION
ARG APP_SOURCE_REVISION=unknown
ARG OCI_CREATED=unknown
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 APP_VERSION=$APP_VERSION APP_SOURCE_REVISION=$APP_SOURCE_REVISION PORT=3000 HOSTNAME=0.0.0.0
LABEL org.opencontainers.image.title="Medical AI Suite Web" \
      org.opencontainers.image.version="$APP_VERSION" \
      org.opencontainers.image.source="https://github.com/mohammedsuhailrafek28/ml" \
      org.opencontainers.image.revision="$APP_SOURCE_REVISION" \
      org.opencontainers.image.created="$OCI_CREATED" \
      org.opencontainers.image.description="Educational/research interface; not clinically validated"
RUN apk upgrade --no-cache \
    && rm -rf /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/corepack \
       /usr/local/bin/npm /usr/local/bin/npx /usr/local/bin/corepack \
    && addgroup -g 10001 app && adduser -u 10001 -G app -D app
COPY --from=build --chown=10001:10001 /app/.next/standalone ./
USER 10001:10001
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:3000/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
STOPSIGNAL SIGTERM
CMD ["node", "server.js"]
