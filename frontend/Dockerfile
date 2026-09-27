# Frontend (Next.js) image. Build from the frontend/ directory:
#   docker build -t medical-ai-web frontend
FROM node:20.19-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20.19-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:20.19-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
RUN addgroup -g 10001 app && adduser -u 10001 -G app -D app
COPY --from=build /app/package.json ./package.json
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/.next ./.next
USER app
EXPOSE 3000
CMD ["npm", "run", "start"]
