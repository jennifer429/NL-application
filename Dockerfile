FROM node:20-slim
ENV NODE_ENV=production
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --omit=dev && npm cache clean --force
COPY src ./src
COPY tsconfig.json ./
EXPOSE 8080
CMD ["npx", "tsx", "src/server.ts"]
