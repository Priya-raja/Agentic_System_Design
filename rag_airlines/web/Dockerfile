FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
ARG BACKEND_URL=http://api:8000
ENV BACKEND_URL=$BACKEND_URL
RUN npm run build
FROM node:22-alpine
WORKDIR /app
COPY --from=build /app /app
USER node
EXPOSE 3000
CMD ["sh", "-c", "exec npm run start -- --port ${PORT:-3000}"]
