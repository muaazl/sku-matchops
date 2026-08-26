# Use node:24-slim to align with the frontend engines requirement (>=24)
FROM node:24-slim

# Set working directory inside container
WORKDIR /app/frontend

# Install pnpm package manager globally
RUN npm install -g pnpm

# Copy package management files to leverage caching during builds
COPY package.json pnpm-lock.yaml .npmrc pnpm-workspace.yaml ./

# Install dependencies using pnpm
RUN pnpm install --no-frozen-lockfile

# Copy all source files
COPY . .

# Expose Vite dev server port
EXPOSE 5173

# Start Vite server, bound to all interfaces
CMD ["pnpm", "start", "--host", "0.0.0.0"]
