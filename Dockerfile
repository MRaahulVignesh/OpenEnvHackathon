FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY apex_env/ ./apex_env/
COPY main.py .
COPY .env .

# Expose port
EXPOSE 8000

# Run the server
CMD ["uv", "run", "python", "main.py"]
