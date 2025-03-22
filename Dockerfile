FROM python:3.12-slim

WORKDIR /app

# Install uv package manager
RUN pip install uv

# Copy dependencies
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv pip install --system -e .

# Copy application code and startup script
COPY server/ ./server/
COPY scripts/start.sh ./

# Create non-root user and set permissions
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port that the application uses
EXPOSE 8000

# Run the startup script
CMD ["/app/start.sh"] 
