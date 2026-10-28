# backend/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# System dependencies required by geospatial libraries (GDAL/GEOS)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend source code, offline artifacts (graph/matrix/model) and scripts
COPY backend/ ./backend/
COPY artifacts/ ./artifacts/
COPY scripts/ ./scripts/
COPY pytest.ini ./

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
