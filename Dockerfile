FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# PyTorch CPU (image plus légère ; l'inférence d'un modèle nano n'exige pas de GPU)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir ultralytics fastapi "uvicorn[standard]" python-multipart opencv-python-headless
COPY common.py api.py ./
COPY runs/cls/seed0/weights/best.pt ./model/best.pt
ENV MODEL_PATH=/app/model/best.pt
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
