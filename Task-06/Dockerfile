FROM python:3.13.5-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model/champion_model.joblib

WORKDIR /app

COPY Task-06/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY Task-06/app.py ./app.py
COPY Task-04/champion_model.joblib ./model/champion_model.joblib

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
