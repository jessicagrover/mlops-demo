# File: Dockerfile
FROM python:3.9-slim
WORKDIR /app

COPY src/app.py .
COPY outputs/model.pkl .

RUN pip install flask scikit-learn numpy

EXPOSE 5000
CMD ["python", "app.py"]
