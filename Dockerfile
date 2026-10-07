# Python 3.10 ka lightweight version use karo
FROM python:3.10-slim

# Working directory set karo
WORKDIR /app

# Requirements file copy karo aur install karo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baaki saara project code copy karo
COPY . .

# Hugging Face ke liye port 7860 expose karo
EXPOSE 7860

# FastAPI server start karne ka command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]