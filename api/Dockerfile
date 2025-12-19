FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir pandas==1.5.3 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DATA_PATH=./data/creditcard.csv
ENV MODEL_URI=./model/saved_models/model.pkl

# Train the model during build
RUN mkdir -p model/saved_models assets && \
    python model/model.py

#CMD ["python", "model/model.py"]
CMD ["python", "api/app.py"]