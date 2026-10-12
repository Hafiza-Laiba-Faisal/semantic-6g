FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY project/ project/
COPY tools/ tools/
COPY tests/ tests/
COPY main.py .
COPY *.md .

CMD ["python", "tools/build_dashboard.py", "--open"]
