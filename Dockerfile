FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY project/ project/
COPY tools/ tools/
COPY tests/ tests/
COPY main.py .
COPY *.md .

# The image ships code only (no results/ tree), so the default command is
# the dependency-free structural validation. Build the dashboard after
# mounting or copying results/:
#   docker run --rm -v "$PWD/results:/app/results" <image> \
#       python tools/build_dashboard.py
CMD ["python", "main.py", "validate-matrix"]
