# prefixgate dashboard as a free CPU Hugging Face Space (Docker SDK, 2 vCPU / 16 GB).
# The Space repo holds only this file and README.md; the app is cloned from GitHub at REF.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers==4.46.3 "jinja2>=3.1" safetensors numpy scipy
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface HOST=0.0.0.0 OMP_NUM_THREADS=2 PREFIXGATE_AGENT_LIVE=0
ARG REF=devin/prefixgate-hackathon
# re-clone whenever REF moves
ADD https://api.github.com/repos/Sebastianrodaaa/prefixgate/commits/${REF} /tmp/ref.json
RUN git clone --depth 1 -b ${REF} https://github.com/Sebastianrodaaa/prefixgate /home/user/app
WORKDIR /home/user/app
# bake the pinned Policy Gates data (sha256-checked) and the Qwen2.5-0.5B gate into the image; no Llama download
RUN python -c "from prefixgate.data import build; build()" && python -c "from prefixgate.dm import load; load()"
EXPOSE 7860
CMD ["python", "app/server.py", "7860"]
