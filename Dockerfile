FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY ./app /app

COPY ./app/brewnames.joblib /app

COPY ./app/stdscale.joblib /app

COPY ./app/target.joblib /app

COPY ./app/pytorch_beer_selector.pt /app

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]