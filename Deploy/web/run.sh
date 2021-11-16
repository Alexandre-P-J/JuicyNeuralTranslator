#!/bin/bash
cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1
#export FLASK_APP=src/main.py
#export FLASK_ENV=development
#export FLASK_ENV=production
#flask run --host=0.0.0.0
gunicorn --bind 0.0.0.0:5000 app:app