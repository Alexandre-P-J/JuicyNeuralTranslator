#!/bin/sh

export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
celery -A tasks worker -Q file --loglevel=WARNING