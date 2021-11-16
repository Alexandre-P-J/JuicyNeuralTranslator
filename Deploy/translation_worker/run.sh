#!/bin/sh

export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
celery -A tasks worker -P solo -Q translation_high --detach --loglevel=WARNING
celery -A tasks worker -P solo -Q translation_low,translation_high --loglevel=WARNING