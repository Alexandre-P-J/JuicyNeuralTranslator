from celery import Celery
import os
import uuid

def unique_filename() -> str:
    return str(uuid.uuid4().hex)

app = Celery('translation_tasks', backend='redis://redis:6379/0',
             broker='redis://redis:6379/0')
STORAGE_DIR = os.getenv("STORAGE_MOUNT_DIR")

from tasks import tasks

if __name__ == "__main__":
    app.start()