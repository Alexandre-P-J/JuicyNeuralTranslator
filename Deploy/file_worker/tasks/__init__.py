from celery import Celery
app = Celery('translation_tasks', backend='redis://redis:6379/0',
             broker='redis://redis:6379/0')
from tasks import tasks

if __name__ == "__main__":
    app.start()