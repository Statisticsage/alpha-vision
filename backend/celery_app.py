from celery import Celery

celery_app = Celery(
    "worker",
    backend="redis://localhost:6379/0",
    broker="redis://localhost:6379/0"
)

@celery_app.task
def sample_task(x, y):
    return x + y

