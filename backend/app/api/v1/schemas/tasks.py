

from pydantic import BaseModel, Field

from app.core.task_registry import TaskRegistry, TaskInfo

class TaskResponse(BaseModel):
    name: str
    status: str
    type: str
    done: bool
    cancelled: bool
    exception: str | None = None

def _serialize_task(name: str, task_info: TaskInfo | None) -> TaskResponse:
    
    if task_info is None:
        return TaskResponse(name=name, status="unknown", type="unknown", done=False, cancelled=False, exception=None)

    task = task_info.task
    exception: str | None = None
    if task.done() and not task.cancelled():
        task_exception = task.exception()
        if task_exception is not None:
            exception = f"{task_exception.__class__.__name__}: {task_exception}"

    return TaskResponse(
        name=name,
        status=task_info.status.value,
        type=task_info.type.value,
        done=task.done(),
        cancelled=task.cancelled(),
        exception=exception
    )