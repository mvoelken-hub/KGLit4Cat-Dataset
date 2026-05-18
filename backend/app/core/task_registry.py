import asyncio
from contextlib import suppress
from collections.abc import Coroutine
from enum import Enum
from typing import Any
from dataclasses import dataclass

from logging import Logger
from app.core.logging import logger
from app.core.config import Settings, settings

class TaskStillRunningError(Exception):
    """Raised when trying to create a task with a name that is already running."""

class TaskStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    CRASHED = "crashed"

class TaskType(Enum):
    WORKFLOW = "workflow"
    EMBEDDING = "embedding"
    SUMMARIZATION = "summarization"
    CHUNKING = "chunking"
    WAITING = "waiting"
    STARTUP = "startup"
    OTHER = "other"

@dataclass
class TaskInfo:
    task: asyncio.Task
    status: TaskStatus
    type: TaskType

class TaskRegistry:
    """
    In-memory registry for tracking background tasks in the application, allowing for management and monitoring of their status.

    Name convention for tasks: "(type):(description):(id)"
    """

    def __init__(self, settings: Settings, logger: Logger):
        self.tasks: dict[str, TaskInfo] = {}
        self.logger = logger

    async def create_task(self, coro: Coroutine[Any, Any, Any], type: TaskType, name: str) -> asyncio.Task:
        """
        Creates and registers a new background task. If a task with the same name is already running, raises TaskStillRunningError. If a task with the same name exists but is not running, it will be removed and replaced by the new task.

        Args:
            coro: The coroutine to run as a background task.
            type: The type/category of the task, used for organizational purposes.
            name: A unique name for the task, following the convention "(type):(description):(id)".

        Returns:
            The created asyncio.Task object.
        
        """
            
        task_info: TaskInfo | None = self.get_task_info(name)
        
        if task_info:
            if task_info.status == TaskStatus.RUNNING:
                raise TaskStillRunningError(f"Task with name '{name}' is already running.")
            
            await self.remove_task(name)
        
        task = asyncio.create_task(coro, name=name)

        self.tasks[name] = TaskInfo(task=task, status=TaskStatus.RUNNING, type=type)

        task.add_done_callback(self._log_task_outcome)
        task.add_done_callback(lambda t: self._change_task_status(name, self._root_task_status(t)))

        self.logger.info(
            "\n\n\t\t***** Created task '%s' of type '%s' *****\n\n",
            name,
            type.value
        )

        return task

    def get_task_info(self, name: str) -> TaskInfo | None:
        return self.tasks.get(name)
    
    async def cancel_task(self, name: str):
        task_info = self.get_task_info(name)

        if not task_info:
            return

        if task_info.status != TaskStatus.RUNNING:
            return
        
        task_info.task.cancel()

        with suppress(asyncio.CancelledError):
            await task_info.task

    def get_all_running_tasks(self) -> list[TaskInfo]:
        return [task_info for task_info in self.tasks.values() if task_info.status == TaskStatus.RUNNING]
        
    async def cancel_all_tasks(self):
        running_tasks = self.get_all_running_tasks()
        
        for task_info in running_tasks:
            task_info.task.cancel()

        await asyncio.gather(*[task_info.task for task_info in running_tasks], return_exceptions=True)

    async def remove_task(self, name: str) -> None:
        task_info = self.get_task_info(name)
        if task_info is None:
            return
        await self.cancel_task(name)
        del self.tasks[name]

    async def wait_for_task(self, name: str, timeout: float | None = None) -> None:
        task_info = self.get_task_info(name)
        if not task_info:
            raise ValueError(f"No task found with name '{name}'")
        await asyncio.wait_for(asyncio.shield(task_info.task), timeout=timeout)


    # Internal helper methods for task status management and logging

    def _change_task_status(self, name: str, status: TaskStatus) -> None:
        task_info = self.get_task_info(name)
        if task_info:
            task_info.status = status

            self.logger.info(
                "\n\n\t\t***** Task '%s' changed status to '%s' *****\n\n",
                name,
                status.value
            )

    def _root_task_status(self, task: asyncio.Task) -> TaskStatus:
        if task.cancelled():
            return TaskStatus.CANCELLED
        elif task.exception():
            return TaskStatus.CRASHED
        else:
            return TaskStatus.COMPLETED

    def _log_task_outcome(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            self.logger.warning("Task cancelled: %s", task.get_name())
        except Exception:
            self.logger.exception("Task crashed: %s", task.get_name())

task_registry = TaskRegistry(settings=settings, logger=logger)
