import asyncio
import unittest

from app.core.task_registry import TaskRegistry, TaskStatus, TaskType


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


class TaskRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_for_task_timeout_does_not_cancel_task(self):
        registry = TaskRegistry(settings=None, logger=FakeLogger())  # type: ignore[arg-type]

        async def slow_task():
            await asyncio.sleep(0.05)

        await registry.create_task(slow_task(), type=TaskType.STARTUP, name="startup:slow:01")

        with self.assertRaises(TimeoutError):
            await registry.wait_for_task("startup:slow:01", timeout=0.001)

        task_info = registry.get_task_info("startup:slow:01")
        self.assertIsNotNone(task_info)
        self.assertFalse(task_info.task.cancelled())  # type: ignore[union-attr]

        await registry.wait_for_task("startup:slow:01", timeout=1.0)
        self.assertEqual(task_info.status, TaskStatus.COMPLETED)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
