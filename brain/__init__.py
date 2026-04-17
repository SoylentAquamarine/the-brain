"""
brain — The Brain AI orchestration package.

Quick start:
    from brain import Orchestrator
    from brain.task import Task, TaskType

    orchestrator = Orchestrator()
    result = orchestrator.run(Task(prompt="Summarise this text...", task_type=TaskType.SUMMARIZATION))
    print(result.content)
"""

from brain.orchestrator import Orchestrator
from brain.task import Task, TaskResult, TaskType, Priority

__all__ = ["Orchestrator", "Task", "TaskResult", "TaskType", "Priority"]
__version__ = "0.1.0"
