"""
Background Task Manager — Thread-safe async generation for CodebookLM.

Manages background LLM response generation so that:
  - Generation continues after switching chats
  - Other chats remain usable during generation
  - Finished results appear when revisiting a chat

Engineering constraints (from addendum):
  - Each task has: task_id, chat_id, status, progress, created_at, completed_at, result, error
  - Shared state protected with threading.Lock()
  - Every background thread is daemon=True
  - Automatically clean completed tasks after configurable timeout
  - Prevent multiple simultaneous generations for the same chat
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    """Represents a single background generation task."""
    task_id: str
    chat_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    question: str = ""


class BackgroundTaskManager:
    """Thread-safe manager for background LLM generation tasks.

    Usage:
        manager = BackgroundTaskManager.get_instance()
        task_id = manager.submit(chat_id, question, generate_fn, **kwargs)
        status = manager.get_status(task_id)
        result = manager.get_result(chat_id)  # returns latest completed result for chat
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, cleanup_timeout_seconds: int = 300):
        self._lock = threading.Lock()
        self._tasks: Dict[str, BackgroundTask] = {}
        self._chat_tasks: Dict[str, str] = {}  # chat_id -> active task_id
        self._cleanup_timeout = cleanup_timeout_seconds
        self._started_cleanup = False

    @classmethod
    def get_instance(cls, cleanup_timeout_seconds: int = 300):
        """Return the singleton instance (thread-safe)."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(cleanup_timeout_seconds)
            return cls._instance

    def submit(
        self,
        chat_id: str,
        question: str,
        generate_fn: Callable,
        **kwargs,
    ) -> str:
        """Submit a background generation task.

        Args:
            chat_id: The chat this generation belongs to.
            question: The user question.
            generate_fn: Callable that performs the generation.
                Must accept (question, progress_fn=, **kwargs) and return
                (answer, sources, mode, best_score, context_truncated).
            **kwargs: Additional keyword arguments for generate_fn.

        Returns:
            The task_id of the submitted task.

        Raises:
            RuntimeError: If a task is already running for this chat_id.
        """
        with self._lock:
            # Prevent duplicate generation for same chat
            if chat_id in self._chat_tasks:
                existing_id = self._chat_tasks[chat_id]
                existing = self._tasks.get(existing_id)
                if existing and existing.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    raise RuntimeError(
                        f"Generation already in progress for chat {chat_id}. "
                        f"Task {existing_id} is {existing.status.value}."
                    )

            task_id = str(uuid.uuid4())[:8]
            task = BackgroundTask(
                task_id=task_id,
                chat_id=chat_id,
                question=question,
            )
            self._tasks[task_id] = task
            self._chat_tasks[chat_id] = task_id

        # Start generation in a daemon thread
        thread = threading.Thread(
            target=self._run_task,
            args=(task_id, generate_fn, question),
            kwargs=kwargs,
            daemon=True,
            name=f"bg-gen-{task_id}",
        )
        thread.start()

        # Start cleanup thread if not already running
        self._ensure_cleanup_thread()

        logger.info("Submitted background task %s for chat %s", task_id, chat_id)
        return task_id

    def _run_task(self, task_id: str, generate_fn: Callable, question: str, **kwargs):
        """Execute the generation task in a background thread."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.status = TaskStatus.RUNNING

        def progress_fn(msg):
            with self._lock:
                t = self._tasks.get(task_id)
                if t:
                    t.progress = msg

        try:
            result = generate_fn(question, progress_fn=progress_fn, **kwargs)
            # Unpack the 5-tuple from ask()
            answer, sources, mode, best_score, ctx_truncated = result
            with self._lock:
                task = self._tasks.get(task_id)
                if task:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now().isoformat()
                    task.result = {
                        "answer": answer,
                        "sources": sources,
                        "mode": mode,
                        "best_score": best_score,
                        "context_truncated": ctx_truncated,
                    }
                    task.progress = "Done!"
            logger.info("Task %s completed successfully", task_id)
        except Exception as e:
            with self._lock:
                task = self._tasks.get(task_id)
                if task:
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now().isoformat()
                    task.error = str(e)
                    task.progress = f"Error: {e}"
            logger.error("Task %s failed: %s", task_id, e)

    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            return {
                "task_id": task.task_id,
                "chat_id": task.chat_id,
                "status": task.status.value,
                "progress": task.progress,
                "created_at": task.created_at,
                "completed_at": task.completed_at,
                "has_result": task.result is not None,
                "error": task.error,
                "question": task.question,
            }

    def get_result(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest completed result for a chat, if available.

        Returns None if no completed task exists for this chat.
        Does NOT remove the result — call consume_result() for that.
        """
        with self._lock:
            task_id = self._chat_tasks.get(chat_id)
            if not task_id:
                return None
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.COMPLETED:
                return None
            return task.result

    def consume_result(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get and remove the latest completed result for a chat.

        Returns None if no completed task exists.
        """
        with self._lock:
            task_id = self._chat_tasks.get(chat_id)
            if not task_id:
                return None
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.COMPLETED:
                return None
            result = task.result
            # Clean up
            del self._chat_tasks[chat_id]
            del self._tasks[task_id]
            return result

    def is_generating(self, chat_id: str) -> bool:
        """Check if a generation is currently in progress for a chat."""
        with self._lock:
            task_id = self._chat_tasks.get(chat_id)
            if not task_id:
                return False
            task = self._tasks.get(task_id)
            return task is not None and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)

    def get_progress(self, chat_id: str) -> str:
        """Get the current progress message for a chat's active task."""
        with self._lock:
            task_id = self._chat_tasks.get(chat_id)
            if not task_id:
                return ""
            task = self._tasks.get(task_id)
            return task.progress if task else ""

    def cancel(self, chat_id: str) -> bool:
        """Cancel a running task for a chat (marks as cancelled, thread will finish naturally)."""
        with self._lock:
            task_id = self._chat_tasks.get(chat_id)
            if not task_id:
                return False
            task = self._tasks.get(task_id)
            if task and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now().isoformat()
                del self._chat_tasks[chat_id]
                logger.info("Cancelled task %s for chat %s", task_id, chat_id)
                return True
            return False

    def _ensure_cleanup_thread(self):
        """Start the cleanup thread if not already running."""
        with self._lock:
            if self._started_cleanup:
                return
            self._started_cleanup = True

        thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="bg-cleanup",
        )
        thread.start()

    def _cleanup_loop(self):
        """Periodically clean up completed/failed tasks older than timeout."""
        while True:
            time.sleep(60)  # Check every minute
            cutoff = time.time() - self._cleanup_timeout
            with self._lock:
                to_remove = []
                for task_id, task in self._tasks.items():
                    if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                        if task.completed_at:
                            try:
                                completed_ts = datetime.fromisoformat(task.completed_at).timestamp()
                                if completed_ts < cutoff:
                                    to_remove.append(task_id)
                            except (ValueError, TypeError):
                                to_remove.append(task_id)
                for task_id in to_remove:
                    task = self._tasks.pop(task_id, None)
                    if task:
                        self._chat_tasks.pop(task.chat_id, None)
                if to_remove:
                    logger.info("Cleaned up %d expired background tasks", len(to_remove))
