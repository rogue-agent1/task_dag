#!/usr/bin/env python3
"""Task DAG scheduler — dependency-aware parallel task execution.

Models tasks as a DAG, resolves dependencies, schedules in topological
order with configurable parallelism using thread pool.

Usage:
    python task_dag.py --test
"""
import sys, threading, time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

class Task:
    def __init__(self, name, fn=None, deps=None):
        self.name = name; self.fn = fn or (lambda: None); self.deps = set(deps or [])
        self.result = None; self.error = None; self.done = False; self.duration = 0

class DAGScheduler:
    def __init__(self, workers=4):
        self.tasks = {}; self.workers = workers

    def add(self, name, fn=None, deps=None) -> 'DAGScheduler':
        self.tasks[name] = Task(name, fn, deps); return self

    def validate(self) -> list:
        errors = []
        for name, task in self.tasks.items():
            for dep in task.deps:
                if dep not in self.tasks: errors.append(f"'{name}' depends on unknown '{dep}'")
        if self._has_cycle(): errors.append("Cycle detected")
        return errors

    def _has_cycle(self):
        visited = set(); stack = set()
        def dfs(name):
            visited.add(name); stack.add(name)
            for dep in self.tasks.get(name, Task(name)).deps:
                if dep in stack: return True
                if dep not in visited and dfs(dep): return True
            stack.discard(name); return False
        return any(name not in visited and dfs(name) for name in self.tasks)

    def _topo_levels(self):
        in_deg = {n: len(t.deps) for n, t in self.tasks.items()}
        ready = deque([n for n, d in in_deg.items() if d == 0])
        levels = []; order = []
        while ready:
            level = list(ready); levels.append(level); order.extend(level)
            next_ready = deque()
            for n in level:
                for name, task in self.tasks.items():
                    if n in task.deps:
                        in_deg[name] -= 1
                        if in_deg[name] == 0: next_ready.append(name)
            ready = next_ready
        return levels, order

    def run(self, dry_run=False):
        errors = self.validate()
        if errors: raise ValueError(f"Invalid DAG: {errors}")
        levels, _ = self._topo_levels()
        results = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for level in levels:
                futures = {}
                for name in level:
                    task = self.tasks[name]
                    if dry_run:
                        task.done = True; continue
                    def run_task(t=task):
                        t0 = time.perf_counter()
                        try: t.result = t.fn(); t.done = True
                        except Exception as e: t.error = e
                        t.duration = time.perf_counter() - t0
                    futures[name] = pool.submit(run_task)
                for name, fut in futures.items():
                    fut.result()
                    if self.tasks[name].error:
                        raise RuntimeError(f"Task '{name}' failed: {self.tasks[name].error}")
                    results[name] = self.tasks[name].result
        return results

    def critical_path(self):
        levels, order = self._topo_levels()
        return [level[0] for level in levels] if levels else []

    def summary(self):
        lines = []
        for name, task in self.tasks.items():
            status = "✓" if task.done else ("✗" if task.error else "○")
            deps = ", ".join(task.deps) or "none"
            lines.append(f"  {status} {name} (deps: {deps}) {f'{task.duration*1000:.0f}ms' if task.done else ''}")
        return '\n'.join(lines)


def test():
    print("=== Task DAG Scheduler Tests ===\n")

    dag = DAGScheduler(workers=2)
    log = []
    dag.add("fetch_data", lambda: (time.sleep(0.01), log.append("fetch"), "data")[2])
    dag.add("parse", lambda: (time.sleep(0.01), log.append("parse"), "parsed")[2], deps=["fetch_data"])
    dag.add("validate", lambda: (log.append("validate"), "ok")[1], deps=["parse"])
    dag.add("fetch_config", lambda: (time.sleep(0.01), log.append("config"), "cfg")[2])
    dag.add("build", lambda: (log.append("build"), "built")[1], deps=["validate", "fetch_config"])
    dag.add("deploy", lambda: (log.append("deploy"), "done")[1], deps=["build"])

    # Validate
    errors = dag.validate()
    assert errors == []
    print("✓ DAG validation: no errors")

    # Run
    results = dag.run()
    assert results["deploy"] == "done"
    assert log.index("fetch") < log.index("parse")
    assert log.index("parse") < log.index("validate")
    assert log.index("build") < log.index("deploy")
    print(f"✓ Execution order: {log}")

    # Parallelism: fetch_data and fetch_config should both complete before build
    assert log.index("config") < log.index("build")
    print("✓ Parallel tasks executed")

    print(dag.summary())

    # Cycle detection
    dag2 = DAGScheduler()
    dag2.add("a", deps=["b"]); dag2.add("b", deps=["c"]); dag2.add("c", deps=["a"])
    errors2 = dag2.validate()
    assert any("Cycle" in e for e in errors2)
    print("\n✓ Cycle detected")

    # Missing dep
    dag3 = DAGScheduler()
    dag3.add("x", deps=["missing"])
    assert any("unknown" in e for e in dag3.validate())
    print("✓ Missing dependency detected")

    # Dry run
    dag4 = DAGScheduler()
    dag4.add("a"); dag4.add("b", deps=["a"])
    dag4.run(dry_run=True)
    assert dag4.tasks["b"].done
    print("✓ Dry run")

    print("\nAll tests passed! ✓")

if __name__ == "__main__":
    test() if not sys.argv[1:] or sys.argv[1] == "--test" else None
