"""跨脚本文件锁 — 防止 systemd timer + 用户手动跑 / 多个 dashboard 按钮
并发写 active_factors / factor_eval.selected / refetch_qfq daily_qfq 留下
不一致状态。

使用 fcntl.flock 建议锁,非阻塞,获取失败立即 return False(让上游决定
log.warning + sys.exit(0) 还是 log.warning + skip)。
"""
import contextlib
import fcntl
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("fwsp.lock")

LOCK_DIR = Path("/tmp/fwsp-locks")
LOCK_DIR.mkdir(exist_ok=True)

# 各关键操作独占锁名(避免不同操作互踩)
LOCK_ACTIVE = LOCK_DIR / "active_factors.lock"        # active_factors 写
LOCK_QFQ = LOCK_DIR / "refetch_qfq.lock"              # daily_qfq 重抓
LOCK_EVOLVE = LOCK_DIR / "auto_evolve.lock"           # auto_evolve 跑
LOCK_RECOMMEND = LOCK_DIR / "recommend.lock"          # screener.run_screen persist=True


@contextlib.contextmanager
def file_lock(path: Path = LOCK_ACTIVE, timeout: float = 0.0,
              op: str = "?"):
    """非阻塞文件锁上下文管理器,timeout=0 立即失败。

    用法:
        with file_lock(LOCK_ACTIVE, op="set_active_factors") as ok:
            if not ok:
                log.warning("active_factors 锁被其他进程持有, 跳过")
                return
            do_set_active_factors()
    """
    fd = None
    acquired = False
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
        start = time.time()
        # fcntl.LOCK_NB 非阻塞, timeout>0 用 LOCK_EX (阻塞) + sleep 重试
        if timeout <= 0:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except (BlockingIOError, OSError):
                acquired = False
        else:
            while time.time() - start < timeout:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    time.sleep(0.1)
        if not acquired:
            log.warning("锁 %s 被其他进程持有, op=%s 跳过", path.name, op)
        yield acquired
    finally:
        if fd is not None:
            if acquired:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except Exception:  # noqa: BLE001
                    pass
            os.close(fd)
