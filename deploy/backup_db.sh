#!/usr/bin/env bash
# fireworks-sp 每日 DB 备份 — 保留 7 天本地, 30 天压缩归档
# 子代理 2 轮 R2-运维2: 1.1GB 单 DB 无备份策略, 任何 disk 坏/误删都重建不了
#
# cron: 0 23 * * *  /home/zor/fireworks-sp/deploy/backup_db.sh >> /home/zor/fireworks-sp/logs/backup.log 2>&0
set -euo pipefail

DB="/home/zor/fireworks-sp/data/fireworks.db"
BACKUP_DIR="/home/zor/fireworks-sp/data/backup"
KEEP_DAYS=7
ARCHIVE_DAYS=30
TS=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR/daily" "$BACKUP_DIR/archive"

if [ ! -f "$DB" ]; then
    echo "[$(date -Iseconds)] DB not found: $DB" >&2
    exit 1
fi

# 1. 在线热备份(用 sqlite .backup, 不会锁住 writer)
DAILY="$BACKUP_DIR/daily/fireworks_${TS}.db"
sqlite3 "$DB" ".backup '$DAILY'"
echo "[$(date -Iseconds)] daily backup: $DAILY ($(du -h "$DAILY" | cut -f1))"

# 2. 保留最近 N 天
find "$BACKUP_DIR/daily" -name "fireworks_*.db" -mtime +$KEEP_DAYS -exec rm -f {} \;

# 3. 月度归档(每月 1 号压缩归档)
if [ "$(date +%d)" = "01" ]; then
    ARCHIVE="$BACKUP_DIR/archive/fireworks_${TS}_monthly.tar.gz"
    find "$BACKUP_DIR/daily" -name "fireworks_*.db" | tar czf "$ARCHIVE" -T -
    echo "[$(date -Iseconds)] monthly archive: $ARCHIVE"
fi

# 4. 清理超长归档(30 天前)
find "$BACKUP_DIR/archive" -name "*.tar.gz" -mtime +$ARCHIVE_DAYS -exec rm -f {} \;

# 5. 写 meta 让 dashboard 显示最近备份时间
.venv/bin/python -c "
import sqlite3
from datetime import datetime
c = sqlite3.connect('$DB')
c.execute('INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)', ('last_backup_at', datetime.now().isoformat(timespec='seconds')))
c.commit()
print('[meta] last_backup_at updated')
" || true

echo "[$(date -Iseconds)] backup done"
