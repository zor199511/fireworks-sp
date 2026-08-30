# fireworks-sp systemd 部署

## 选 systemd 调度(推荐)还是 cron 调度

- **systemd**(推荐): 自带日志/重启/超时控制,失败易排查。生产建议。
- **cron**(现有,简单): 已有 `scripts/run_daily.sh` 在 cron 跑,直接能用。

两者**二选一**,不要同时开(会重复推送/写库)。

## systemd 部署步骤

```bash
# 1. 复制 unit 文件(必须先修改 User, WorkingDirectory, ExecStart 路径)
cp deploy/systemd/fireworks-dashboard.service ~/.config/systemd/user/
cp deploy/systemd/fireworks-daily.service ~/.config/systemd/user/
cp deploy/systemd/fireworks-daily.timer ~/.config/systemd/user/

# 2. 启用用户 systemd(若首次)
loginctl enable-linger $USER

# 3. 启动 + 开机自启
systemctl --user daemon-reload
systemctl --user enable --now fireworks-dashboard.service
systemctl --user enable --now fireworks-daily.timer

# 4. 检查状态
systemctl --user status fireworks-dashboard
systemctl --user list-timers fireworks-daily.timer
```

## cron 部署(现有)

`scripts/run_daily.sh` 已有 cron 调度:
```
# crontab -e
30 17 * * 1-5 /home/zor/fireworks-sp/scripts/run_daily.sh >> /home/zor/fireworks-sp/logs/cron.log 2>&1
```

## 反向代理(Streamlit 鉴权)

`dashboard.py` 8501 默认无鉴权,生产建议加 nginx + basic_auth:

参考 `deploy/nginx.conf.example` (TBD):
```nginx
server {
    listen 443 ssl;
    server_name fwsp.example.com;
    location / {
        auth_basic "fireworks-sp";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://127.0.0.1:8501;
    }
}
```

## 文件锁(防并发)

跨脚本并发写 active_factors/daily_qfq 用 `fwsp/lock.py` 的 fcntl 文件锁,锁文件在 `/tmp/fwsp-locks/`。
