#!/bin/bash
echo "🧹 清理所有 MySQL Docker 容器..."

echo "停止容器..."
docker stop $(docker ps -q --filter ancestor=mysql) 2>/dev/null

echo "删除容器..."
docker rm -f $(docker ps -aq --filter ancestor=mysql) 2>/dev/null

echo "✅ 清理完成！当前 MySQL 容器："
docker ps -a | grep mysql || echo "无"