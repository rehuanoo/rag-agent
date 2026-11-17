# 开放 API 接入指南

## 认证方式

开放平台采用 OAuth2 的 client_credentials 模式。客户先在控制台创建应用，获取 client_id 和 client_secret，再调用 /api/v1/oauth/token 换取 access_token。access_token 有效期为 7200 秒，过期后需重新获取。

## 请求规范

所有接口统一使用 HTTPS，请求头须携带 Authorization: Bearer {access_token} 和 Content-Type: application/json。时间戳精确到秒，签名有效期 5 分钟。

## 限流规则

默认限流为每个应用每分钟 600 次请求，突发上限为每秒 30 次。超过限流返回 HTTP 429，响应头 X-RateLimit-Reset 给出重置时间。企业版客户可申请提升至每分钟 3000 次。

## 主要接口

会话接口 /api/v1/chat/completions 用于发送消息并获取回复；知识库接口 /api/v1/knowledge/search 用于检索知识；工单接口 /api/v1/tickets 支持创建和查询工单；客户接口 /api/v1/customers 支持同步客户资料。

## 错误码

401 表示 token 无效或过期；403 表示无接口权限；404 表示资源不存在；429 表示触发限流；500 表示服务内部错误；1001 表示参数校验失败。建议对 429 和 500 采用指数退避重试，最多 3 次。

## Webhook

平台支持会话结束、工单状态变更、满意度评价三类事件推送。推送失败后按 5 秒、30 秒、120 秒的间隔重试 3 次，仍失败则丢弃并记录日志。接收端须在 3 秒内返回 HTTP 200。

## SDK 与示例

官方提供 Python、Java、Go 三种 SDK。Python SDK 安装命令为 pip install xlchat-sdk，初始化时传入 client_id、client_secret 和 region。
