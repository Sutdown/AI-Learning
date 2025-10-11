import asyncio
import aiohttp


async def fetch_api(session, url):
    """异步请求API并返回结果"""
    try:
        async with session.get(url, timeout=10) as response:
            return {
                "url": url,
                "status": response.status,
                "data": await response.json()  # 异步等待响应内容
            }
    except Exception as e:
        return {
            "url": url,
            "status": "error",
            "data": str(e)
        }


async def main():
    # API列表
    api_urls = [
        "https://api.github.com/users/octocat",
        "https://api.github.com/users/github",
        "https://api.github.com/users/pytorch",
        "https://api.github.com/users/tensorflow"
    ]

    # 创建异步HTTP客户端
    async with aiohttp.ClientSession() as session:
        # 创建所有任务
        tasks = [fetch_api(session, url) for url in api_urls]
        # 并发执行任务
        results = await asyncio.gather(*tasks)

    # 打印结果
    for res in results:
        print(f"{res['url']} -> {res['status']}")


if __name__ == "__main__":
    # 运行事件循环
    asyncio.run(main())