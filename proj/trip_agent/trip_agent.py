import os
import requests
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.memory import ConversationSummaryMemory
from langchain.agents import initialize_agent, AgentType
from datetime import datetime

load_dotenv()

"""tool"""
@tool("谷歌搜索",
      description="使用 Google 搜索获取实时信息，如‘北京天气’、‘东京景点推荐’、‘签证政策’等")
def google_search(query: str) -> str:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "未设置 SERPER_API_KEY 环境变量"

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": 5}

    try:
        res = requests.post(url, headers=headers, json=payload)
        data = res.json()

        if "organic" not in data:
            return "未找到搜索结果。"

        results = [
            f"{r.get('title')}: {r.get('snippet')} ({r.get('link')})"
            for r in data["organic"][:3]
        ]
        return "\n\n".join(results)

    except Exception as e:
        return f"搜索出错: {e}"

@tool("计算旅行预算",
      description="计算旅行总预算，输入格式为“天数,每日预算”（如“3,500”），返回总预算结果",
      return_direct=False)
def calculate_budget(input_str: str) -> str:
    """计算旅行总预算，输入格式为“天数,每日预算”（如“3,500”）"""
    try:
        days, daily_cost = input_str.split(",")
        total = int(days) * int(daily_cost)
        return f"总预算：{days}天×{daily_cost}元/天={total}元"
    except ValueError:
        return "输入格式错误，请使用“天数,每日预算”（如“3,500”）"

tools = [google_search, calculate_budget]

"""llm"""
llm = ChatTongyi(model="qwen3-max", api_key=os.getenv("DASHSCOPE_API_KEY"))

"""memory"""
# 摘要记忆，节省token
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

"""prompt"""
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能旅行助手，拥有以下能力：
1. 当用户提出涉及实时信息（如天气、景点、签证等）的问题时，请调用【谷歌搜索】工具。
2. 当用户要求预算计算时，调用【计算旅行预算】工具。
3. 其它情况（例如行程建议、交通说明）直接回答。
4. 保持上下文一致性（chat_history），比如“刚才的城市”指之前提到的城市。

工作流程：
- 先分析用户问题是否需要工具：需要则调用，不需要则直接回答
- 调用工具时严格遵循工具的输入格式
- 结合历史对话（chat_history）理解上下文（如“刚才说的城市”指之前提到的城市）
- 用中文简洁回答，避免冗余
"""),
    MessagesPlaceholder(variable_name="chat_history"),  # 插入历史对话
    ("user", "{input}"),                               # 当前用户输入
    ("ai", "{agent_scratchpad}")                       # Agent思考过程（自动填充）
])

"""trip_agent"""
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,  # 带记忆的聊天型Agent
    memory=memory,  # 使用记忆
    # verbose=True,  # 输出详细日志
    agent_kwargs={
        "prompt": prompt,  # 绑定提示词模板
        "system_message": prompt.messages[0].prompt.template,  # 系统提示
        "extra_prompt_messages": [
            MessagesPlaceholder(variable_name="chat_history")
        ]
    },
    handle_parsing_errors=True,  # 忽略解析错误
    return_intermediate_steps=False
)

"""多轮对话"""
def chat_loop():
    print("智能旅行助手")
    print("提示：输入任意问题与我对话，例如：'我打算去东京玩三天' 或 '帮我查一下北京天气'")
    print("输入 'exit' 或 '退出' 可结束对话。\n")

    while True:
        user_input = input("用户：").strip()
        if user_input.lower() in ["exit", "quit", "退出", "bye"]:
            print("助手：好的，下次再见，祝你旅途愉快！👋")
            break

        if not user_input:
            continue  # 忽略空输入

        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        query = f"{user_input}\n（当前时间：{current_time}）"

        try:
            response = agent.invoke({"input": query})
            result = response["output"] if isinstance(response, dict) and "output" in response else response
            print(f"助手：{result}\n")
        except Exception as e:
            print(f"❌ 出错：{e}\n")


if __name__ == "__main__":
    chat_loop()


