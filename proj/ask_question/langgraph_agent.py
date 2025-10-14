import os
import json
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# 载入环境变量
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化模型
llm = ChatTongyi(model="qwen3-max", api_key=DASHSCOPE_API_KEY, temperature=0.3)

# -------------------------
# 定义节点函数（Graph Nodes）
# -------------------------

def extract_products(state):
    """Step 1: 从用户问题中抽取商品与类别"""
    user_msg = state["user_msg"]

    system_prompt = """
你将获得一段电商客服对话，请从中提取出“商品名称”和“所属类别”。
输出格式为JSON数组，例如：
[{"category": "Televisions", "product_name": "CineView 8K TV"}]
"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg)
    ]

    result = llm.invoke(messages)
    try:
        extracted = json.loads(result.content)
    except Exception:
        extracted = [{"category": "Unknown", "product_name": "Unknown"}]

    state["extracted_products"] = extracted
    return state


def query_product_info(state):
    """Step 2: 根据识别结果从本地文件查找商品信息"""
    products_file = "products.json"
    if not os.path.exists(products_file):
        raise FileNotFoundError("products.json 文件不存在，请先创建。")

    with open(products_file, "r", encoding="utf-8") as f:
        all_products = json.load(f)

    extracted = state["extracted_products"]
    product_info_list = []

    for item in extracted:
        name = item.get("product_name", "")
        found = next((p for p in all_products if p["name"] == name), None)
        if found:
            product_info_list.append(found)

    state["product_info"] = product_info_list
    return state


def generate_answer(state):
    """Step 3: 基于商品信息生成客服回答"""
    user_msg = state["user_msg"]
    product_info = state.get("product_info", [])

    info_str = "\n".join([f"{p['name']}：{p['description']}" for p in product_info])

    system_prompt = """
你是一名电商智能客服，请根据提供的商品信息回答用户的问题。
若无法回答，请礼貌地提示用户。
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"用户问题：{user_msg}\n\n商品信息：\n{info_str}")
    ]

    result = llm.invoke(messages)
    state["final_answer"] = result.content
    return state


# -------------------------
# 构建 LangGraph 流程
# -------------------------

graph = StateGraph()

graph.add_node("extract", extract_products)
graph.add_node("query", query_product_info)
graph.add_node("answer", generate_answer)

graph.set_entry_point("extract")
graph.add_edge("extract", "query")
graph.add_edge("query", "answer")
graph.add_edge("answer", END)

app = graph.compile()

# -------------------------
# 执行入口
# -------------------------

if __name__ == "__main__":
    user_input = input("用户问题：")
    result = app.invoke({"user_msg": user_input})
    print("\n🤖 客服回答：")
    print(result["final_answer"])
