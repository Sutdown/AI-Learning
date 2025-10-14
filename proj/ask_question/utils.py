import json
import os
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
llm = ChatTongyi(model="qwen3-max", api_key=DASHSCOPE_API_KEY)

# 商品和目录的数据文件
PRODUCTS_FILE = "products.json"
CATEGORIES_FILE = "categories.json"
DELIMITER = "####"

# -------------------------- 系统提示词（修正原重复类别问题）--------------------------
# 第二步（抽取商品）系统信息文本，校验不同类别，并返回一个列表，其中包含所有类别。
# 提取问题分布解决，此为思维链
step_2_system_message_content = f"""
您将获得一次客户服务对话。最近的用户查询将使用{DELIMITER}字符进行分隔。

输出一个Python对象列表，其中每个对象具有以下格式：
'category': <包括以下几个类别：Computers and Laptops、Smartphones and Accessories、Televisions and Home Theater Systems、Gaming Consoles and Accessories、Audio Equipment、Cameras and Camcorders
'products': <必须是下面的允许产品列表中找到的产品>

类别和产品必须在客户服务查询中找到。
如果提到了产品，它必须与下面的允许产品列表中的正确类别相关联。
如果未找到任何产品或类别，请输出一个空列表。
只列出之前对话的早期部分未提及和讨论的产品和类别。

允许的产品：

Computers and Laptops类别：
TechPro Ultrabook
BlueWave Gaming Laptop
PowerLite Convertible
TechPro Desktop
BlueWave Chromebook

Smartphones and Accessories类别：
SmartX ProPhone
MobiTech PowerCase
SmartX MiniPhone
MobiTech Wireless Charger
SmartX EarBuds

Televisions and Home Theater Systems类别：
CineView 4K TV
SoundMax Home Theater
CineView 8K TV
SoundMax Soundbar
CineView OLED TV

Gaming Consoles and Accessories类别：
GameSphere X
ProGamer Controller
GameSphere Y
ProGamer Racing Wheel
GameSphere VR Headset

Audio Equipment类别：
AudioPhonic Noise-Canceling Headphones
WaveSound Bluetooth Speaker
AudioPhonic True Wireless Earbuds
WaveSound Soundbar
AudioPhonic Turntable

Cameras and Camcorders类别：
FotoSnap DSLR Camera
ActionCam 4K
FotoSnap Mirrorless Camera
ZoomMaster Camcorder
FotoSnap Instant Camera

只输出对象列表，不包含其他内容。
"""

# 第四步（生成用户回答）的系统信息，添加身份，进一步区分，可以理解成prompt chain
step_4_system_message_content = f"""
    你是一家大型电子商店的客户服务助理。
    以友好和乐于助人的语气回答，回答保持简洁明了。
    确保让用户提出相关的后续问题。
"""

# 第六步（验证模型回答）的系统信息，重新根据数据校验结果
# 思维链的一部分，检查结果
step_6_system_message_content = f"""
你是一个助手，评估客户服务代理的回答是否足够回答客户的问题，并验证回答中所有产品信息是否与提供的商品数据一致。
请基于以下三部分内容进行判断：
1. 用户的问题
2. 客服的回答
3. 商品数据集（包含所有产品的真实信息）

输出格式：
Y - 回答足够且所有产品信息与数据集一致
N - 回答不足够，或存在与数据集不符的信息

只输出一个字母。
"""

# -------------------------- 模型调用函数（基于 ChatTongyi）--------------------------
def call_llm(messages):
    """统一的模型调用接口"""
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"[模型错误] {e}")
        return ""

# --------------------------基础数据加载--------------------------
def load_json_file(path):
    """从文件读取 JSON 数据"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_products():
    return load_json_file(PRODUCTS_FILE)

def load_categories():
    return load_json_file(CATEGORIES_FILE)

# ---------------------- 功能函数 ----------------------
def extract_products_and_categories(user_msg):
    """调用模型识别用户提到的产品和类别"""
    messages = [
        SystemMessage(content=step_2_system_message_content),
        HumanMessage(content=f"{DELIMITER}{user_msg}{DELIMITER}")
    ]
    return call_llm(messages)

def read_string_to_list(json_like_str):
    """将模型输出转为 Python 对象"""
    if not json_like_str:
        return []
    try:
        fixed = json_like_str.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        print("[警告] 无法解析模型输出：", json_like_str)
        return []

def generate_product_info(data_list, products):
    """根据识别结果提取产品详情"""
    info_text = ""
    for item in data_list:
        for pname in item.get("products", []):
            if pname in products:
                info_text += json.dumps(products[pname], ensure_ascii=False, indent=2) + "\n"
    return info_text.strip()

def answer_user_question(user_msg, product_info):
    """生成客服回答"""
    messages = [
        SystemMessage(content=step_4_system_message_content),
        HumanMessage(content=f"用户问题：{user_msg}\n\n相关产品信息：\n{product_info}")
    ]
    return call_llm(messages)

def validate_answer(user_msg, answer, products):
    """验证客服回答是否正确（传入商品数据作为参考）"""
    # 将商品数据转为字符串，作为参考信息传入
    products_str = json.dumps(products, ensure_ascii=False, indent=2)
    messages = [
        SystemMessage(content=step_6_system_message_content),
        HumanMessage(content=f"""
用户问题：{user_msg}
客服回答：{answer}
商品数据集：{products_str}
        """.strip())
    ]
    return call_llm(messages)

def main():
    """
    电商客服 AI 系统主流程
    """
    print("=== Step 1: 初始化商品与分类数据 ===")
    products = load_products()

    print("=== Step 2: 模型识别用户提到的商品和类别 ===")
    user_msg = "你好，我想了解一下 SmartX ProPhone 的电池续航，以及 CineView 8K TV 有没有HDR功能？"
    print(f"用户消息：{user_msg}")
    response = extract_products_and_categories(user_msg)
    print(f"模型识别结果（原始文本）：\n{response}")

    data_list = read_string_to_list(response)
    print(f"解析后结构：{data_list}")
    product_info_str = generate_product_info(data_list, products)
    print(f"生成的产品信息：\n{product_info_str}")

    print("\n=== Step 3: 生成客服回答 ===")
    answer = answer_user_question(user_msg, product_info_str)
    print(f"客服回答：\n{answer}")

    print("\n=== Step 4: 检查回答质量 ===")
    validation = validate_answer(user_msg, answer, product_info_str)
    print(f"验证结果（Y=合格，N=不合格）：{validation}")

if __name__ == "__main__":
    main()
