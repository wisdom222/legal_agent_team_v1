import streamlit as st
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.team import Team
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.openai import OpenAIChat
from agno.knowledge.embedder.openai import OpenAIEmbedder
import tempfile
import os

# 定义默认的 Base URL，作为输入框的默认值
DEFAULT_BASE_URL = "https://api.zhizengzeng.com/v1"

def init_session_state():
    """Initialize session state variables"""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    if 'openai_base_url' not in st.session_state:
        st.session_state.openai_base_url = DEFAULT_BASE_URL
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = None
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = None
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'legal_team' not in st.session_state:
        st.session_state.legal_team = None
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None
    # Add a new state variable to track processed files
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()

COLLECTION_NAME = "legal_documents"  # Define your collection name

def init_qdrant():
    """Initialize Qdrant client with configured settings."""
    if not all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
        return None
    try:
        # Create Agno's Qdrant instance which implements VectorDb
        vector_db = Qdrant(
            collection=COLLECTION_NAME,
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            embedder=OpenAIEmbedder(
                id="text-embedding-3-small", 
                api_key=st.session_state.openai_api_key,
                base_url=st.session_state.openai_base_url # 使用动态配置的 Base URL
            )
        )
        return vector_db
    except Exception as e:
        st.error(f"🔴 Qdrant 连接失败: {str(e)}")
        return None

def process_document(uploaded_file, vector_db: Qdrant):
    """
    Process document, create embeddings and store in Qdrant vector database
    """
    if not st.session_state.openai_api_key:
        raise ValueError("未提供 OpenAI API 密钥")
        
    os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
    os.environ['OPENAI_BASE_URL'] = st.session_state.openai_base_url # 同时也设置环境变量
    
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        st.info("正在加载并处理文档...")
        
        # Create a Knowledge base with the vector_db
        knowledge_base = Knowledge(
            vector_db=vector_db
        )
        
        # Add the document to the knowledge base
        with st.spinner('📤 正在将文档加载到知识库...'):
            try:
                knowledge_base.add_content(path=temp_file_path)
                st.success("✅ 文档存储成功！")
            except Exception as e:
                st.error(f"加载文档出错: {str(e)}")
                raise
        
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass
            
        return knowledge_base
            
    except Exception as e:
        st.error(f"文档处理错误: {str(e)}")
        raise Exception(f"处理文档时出错: {str(e)}")

def main():
    st.set_page_config(page_title="法律文档分析助手", layout="wide")
    init_session_state()

    st.title("AI 法律智能体团队 👨‍⚖️")

    with st.sidebar:
        st.header("🔑 API 配置")
   
        # 1. OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key if st.session_state.openai_api_key else "",
            help="输入您的 OpenAI API 密钥"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key

        # 2. OpenAI Base URL (新增的输入框)
        base_url = st.text_input(
            "OpenAI Base URL",
            value=st.session_state.openai_base_url,
            help="输入 OpenAI 代理地址（如果使用官方 API 可不填或填官方地址）"
        )
        if base_url:
            st.session_state.openai_base_url = base_url

        st.divider() # 分隔线

        # 3. Qdrant API Key
        qdrant_key = st.text_input(
            "Qdrant API Key",
            type="password",
            value=st.session_state.qdrant_api_key if st.session_state.qdrant_api_key else "",
            help="输入您的 Qdrant API 密钥"
        )
        if qdrant_key:
            st.session_state.qdrant_api_key = qdrant_key

        # 4. Qdrant URL
        qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url if st.session_state.qdrant_url else "",
            help="输入您的 Qdrant 实例 URL"
        )
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url

        if all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
            try:
                if not st.session_state.vector_db:
                    # Make sure we're initializing a QdrantClient here
                    st.session_state.vector_db = init_qdrant()
                    if st.session_state.vector_db:
                        st.success("成功连接到 Qdrant！")
            except Exception as e:
                st.error(f"连接 Qdrant 失败: {str(e)}")

        st.divider()

        if all([st.session_state.openai_api_key, st.session_state.vector_db]):
            st.header("📄 文档上传")
            uploaded_file = st.file_uploader("上传法律文档", type=['pdf'])
            
            if uploaded_file:
                # Check if this file has already been processed
                if uploaded_file.name not in st.session_state.processed_files:
                    with st.spinner("正在处理文档..."):
                        try:
                            # Process the document and get the knowledge base
                            knowledge_base = process_document(uploaded_file, st.session_state.vector_db)
                            
                            if knowledge_base:
                                st.session_state.knowledge_base = knowledge_base
                                # Add the file to processed files
                                st.session_state.processed_files.add(uploaded_file.name)
                                
                                # 获取当前的 Base URL
                                current_base_url = st.session_state.openai_base_url

                                # Initialize agents
                                legal_researcher = Agent(
                                    name="法律研究员",
                                    role="法律研究专家",
                                    model=OpenAIChat(id="gpt-4.1",
                                                     api_key=st.session_state.openai_api_key, 
                                                     base_url=current_base_url), # 使用配置的 Base URL
                                    tools=[DuckDuckGoTools()],
                                    knowledge=st.session_state.knowledge_base,
                                    search_knowledge=True,
                                    instructions=[
                                        "查找并引用相关的法律案例和判例",
                                        "提供带有来源的详细研究摘要",
                                        "引用上传文档中的具体章节",
                                        "始终在知识库中搜索相关信息"
                                    ],
                                    debug_mode=True,
                                    markdown=True
                                )

                                contract_analyst = Agent(
                                    name="合同分析师",
                                    role="合同分析专家",
                                    model=OpenAIChat(id="gpt-4.1",
                                                     api_key=st.session_state.openai_api_key, 
                                                     base_url=current_base_url), # 使用配置的 Base URL
                                    knowledge=st.session_state.knowledge_base,
                                    search_knowledge=True,
                                    instructions=[
                                        "彻底审查合同",
                                        "识别关键条款和潜在问题",
                                        "引用文档中的具体条款"
                                    ],
                                    markdown=True
                                )

                                legal_strategist = Agent(
                                    name="法律策略师", 
                                    role="法律策略专家",
                                    model=OpenAIChat(id="gpt-4.1",
                                                     api_key=st.session_state.openai_api_key, 
                                                     base_url=current_base_url), # 使用配置的 Base URL
                                    knowledge=st.session_state.knowledge_base,
                                    search_knowledge=True,
                                    instructions=[
                                        "制定全面的法律策略",
                                        "提供可执行的建议",
                                        "同时考虑风险和机遇"
                                    ],
                                    markdown=True
                                )

                                # Legal Agent Team
                                st.session_state.legal_team = Team(
                                    name="法律团队负责人",
                                    model=OpenAIChat(id="gpt-4.1",
                                                     api_key=st.session_state.openai_api_key, 
                                                     base_url=current_base_url), # 使用配置的 Base URL
                                    members=[legal_researcher, contract_analyst, legal_strategist],
                                    knowledge=st.session_state.knowledge_base,
                                    search_knowledge=True,
                                    instructions=[
                                        "协调团队成员之间的分析工作",
                                        "提供全面的回复",
                                        "确保所有建议都有适当的来源",
                                        "引用上传文档的具体部分",
                                        "在分配任务前始终先搜索知识库"
                                    ],
                                    debug_mode=True,
                                    markdown=True
                                )
                                
                                st.success("✅ 文档处理完成，团队初始化完毕！")
                                
                        except Exception as e:
                            st.error(f"处理文档出错: {str(e)}")
                else:
                    # File already processed, just show a message
                    st.success("✅ 文档已处理，团队准备就绪！")

            st.divider()
            st.header("🔍 分析选项")
            analysis_type = st.selectbox(
                "选择分析类型",
                [
                    "合同审查",
                    "法律研究",
                    "风险评估",
                    "合规性检查",
                    "自定义查询"
                ]
            )
        else:
            st.warning("请配置所有 API 凭证以继续")

    # Main content area
    if not all([st.session_state.openai_api_key, st.session_state.vector_db]):
        st.info("👈 请在侧边栏配置您的 API 凭证以开始")
    elif not uploaded_file:
        st.info("👈 请上传法律文档以开始分析")
    elif st.session_state.legal_team:
        # Create a dictionary for analysis type icons
        analysis_icons = {
            "合同审查": "📑",
            "法律研究": "🔍",
            "风险评估": "⚠️",
            "合规性检查": "✅",
            "自定义查询": "💭"
        }

        # Dynamic header with icon
        st.header(f"{analysis_icons[analysis_type]} {analysis_type}")
  
        analysis_configs = {
            "合同审查": {
                "query": "审查此合同并识别关键条款、义务和潜在问题。",
                "agents": ["合同分析师"],
                "description": "专注于条款和义务的详细合同分析"
            },
            "法律研究": {
                "query": "研究与此文档相关的案例和判例。",
                "agents": ["法律研究员"],
                "description": "相关法律案例和判例的研究"
            },
            "风险评估": {
                "query": "分析此文档中的潜在法律风险和责任。",
                "agents": ["合同分析师", "法律策略师"],
                "description": "综合风险分析和战略评估"
            },
            "合规性检查": {
                "query": "检查此文档的监管合规性问题。",
                "agents": ["法律研究员", "合同分析师", "法律策略师"],
                "description": "全面的合规性分析"
            },
            "自定义查询": {
                "query": None,
                "agents": ["法律研究员", "合同分析师", "法律策略师"],
                "description": "使用所有可用智能体的自定义分析"
            }
        }

        st.info(f"📋 {analysis_configs[analysis_type]['description']}")
        st.write(f"🤖 活跃法律 AI 智能体: {', '.join(analysis_configs[analysis_type]['agents'])}")  #dictionary!!

        # Replace the existing user_query section with this:
        if analysis_type == "自定义查询":
            user_query = st.text_area(
                "输入您的具体问题:",
                help="添加您想分析的任何具体问题或要点"
            )
        else:
            user_query = None  # Set to None for non-custom queries


        if st.button("开始分析"):
            if analysis_type == "自定义查询" and not user_query:
                st.warning("请输入问题")
            else:
                with st.spinner("正在分析文档..."):
                    try:
                        # Ensure OpenAI API key is set
                        os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
                        os.environ['OPENAI_BASE_URL'] = st.session_state.openai_base_url # 确保环境变量也更新
                        
                        # Combine predefined and user queries
                        if analysis_type != "自定义查询":
                            combined_query = f"""
                            使用上传的文档作为参考：
                            
                            主要分析任务：{analysis_configs[analysis_type]['query']}
                            关注领域：{', '.join(analysis_configs[analysis_type]['agents'])}
                            
                            请搜索知识库并提供文档中的具体引用。
                            """
                        else:
                            combined_query = f"""
                            使用上传的文档作为参考：
                            
                            {user_query}
                            
                            请搜索知识库并提供文档中的具体引用。
                            关注领域：{', '.join(analysis_configs[analysis_type]['agents'])}
                            """

                        response: RunOutput = st.session_state.legal_team.run(combined_query)
                        
                        # Display results in tabs
                        tabs = st.tabs(["分析结果", "关键点", "建议"])
                        
                        with tabs[0]:
                            st.markdown("### 详细分析")
                            if response.content:
                                st.markdown(response.content)
                            else:
                                for message in response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[1]:
                            st.markdown("### 关键点")
                            key_points_response: RunOutput = st.session_state.legal_team.run(
                                f"""基于之前的分析：    
                                {response.content}
                                
                                请用要点形式总结关键点。
                                重点关注来自以下方面的见解：{', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if key_points_response.content:
                                st.markdown(key_points_response.content)
                            else:
                                for message in key_points_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[2]:
                            st.markdown("### 建议")
                            recommendations_response: RunOutput = st.session_state.legal_team.run(
                                f"""基于之前的分析：
                                {response.content}
                                
                                基于分析，您的关键建议是什么，最佳行动方案是什么？
                                提供来自以下方面的具体建议：{', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if recommendations_response.content:
                                st.markdown(recommendations_response.content)
                            else:
                                for message in recommendations_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)

                    except Exception as e:
                        st.error(f"分析过程中出错: {str(e)}")
    else:
        st.info("请上传法律文档以开始分析")

if __name__ == "__main__":
    main()