"""
昇腾RAG系统核心模块
"""

import os
import yaml
from typing import List, Dict, Any, Optional
from loguru import logger

from .vector_store import VectorStore
from .embedding_service import EmbeddingService
from .llm_service import LLMService


class RAGSystem:
    """RAG系统主类"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化RAG系统"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # 初始化各个组件
        self.embedding_service = EmbeddingService(self.config['embedding'])
        self.vector_store = VectorStore(self.config['vector_store'])
        self.llm_service = LLMService(self.config['llm'])
        
        logger.info("RAG系统初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _setup_logging(self):
        """设置日志"""
        log_level = self.config['system'].get('log_level', 'INFO')
        logger.remove()
        logger.add(
            "logs/rag_system.log",
            rotation="10 MB",
            retention="7 days",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        )
        logger.add(
            lambda msg: print(msg, end=""),
            level=log_level,
            format="{time:HH:mm:ss} | {level} | {message}"
        )
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> bool:
        """添加文档到向量数据库"""
        try:
            # 生成嵌入向量
            embeddings = self.embedding_service.encode(documents)
            
            # 存储到向量数据库
            self.vector_store.add_documents(documents, embeddings, metadata)
            
            logger.info(f"成功添加 {len(documents)} 个文档")
            return True
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        try:
            if top_k is None:
                top_k = self.config['retrieval']['top_k']
            
            # 生成查询嵌入
            query_embedding = self.embedding_service.encode([query])[0]
            
            # 搜索相关文档
            results = self.vector_store.search(query_embedding, top_k)
            
            # 过滤相似度阈值
            threshold = self.config['retrieval']['similarity_threshold']
            filtered_results = [r for r in results if r['score'] >= threshold]
            
            logger.info(f"搜索到 {len(filtered_results)} 个相关文档")
            return filtered_results
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[str]) -> str:
        """基于检索到的文档生成答案"""
        try:
            # 构建提示词
            context = "\n\n".join(context_docs)
            prompt = self._build_prompt(query, context)
            
            # 生成答案
            answer = self.llm_service.generate(prompt)
            
            logger.info("成功生成答案")
            return answer
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return "抱歉，生成答案时出现错误。"
    
    def _build_prompt(self, query: str, context: str) -> str:
        """构建提示词"""
        prompt_template = """基于以下上下文信息，回答用户的问题。如果上下文中没有相关信息，请说明无法回答。

上下文信息：
{context}

用户问题：{query}

请提供准确、详细的回答："""
        
        return prompt_template.format(context=context, query=query)
    
    def query(self, question: str) -> Dict[str, Any]:
        """完整的RAG查询流程"""
        try:
            # 1. 检索相关文档
            search_results = self.search(question)
            
            if not search_results:
                return {
                    "answer": "抱歉，没有找到相关的文档信息。",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # 2. 提取文档内容
            context_docs = [result['content'] for result in search_results]
            
            # 3. 生成答案
            answer = self.generate_answer(question, context_docs)
            
            # 4. 计算置信度（基于最高相似度分数）
            confidence = max([result['score'] for result in search_results])
            
            return {
                "answer": answer,
                "sources": search_results,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return {
                "answer": "抱歉，查询过程中出现错误。",
                "sources": [],
                "confidence": 0.0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        try:
            doc_count = self.vector_store.get_document_count()
            return {
                "document_count": doc_count,
                "embedding_model": self.config['embedding']['model_name'],
                "llm_model": self.config['llm']['model_name'],
                "vector_store_type": self.config['vector_store']['type']
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}