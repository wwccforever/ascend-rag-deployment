"""
向量数据库模块
支持Chroma和FAISS两种向量数据库
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB未安装，将无法使用Chroma向量数据库")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS未安装，将无法使用FAISS向量数据库")


class VectorStore:
    """向量数据库抽象类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.store_type = config.get('type', 'chroma').lower()
        
        if self.store_type == 'chroma':
            if not CHROMA_AVAILABLE:
                raise ImportError("ChromaDB未安装，请运行: pip install chromadb")
            self._init_chroma()
        elif self.store_type == 'faiss':
            if not FAISS_AVAILABLE:
                raise ImportError("FAISS未安装，请运行: pip install faiss-cpu")
            self._init_faiss()
        else:
            raise ValueError(f"不支持的向量数据库类型: {self.store_type}")
    
    def _init_chroma(self):
        """初始化ChromaDB"""
        try:
            persist_directory = self.config.get('persist_directory', './data/vector_store')
            os.makedirs(persist_directory, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            collection_name = self.config.get('collection_name', 'documents')
            try:
                self.collection = self.client.get_collection(collection_name)
                logger.info(f"加载现有ChromaDB集合: {collection_name}")
            except:
                self.collection = self.client.create_collection(collection_name)
                logger.info(f"创建新的ChromaDB集合: {collection_name}")
                
        except Exception as e:
            logger.error(f"初始化ChromaDB失败: {e}")
            raise
    
    def _init_faiss(self):
        """初始化FAISS"""
        try:
            self.persist_directory = self.config.get('persist_directory', './data/vector_store')
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self.index_path = os.path.join(self.persist_directory, 'faiss.index')
            self.metadata_path = os.path.join(self.persist_directory, 'metadata.json')
            
            # 尝试加载现有索引
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info("加载现有FAISS索引")
            else:
                # 创建新索引（暂时使用768维，实际使用时会根据嵌入维度调整）
                self.index = None
                logger.info("将创建新的FAISS索引")
            
            # 加载元数据
            self.documents = []
            self.metadata = []
            if os.path.exists(self.metadata_path):
                import json
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', [])
                    self.metadata = data.get('metadata', [])
                    
        except Exception as e:
            logger.error(f"初始化FAISS失败: {e}")
            raise
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadata: Optional[List[Dict]] = None):
        """添加文档到向量数据库"""
        try:
            if metadata is None:
                metadata = [{"id": i} for i in range(len(documents))]
            
            if self.store_type == 'chroma':
                self._add_to_chroma(documents, embeddings, metadata)
            elif self.store_type == 'faiss':
                self._add_to_faiss(documents, embeddings, metadata)
                
            logger.info(f"成功添加 {len(documents)} 个文档到向量数据库")
            
        except Exception as e:
            logger.error(f"添加文档到向量数据库失败: {e}")
            raise
    
    def _add_to_chroma(self, documents: List[str], embeddings: np.ndarray, metadata: List[Dict]):
        """添加文档到ChromaDB"""
        ids = [f"doc_{i}_{len(self.collection.get()['ids'])}" for i in range(len(documents))]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
    
    def _add_to_faiss(self, documents: List[str], embeddings: np.ndarray, metadata: List[Dict]):
        """添加文档到FAISS"""
        # 如果索引不存在，创建新索引
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
            logger.info(f"创建FAISS索引，维度: {dimension}")
        
        # 添加向量
        self.index.add(embeddings.astype('float32'))
        
        # 保存文档和元数据
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        # 持久化
        self._save_faiss()
    
    def _save_faiss(self):
        """保存FAISS索引和元数据"""
        try:
            faiss.write_index(self.index, self.index_path)
            
            import json
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存FAISS数据失败: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        try:
            if self.store_type == 'chroma':
                return self._search_chroma(query_embedding, top_k)
            elif self.store_type == 'faiss':
                return self._search_faiss(query_embedding, top_k)
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def _search_chroma(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """在ChromaDB中搜索"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i],  # 转换为相似度分数
                'id': results['ids'][0][i]
            })
        
        return search_results
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """在FAISS中搜索"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # 搜索
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        search_results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.documents):
                search_results.append({
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                    'score': float(score),
                    'id': str(idx)
                })
        
        return search_results
    
    def get_document_count(self) -> int:
        """获取文档数量"""
        try:
            if self.store_type == 'chroma':
                return self.collection.count()
            elif self.store_type == 'faiss':
                return len(self.documents)
        except Exception as e:
            logger.error(f"获取文档数量失败: {e}")
            return 0
    
    def clear(self):
        """清空向量数据库"""
        try:
            if self.store_type == 'chroma':
                self.client.delete_collection(self.config.get('collection_name', 'documents'))
                self.collection = self.client.create_collection(self.config.get('collection_name', 'documents'))
            elif self.store_type == 'faiss':
                self.index = None
                self.documents = []
                self.metadata = []
                if os.path.exists(self.index_path):
                    os.remove(self.index_path)
                if os.path.exists(self.metadata_path):
                    os.remove(self.metadata_path)
            
            logger.info("向量数据库已清空")
            
        except Exception as e:
            logger.error(f"清空向量数据库失败: {e}")
            raise