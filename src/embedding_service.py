"""
嵌入服务模块
支持sentence-transformers和其他嵌入模型
"""

import os
import numpy as np
from typing import List, Union
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers未安装，请运行: pip install sentence-transformers")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch未安装")


class EmbeddingService:
    """嵌入服务类"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get('model_name', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.model_path = config.get('model_path', './models/embeddings')
        self.device = config.get('device', 'cpu')
        self.batch_size = config.get('batch_size', 32)
        self.max_length = config.get('max_length', 512)
        
        self._setup_device()
        self._load_model()
    
    def _setup_device(self):
        """设置计算设备"""
        if self.device == 'auto':
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    self.device = 'cuda'
                elif hasattr(torch, 'npu') and torch.npu.is_available():
                    self.device = 'npu'
                else:
                    self.device = 'cpu'
            else:
                self.device = 'cpu'
        
        logger.info(f"使用设备: {self.device}")
    
    def _load_model(self):
        """加载嵌入模型"""
        try:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers未安装")
            
            # 检查本地模型路径
            local_model_path = os.path.join(self.model_path, self.model_name.split('/')[-1])
            
            if os.path.exists(local_model_path):
                logger.info(f"从本地加载模型: {local_model_path}")
                self.model = SentenceTransformer(local_model_path, device=self.device)
            else:
                logger.info(f"从Hugging Face加载模型: {self.model_name}")
                self.model = SentenceTransformer(self.model_name, device=self.device)
                
                # 保存模型到本地
                os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
                self.model.save(local_model_path)
                logger.info(f"模型已保存到: {local_model_path}")
            
            # 设置模型参数
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.max_length
            
            logger.info(f"嵌入模型加载成功: {self.model_name}")
            
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True) -> np.ndarray:
        """编码文本为嵌入向量"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            # 批量编码
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=len(texts) > 100
            )
            
            logger.debug(f"成功编码 {len(texts)} 个文本，嵌入维度: {embeddings.shape[1]}")
            return embeddings
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"获取嵌入维度失败: {e}")
            return 768  # 默认维度
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """计算嵌入向量之间的相似度"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(embeddings1, embeddings2)
        except ImportError:
            # 如果sklearn不可用，使用numpy计算余弦相似度
            return self._cosine_similarity_numpy(embeddings1, embeddings2)
    
    def _cosine_similarity_numpy(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """使用numpy计算余弦相似度"""
        # 归一化向量
        embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # 计算点积
        return np.dot(embeddings1_norm, embeddings2_norm.T)
    
    def encode_query(self, query: str) -> np.ndarray:
        """专门用于编码查询的方法"""
        return self.encode([query])[0]
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """专门用于编码文档的方法"""
        return self.encode(documents)
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.get_embedding_dimension(),
            "max_length": self.max_length,
            "batch_size": self.batch_size
        }