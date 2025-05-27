"""
大语言模型服务模块
支持多种开源LLM模型的本地部署
"""

import os
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers未安装，请运行: pip install transformers")


class LLMService:
    """大语言模型服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name', 'chatglm3-6b')
        self.model_path = config.get('model_path', './models/llm')
        self.device = config.get('device', 'cpu')
        self.max_length = config.get('max_length', 2048)
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 0.9)
        
        self.model = None
        self.tokenizer = None
        
        self._setup_device()
        self._load_model()
    
    def _setup_device(self):
        """设置计算设备"""
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch, 'npu') and torch.npu.is_available():
                self.device = 'npu'
            else:
                self.device = 'cpu'
        
        logger.info(f"LLM使用设备: {self.device}")
    
    def _load_model(self):
        """加载大语言模型"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("transformers未安装，使用模拟LLM服务")
                return
            
            # 检查本地模型路径
            local_model_path = os.path.join(self.model_path, self.model_name)
            
            if os.path.exists(local_model_path):
                logger.info(f"从本地加载LLM模型: {local_model_path}")
                model_path = local_model_path
            else:
                logger.info(f"模型路径不存在: {local_model_path}")
                logger.info("使用模拟LLM服务，请下载模型后重新启动")
                return
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 加载模型
            if self.device == 'cpu':
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map='auto'
                )
            
            # 设置为评估模式
            self.model.eval()
            
            logger.info(f"LLM模型加载成功: {self.model_name}")
            
        except Exception as e:
            logger.error(f"加载LLM模型失败: {e}")
            logger.info("将使用模拟LLM服务")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本回复"""
        try:
            if self.model is None or self.tokenizer is None:
                return self._mock_generate(prompt)
            
            # 合并配置参数
            generation_kwargs = {
                'max_length': kwargs.get('max_length', self.max_length),
                'temperature': kwargs.get('temperature', self.temperature),
                'top_p': kwargs.get('top_p', self.top_p),
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id
            }
            
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            if self.device != 'cpu':
                inputs = inputs.to(self.device)
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    **generation_kwargs
                )
            
            # 解码输出
            response = self.tokenizer.decode(
                outputs[0][len(inputs[0]):],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"生成文本失败: {e}")
            return self._mock_generate(prompt)
    
    def _mock_generate(self, prompt: str) -> str:
        """模拟LLM生成（当模型未加载时使用）"""
        logger.warning("使用模拟LLM服务")
        
        # 简单的模拟回复
        if "什么" in prompt or "是什么" in prompt:
            return "根据提供的上下文信息，我无法给出准确的回答。请确保已正确加载LLM模型。"
        elif "如何" in prompt or "怎么" in prompt:
            return "根据上下文信息，建议您参考相关文档或咨询专业人士。请注意，当前使用的是模拟LLM服务。"
        else:
            return "感谢您的问题。由于LLM模型未正确加载，我无法提供详细回答。请检查模型配置并重新启动服务。"
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """对话模式生成"""
        try:
            if self.model is None or self.tokenizer is None:
                return self._mock_generate(str(messages))
            
            # 构建对话提示词
            prompt = self._build_chat_prompt(messages)
            return self.generate(prompt)
            
        except Exception as e:
            logger.error(f"对话生成失败: {e}")
            return self._mock_generate(str(messages))
    
    def _build_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """构建对话提示词"""
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"系统: {content}")
            elif role == 'user':
                prompt_parts.append(f"用户: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"助手: {content}")
        
        prompt_parts.append("助手: ")
        return "\n".join(prompt_parts)
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "is_loaded": self.is_model_loaded()
        }
    
    def unload_model(self):
        """卸载模型以释放内存"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("LLM模型已卸载")
            
        except Exception as e:
            logger.error(f"卸载模型失败: {e}")
    
    def reload_model(self):
        """重新加载模型"""
        try:
            self.unload_model()
            self._load_model()
            logger.info("LLM模型已重新加载")
        except Exception as e:
            logger.error(f"重新加载模型失败: {e}")