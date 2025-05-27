"""
文档处理模块
支持多种文档格式的解析和文本提取
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import markdown
    from bs4 import BeautifulSoup
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.chunk_size = self.config.get('chunk_size', 500)
        self.chunk_overlap = self.config.get('chunk_overlap', 50)
        self.supported_formats = self.config.get('supported_formats', [
            '.txt', '.md', '.pdf', '.docx', '.xlsx', '.csv'
        ])
        
        logger.info(f"文档处理器初始化完成，支持格式: {self.supported_formats}")
    
    def is_supported_format(self, filename: str) -> bool:
        """检查文件格式是否支持"""
        if not filename:
            return False
        
        file_ext = Path(filename).suffix.lower()
        return file_ext in self.supported_formats
    
    def process_file(self, file_path: str) -> List[str]:
        """处理单个文件"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return []
            
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.txt':
                text = self._process_txt(file_path)
            elif file_ext == '.md':
                text = self._process_markdown(file_path)
            elif file_ext == '.pdf':
                text = self._process_pdf(file_path)
            elif file_ext == '.docx':
                text = self._process_docx(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                text = self._process_excel(file_path)
            elif file_ext == '.csv':
                text = self._process_csv(file_path)
            else:
                logger.warning(f"不支持的文件格式: {file_ext}")
                return []
            
            if not text:
                logger.warning(f"文件内容为空: {file_path}")
                return []
            
            # 分块处理
            chunks = self._split_text(text)
            logger.info(f"文件 {file_path} 处理完成，生成 {len(chunks)} 个文本块")
            
            return chunks
            
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            return []
    
    def _process_txt(self, file_path: str) -> str:
        """处理TXT文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            for encoding in ['gbk', 'gb2312', 'latin-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except:
                    continue
            logger.error(f"无法解码文件: {file_path}")
            return ""
    
    def _process_markdown(self, file_path: str) -> str:
        """处理Markdown文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            if MARKDOWN_AVAILABLE:
                # 转换为HTML然后提取文本
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
            else:
                # 简单处理，移除Markdown标记
                text = re.sub(r'#{1,6}\s+', '', md_content)  # 移除标题标记
                text = re.sub(r'\*\*(.*?)\*\*', r'\\1', text)  # 移除粗体标记
                text = re.sub(r'\*(.*?)\*', r'\\1', text)  # 移除斜体标记
                text = re.sub(r'`(.*?)`', r'\\1', text)  # 移除代码标记
                return text
                
        except Exception as e:
            logger.error(f"处理Markdown文件失败: {e}")
            return ""
    
    def _process_pdf(self, file_path: str) -> str:
        """处理PDF文件"""
        if not PDF_AVAILABLE:
            logger.error("PyPDF2未安装，无法处理PDF文件")
            return ""
        
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"处理PDF文件失败: {e}")
            return ""
    
    def _process_docx(self, file_path: str) -> str:
        """处理DOCX文件"""
        if not DOCX_AVAILABLE:
            logger.error("python-docx未安装，无法处理DOCX文件")
            return ""
        
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"处理DOCX文件失败: {e}")
            return ""
    
    def _process_excel(self, file_path: str) -> str:
        """处理Excel文件"""
        if not PANDAS_AVAILABLE:
            logger.error("pandas未安装，无法处理Excel文件")
            return ""
        
        try:
            # 读取所有工作表
            excel_file = pd.ExcelFile(file_path)
            text = ""
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # 将DataFrame转换为文本
                text += f"工作表: {sheet_name}\n"
                text += df.to_string(index=False) + "\n\n"
            
            return text
        except Exception as e:
            logger.error(f"处理Excel文件失败: {e}")
            return ""
    
    def _process_csv(self, file_path: str) -> str:
        """处理CSV文件"""
        if not PANDAS_AVAILABLE:
            logger.error("pandas未安装，无法处理CSV文件")
            return ""
        
        try:
            df = pd.read_csv(file_path)
            return df.to_string(index=False)
        except Exception as e:
            logger.error(f"处理CSV文件失败: {e}")
            return ""
    
    def _split_text(self, text: str) -> List[str]:
        """将文本分割成块"""
        if not text:
            return []
        
        # 清理文本
        text = self._clean_text(text)
        
        # 如果文本长度小于chunk_size，直接返回
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 如果不是最后一块，尝试在句号、换行符等处分割
            if end < len(text):
                # 寻找最近的句号或换行符
                for i in range(end, start + self.chunk_size // 2, -1):
                    if text[i] in '.。\n!！?？':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 计算下一个开始位置（考虑重叠）
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符（保留中文、英文、数字和基本标点）
        text = re.sub(r'[^\u4e00-\u9fff\w\s.,!?;:()\\[\\]{}\"\'\\-]', '', text)
        
        return text.strip()
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """处理目录中的所有支持的文件"""
        results = []
        
        try:
            directory = Path(directory_path)
            if not directory.exists():
                logger.error(f"目录不存在: {directory_path}")
                return results
            
            for file_path in directory.rglob('*'):
                if file_path.is_file() and self.is_supported_format(file_path.name):
                    chunks = self.process_file(str(file_path))
                    if chunks:
                        results.append({
                            'file_path': str(file_path),
                            'chunks': chunks,
                            'chunk_count': len(chunks)
                        })
            
            logger.info(f"目录处理完成，共处理 {len(results)} 个文件")
            return results
            
        except Exception as e:
            logger.error(f"处理目录失败: {e}")
            return results
    
    def get_text_stats(self, text: str) -> Dict[str, Any]:
        """获取文本统计信息"""
        stats = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'line_count': len(text.split('\n'))
        }
        
        if JIEBA_AVAILABLE:
            # 中文分词统计
            words = list(jieba.cut(text))
            stats['chinese_word_count'] = len(words)
        
        return stats