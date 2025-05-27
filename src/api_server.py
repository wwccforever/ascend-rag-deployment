"""
FastAPI服务器
提供RAG系统的REST API接口
"""

import os
import sys
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from loguru import logger

from rag_system import RAGSystem
from document_processor import DocumentProcessor


# Pydantic模型
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float


class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    status: str
    message: str
    stats: Optional[Dict[str, Any]] = None


# 初始化FastAPI应用
app = FastAPI(
    title="昇腾RAG系统API",
    description="基于昇腾服务器的检索增强生成系统",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
rag_system = None
document_processor = None


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化RAG系统"""
    global rag_system, document_processor
    
    try:
        logger.info("正在初始化RAG系统...")
        rag_system = RAGSystem()
        document_processor = DocumentProcessor()
        logger.info("RAG系统初始化完成")
    except Exception as e:
        logger.error(f"RAG系统初始化失败: {e}")
        raise


@app.get("/", response_model=StatusResponse)
async def root():
    """根路径，返回系统状态"""
    try:
        stats = rag_system.get_stats() if rag_system else {}
        return StatusResponse(
            status="success",
            message="昇腾RAG系统运行正常",
            stats=stats
        )
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """查询接口"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG系统未初始化")
        
        result = rag_system.query(request.question)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/add", response_model=StatusResponse)
async def add_document(request: DocumentRequest):
    """添加单个文档"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG系统未初始化")
        
        success = rag_system.add_documents(
            [request.content],
            [request.metadata] if request.metadata else None
        )
        
        if success:
            return StatusResponse(
                status="success",
                message="文档添加成功"
            )
        else:
            raise HTTPException(status_code=500, detail="文档添加失败")
            
    except Exception as e:
        logger.error(f"添加文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload", response_model=StatusResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档文件"""
    try:
        if not rag_system or not document_processor:
            raise HTTPException(status_code=500, detail="系统未初始化")
        
        # 检查文件类型
        if not document_processor.is_supported_format(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {file.filename}"
            )
        
        # 保存临时文件
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        try:
            # 处理文档
            documents = document_processor.process_file(temp_path)
            
            if documents:
                # 添加到RAG系统
                metadata = [{"filename": file.filename, "chunk_id": i} for i in range(len(documents))]
                success = rag_system.add_documents(documents, metadata)
                
                if success:
                    return StatusResponse(
                        status="success",
                        message=f"成功处理并添加 {len(documents)} 个文档块"
                    )
                else:
                    raise HTTPException(status_code=500, detail="文档添加失败")
            else:
                raise HTTPException(status_code=400, detail="文档处理失败或内容为空")
                
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"上传文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/stats", response_model=Dict[str, Any])
async def get_document_stats():
    """获取文档统计信息"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG系统未初始化")
        
        return rag_system.get_stats()
        
    except Exception as e:
        logger.error(f"获取文档统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=List[Dict[str, Any]])
async def search_documents(request: QueryRequest):
    """搜索文档（不生成答案）"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG系统未初始化")
        
        results = rag_system.search(request.question, request.top_k)
        return results
        
    except Exception as e:
        logger.error(f"搜索文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/clear", response_model=StatusResponse)
async def clear_documents():
    """清空所有文档"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG系统未初始化")
        
        rag_system.vector_store.clear()
        
        return StatusResponse(
            status="success",
            message="所有文档已清空"
        )
        
    except Exception as e:
        logger.error(f"清空文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}


if __name__ == "__main__":
    # 从配置文件读取服务器配置
    import yaml
    
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        host = config["server"]["api"]["host"]
        port = config["server"]["api"]["port"]
    except:
        host = "0.0.0.0"
        port = 8000
    
    logger.info(f"启动API服务器: http://{host}:{port}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )