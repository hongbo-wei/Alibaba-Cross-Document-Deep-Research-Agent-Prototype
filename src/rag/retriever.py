from typing import List, Dict, Any, Optional
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from loguru import logger

class DocumentRetriever:
    """文档检索器实现"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        index_type: str = "Flat",
        embedding_dim: int = 1024,
    ):
        """
        初始化文档检索器
        
        Args:
            model_name: 使用的嵌入模型名称
            device: 运行设备
            index_type: FAISS索引类型
            embedding_dim: 嵌入向量维度
        """
        self.device = device
        self.embedding_dim = embedding_dim
        
        # 加载嵌入模型
        logger.info(f"Loading embedding model {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        
        # 初始化FAISS索引
        self.index = self._initialize_index(index_type)
        self.documents: List[Dict[str, Any]] = []
        
    def _initialize_index(self, index_type: str) -> faiss.Index:
        """初始化FAISS索引"""
        if index_type == "Flat":
            return faiss.IndexFlatL2(self.embedding_dim)
        elif index_type == "IVF":
            nlist = 100  # 聚类中心数量
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            return faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        添加文档到检索系统
        
        Args:
            documents: 文档列表，每个文档应包含text字段
        """
        if not documents:
            return
        
        # 提取文本
        texts = [doc["text"] for doc in documents]
        
        # 生成嵌入
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True
        ).cpu().numpy()
        
        # 添加到索引
        if self.index.ntotal == 0:
            self.index.add(embeddings)
        else:
            self.index.add(embeddings)
        
        # 保存文档
        self.documents.extend(documents)
        
        logger.info(f"Added {len(documents)} documents to the index")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            
        Returns:
            相关文档列表
        """
        # 生成查询嵌入
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        ).cpu().numpy().reshape(1, -1)
        
        # 搜索最相似的文档
        scores, indices = self.index.search(query_embedding, top_k)
        
        # 过滤和格式化结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and score <= score_threshold:
                doc = self.documents[idx].copy()
                doc["score"] = float(score)
                results.append(doc)
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> List[List[Dict[str, Any]]]:
        """
        批量搜索相关文档
        
        Args:
            queries: 查询文本列表
            top_k: 每个查询返回结果数量
            score_threshold: 相似度阈值
            
        Returns:
            每个查询的相关文档列表
        """
        # 生成查询嵌入
        query_embeddings = self.model.encode(
            queries,
            convert_to_tensor=True,
            show_progress_bar=True
        ).cpu().numpy()
        
        # 批量搜索
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # 处理结果
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx < len(self.documents) and score <= score_threshold:
                    doc = self.documents[idx].copy()
                    doc["score"] = float(score)
                    results.append(doc)
            all_results.append(results)
        
        return all_results
    
    def update_document(self, doc_id: str, new_text: str):
        """
        更新文档内容
        
        Args:
            doc_id: 文档ID
            new_text: 新的文档内容
        """
        # 查找文档
        for i, doc in enumerate(self.documents):
            if doc.get("id") == doc_id:
                # 生成新嵌入
                new_embedding = self.model.encode(
                    new_text,
                    convert_to_tensor=True
                ).cpu().numpy().reshape(1, -1)
                
                # 更新索引
                self.index.remove_ids(np.array([i]))
                self.index.add(new_embedding)
                
                # 更新文档
                self.documents[i]["text"] = new_text
                logger.info(f"Updated document {doc_id}")
                return
        
        logger.warning(f"Document {doc_id} not found")
    
    def delete_document(self, doc_id: str):
        """
        删除文档
        
        Args:
            doc_id: 文档ID
        """
        # 查找文档
        for i, doc in enumerate(self.documents):
            if doc.get("id") == doc_id:
                # 从索引中删除
                self.index.remove_ids(np.array([i]))
                
                # 从文档列表中删除
                self.documents.pop(i)
                logger.info(f"Deleted document {doc_id}")
                return
        
        logger.warning(f"Document {doc_id} not found")
    
    def save_index(self, path: str):
        """
        保存索引到文件
        
        Args:
            path: 保存路径
        """
        faiss.write_index(self.index, path)
        logger.info(f"Saved index to {path}")
    
    def load_index(self, path: str):
        """
        从文件加载索引
        
        Args:
            path: 索引文件路径
        """
        self.index = faiss.read_index(path)
        logger.info(f"Loaded index from {path}") 