from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

class ResponseGenerator:
    """基于检索文档的响应生成器"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-7B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 2048,
        temperature: float = 0.7,
    ):
        """
        初始化响应生成器
        
        Args:
            model_name: 使用的模型名称
            device: 运行设备
            max_length: 最大生成长度
            temperature: 生成温度
        """
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        
        # 加载模型和分词器
        logger.info(f"Loading generator model {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    def _format_prompt(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        格式化提示词
        
        Args:
            query: 用户查询
            documents: 检索到的文档
            system_prompt: 系统提示词
            
        Returns:
            格式化后的提示词
        """
        if system_prompt is None:
            system_prompt = """你是一个专业的跨文档研究助手。你的任务是基于提供的参考文档，生成准确、全面且结构化的回答。
请遵循以下原则：
1. 只使用提供的文档中的信息
2. 如果文档中没有相关信息，请明确说明
3. 保持回答的客观性和准确性
4. 使用清晰的逻辑结构组织信息
5. 必要时引用具体文档来源"""
        
        # 构建文档上下文
        context = "\n\n".join([
            f"文档 {i+1} (相关度: {doc.get('score', 'N/A')}):\n{doc['text']}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""{system_prompt}

参考文档:
{context}

用户查询: {query}

请基于上述文档生成回答。回答应该：
1. 直接回应用户查询
2. 包含关键信息和支持证据
3. 保持逻辑性和连贯性
4. 必要时引用具体文档

回答："""
        
        return prompt
    
    def generate(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成回答
        
        Args:
            query: 用户查询
            documents: 检索到的文档
            system_prompt: 系统提示词
            **kwargs: 其他生成参数
            
        Returns:
            生成结果
        """
        # 格式化提示词
        prompt = self._format_prompt(query, documents, system_prompt)
        
        # 生成回答
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 设置生成参数
        gen_kwargs = {
            "max_new_tokens": self.max_length,
            "temperature": self.temperature,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            **kwargs
        }
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的回答部分
        response = response[len(prompt):].strip()
        
        return {
            "response": response,
            "prompt": prompt,
            "documents": documents
        }
    
    def batch_generate(
        self,
        queries: List[str],
        documents_list: List[List[Dict[str, Any]]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        批量生成回答
        
        Args:
            queries: 用户查询列表
            documents_list: 每个查询对应的文档列表
            system_prompt: 系统提示词
            **kwargs: 其他生成参数
            
        Returns:
            生成结果列表
        """
        results = []
        for query, documents in zip(queries, documents_list):
            result = self.generate(
                query=query,
                documents=documents,
                system_prompt=system_prompt,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def generate_with_citations(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成带引用的回答
        
        Args:
            query: 用户查询
            documents: 检索到的文档
            system_prompt: 系统提示词
            **kwargs: 其他生成参数
            
        Returns:
            生成结果，包含引用信息
        """
        if system_prompt is None:
            system_prompt = """你是一个专业的跨文档研究助手。你的任务是基于提供的参考文档，生成准确、全面且结构化的回答。
请遵循以下原则：
1. 只使用提供的文档中的信息
2. 如果文档中没有相关信息，请明确说明
3. 保持回答的客观性和准确性
4. 使用清晰的逻辑结构组织信息
5. 必须为每个关键信息添加引用，格式为[文档编号]
6. 在回答末尾列出所有引用的文档"""
        
        # 生成基本回答
        result = self.generate(
            query=query,
            documents=documents,
            system_prompt=system_prompt,
            **kwargs
        )
        
        # 提取引用信息
        citations = []
        for i, doc in enumerate(documents):
            if f"[{i+1}]" in result["response"]:
                citations.append({
                    "document_id": i + 1,
                    "text": doc["text"],
                    "score": doc.get("score", "N/A")
                })
        
        result["citations"] = citations
        return result
    
    def evaluate_response(
        self,
        query: str,
        response: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        评估生成回答的质量
        
        Args:
            query: 用户查询
            response: 生成的回答
            documents: 检索到的文档
            
        Returns:
            评估指标
        """
        # TODO: 实现更复杂的评估逻辑
        # 这里仅作为示例，实际应用中应该使用更全面的评估方法
        
        # 计算回答长度
        response_length = len(response)
        
        # 计算引用数量
        citation_count = sum(1 for doc in documents if f"[{doc.get('id', '')}]" in response)
        
        # 计算文档覆盖率
        doc_coverage = len(set(
            doc.get("id", "") for doc in documents
            if f"[{doc.get('id', '')}]" in response
        )) / len(documents) if documents else 0
        
        return {
            "response_length": response_length,
            "citation_count": citation_count,
            "document_coverage": doc_coverage
        } 