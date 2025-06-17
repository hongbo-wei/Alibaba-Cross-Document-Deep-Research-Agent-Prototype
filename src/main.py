import argparse
from typing import List, Dict, Any
import json
from loguru import logger

from rag.retriever import DocumentRetriever
from rag.generator import ResponseGenerator
from agent.research_agent import ResearchAgent

def load_documents(file_path: str) -> List[Dict[str, Any]]:
    """加载示例文档"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='跨文档深度研究Agent演示')
    parser.add_argument('--mode', choices=['rag', 'agent'], default='rag',
                      help='运行模式: rag (仅RAG系统) 或 agent (完整Agent系统)')
    parser.add_argument('--query', type=str, required=True,
                      help='用户查询')
    parser.add_argument('--documents', type=str, required=True,
                      help='文档文件路径 (JSON格式)')
    parser.add_argument('--model', type=str, default='Qwen/Qwen-7B',
                      help='使用的模型名称')
    args = parser.parse_args()
    
    # 加载文档
    logger.info(f"Loading documents from {args.documents}")
    documents = load_documents(args.documents)
    
    if args.mode == 'rag':
        # 初始化RAG系统
        retriever = DocumentRetriever(model_name='BAAI/bge-large-zh')
        generator = ResponseGenerator(model_name=args.model)
        
        # 添加文档到检索系统
        retriever.add_documents(documents)
        
        # 检索相关文档
        logger.info(f"Searching for documents related to: {args.query}")
        retrieved_docs = retriever.search(args.query, top_k=3)
        
        # 生成回答
        logger.info("Generating response...")
        result = generator.generate_with_citations(
            query=args.query,
            documents=retrieved_docs
        )
        
        # 输出结果
        print("\n=== 检索到的文档 ===")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n文档 {i+1} (相关度: {doc.get('score', 'N/A')}):")
            print(doc['text'][:200] + "...")
        
        print("\n=== 生成的回答 ===")
        print(result['response'])
        
        print("\n=== 引用信息 ===")
        for citation in result['citations']:
            print(f"\n文档 {citation['document_id']}:")
            print(citation['text'][:100] + "...")
        
        # 评估回答质量
        metrics = generator.evaluate_response(
            args.query,
            result['response'],
            retrieved_docs
        )
        print("\n=== 评估指标 ===")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        
    else:  # agent mode
        # 初始化完整Agent系统
        agent = ResearchAgent(model_name=args.model)
        
        # 执行研究任务
        logger.info(f"Starting research task: {args.query}")
        result = agent.research(args.query)
        
        # 输出结果
        print("\n=== 推理过程 ===")
        for i, step in enumerate(result['reasoning_chain']):
            print(f"\n步骤 {i+1}:")
            print(step)
        
        print("\n=== 工具调用 ===")
        for i, tool_call in enumerate(result['tool_calls']):
            print(f"\n调用 {i+1}:")
            print(f"工具: {tool_call['tool']}")
            print(f"参数: {json.dumps(tool_call['args'], ensure_ascii=False)}")
            print(f"结果: {json.dumps(tool_call['result'], ensure_ascii=False)}")
        
        print("\n=== 最终结论 ===")
        print(result['conclusion'])

if __name__ == '__main__':
    main() 