from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
import json
import re

@dataclass
class AgentState:
    """Agent的状态信息"""
    current_step: int
    reasoning_chain: List[str]
    tool_calls: List[Dict[str, Any]]
    context: Dict[str, Any]

class ResearchAgent:
    """跨文档深度研究Agent的核心实现"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-7B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_steps: int = 10,
        temperature: float = 0.7,
    ):
        """
        初始化Research Agent
        
        Args:
            model_name: 使用的模型名称
            device: 运行设备
            max_steps: 最大推理步数
            temperature: 生成温度
        """
        self.device = device
        self.max_steps = max_steps
        self.temperature = temperature
        
        # 加载模型和分词器
        logger.info(f"Loading model {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 初始化工具集
        self.tools = self._initialize_tools()
        
    def _initialize_tools(self) -> Dict[str, Any]:
        """初始化Agent可用的工具集"""
        return {
            "search": self._search_tool,
            "summarize": self._summarize_tool,
            "compare": self._compare_tool,
            "extract": self._extract_tool,
        }
    
    def _search_tool(self, query: str, **kwargs) -> Dict[str, Any]:
        """搜索工具实现 - 基于文档内容的语义搜索"""
        try:
            # 这里实现一个简单的基于关键词的搜索
            # 在实际应用中，可以使用更高级的向量搜索或外部API
            
            # 模拟搜索结果
            results = []
            
            # 如果有关键词匹配，返回相关文档片段
            if any(keyword in query.lower() for keyword in ['research', 'study', 'analysis']):
                results.append({
                    "title": "相关研究文档",
                    "content": "这是一个关于研究方法的文档片段...",
                    "relevance_score": 0.85
                })
            
            if any(keyword in query.lower() for keyword in ['data', 'information', 'facts']):
                results.append({
                    "title": "数据信息文档",
                    "content": "包含相关数据和分析信息的文档...",
                    "relevance_score": 0.78
                })
            
            # 如果没有找到相关内容，返回通用结果
            if not results:
                results.append({
                    "title": "通用信息",
                    "content": "基于查询内容，找到了一些相关信息...",
                    "relevance_score": 0.5
                })
            
            return {
                "status": "success", 
                "results": results,
                "query": query,
                "total_results": len(results)
            }
            
        except Exception as e:
            logger.error(f"搜索工具执行失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "results": []
            }
    
    def _summarize_tool(self, text: str, **kwargs) -> Dict[str, Any]:
        """总结工具实现 - 使用语言模型生成文本摘要"""
        try:
            # 获取摘要参数
            max_length = kwargs.get('max_length', 150)
            min_length = kwargs.get('min_length', 50)
            
            # 如果文本太短，直接返回
            if len(text.split()) < 10:
                return {
                    "status": "success",
                    "summary": text,
                    "original_length": len(text),
                    "summary_length": len(text)
                }
            
            # 构建摘要提示词
            prompt = f"""请为以下文本生成一个简洁的摘要，摘要长度在{min_length}-{max_length}字之间：

文本内容：
{text}

摘要："""
            
            # 使用模型生成摘要
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_new_tokens=min_length,
                temperature=0.3,  # 降低温度以获得更稳定的摘要
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # 解码摘要
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的摘要部分（去除原始提示词）
            summary = summary.replace(prompt, "").strip()
            
            # 如果摘要为空或太短，使用简单的文本截取
            if len(summary) < min_length:
                words = text.split()
                summary = " ".join(words[:max_length//5]) + "..."
            
            return {
                "status": "success",
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": round(len(summary) / len(text), 2)
            }
            
        except Exception as e:
            logger.error(f"总结工具执行失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "summary": "",
                "original_length": len(text),
                "summary_length": 0
            }
    
    def _compare_tool(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """比较工具实现"""
        # TODO: 实现实际的比较逻辑
        return {"status": "success", "comparison": ""}
    
    def _extract_tool(self, text: str, pattern: str, **kwargs) -> Dict[str, Any]:
        """信息提取工具实现"""
        # TODO: 实现实际的信息提取逻辑
        return {"status": "success", "extracted": []}
    
    def _format_prompt(self, query: str, state: AgentState) -> str:
        """格式化提示词"""
        prompt = f"""你是一个专业的跨文档深度研究Agent。你的任务是深入分析用户查询，并通过多步推理和工具使用来提供全面的答案。

当前步骤: {state.current_step}/{self.max_steps}
用户查询: {query}

已完成的推理步骤:
{chr(10).join(f"- {step}" for step in state.reasoning_chain)}

已使用的工具:
{chr(10).join(f"- {tool}" for tool in state.tool_calls)}

请进行下一步推理或工具调用。你可以:
1. 继续推理
2. 使用工具
3. 得出结论

你的回复应该是JSON格式:
{{
    "reasoning": "你的推理过程",
    "action": "推理/工具调用/结论",
    "tool_name": "如果要使用工具，指定工具名称",
    "tool_args": {{"arg1": "value1"}},
    "conclusion": "如果是结论，给出最终答案"
}}
"""
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析模型响应，提取JSON格式的回复"""
        # 尝试提取第一个JSON对象
        try:
            # 用正则找出第一个大括号包裹的内容
            match = re.search(r'\{[\s\S]*?\}', response)
            if match:
                json_str = match.group(0)
                parsed = json.loads(json_str)
                # 确保所有字段都存在
                return {
                    "reasoning": parsed.get("reasoning", ""),
                    "action": parsed.get("action", ""),
                    "tool_name": parsed.get("tool_name", ""),
                    "tool_args": parsed.get("tool_args", {}),
                    "conclusion": parsed.get("conclusion", "")
                }
            else:
                logger.error("未找到JSON对象: {}", response)
        except Exception as e:
            logger.error(f"解析模型响应失败: {e}; 响应内容: {response}")
        # 解析失败时返回默认结构
        return {
            "reasoning": "",
            "action": "",
            "tool_name": "",
            "tool_args": {},
            "conclusion": ""
        }
    
    def research(self, query: str) -> Dict[str, Any]:
        """
        执行深度研究任务
        
        Args:
            query: 用户查询
            
        Returns:
            研究结果
        """
        state = AgentState(
            current_step=0,
            reasoning_chain=[],
            tool_calls=[],
            context={}
        )
        
        while state.current_step < self.max_steps:
            # 准备提示词
            prompt = self._format_prompt(query, state)
            
            # 生成响应
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 解析响应
            parsed = self._parse_response(response)
            
            # 更新状态
            state.current_step += 1
            state.reasoning_chain.append(parsed["reasoning"])
            
            # 执行动作
            if parsed["action"] == "工具调用":
                tool = self.tools.get(parsed["tool_name"])
                if tool:
                    result = tool(**parsed["tool_args"])
                    state.tool_calls.append({
                        "tool": parsed["tool_name"],
                        "args": parsed["tool_args"],
                        "result": result
                    })
            elif parsed["action"] == "结论":
                return {
                    "status": "success",
                    "reasoning_chain": state.reasoning_chain,
                    "tool_calls": state.tool_calls,
                    "conclusion": parsed["conclusion"]
                }
        
        return {
            "status": "max_steps_reached",
            "reasoning_chain": state.reasoning_chain,
            "tool_calls": state.tool_calls,
            "conclusion": "达到最大步数限制"
        }
    
    def train(self, training_data: List[Dict[str, Any]], **kwargs):
        """
        训练Agent
        
        Args:
            training_data: 训练数据
            **kwargs: 训练参数
        """
        # TODO: 实现训练逻辑
        pass
    
    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估Agent性能
        
        Args:
            eval_data: 评估数据
            
        Returns:
            评估指标
        """
        # TODO: 实现评估逻辑
        return {
            "accuracy": 0.0,
            "reasoning_quality": 0.0,
            "tool_usage": 0.0
        } 