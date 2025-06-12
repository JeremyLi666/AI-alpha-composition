import json
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
from config_local import AI_API_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=AI_API_CONFIG['api_key'],
            base_url=AI_API_CONFIG['api_url']
        )
        self.model = AI_API_CONFIG['model']
        self.temperature = AI_API_CONFIG['temperature']
    
    async def _make_request(self, prompt: str) -> Dict[str, Any]:
        """发送请求到Kimi API并返回解析后的JSON响应"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "宝宝你是一个专业的量化投资专家，擅长因子挖掘和策略开发。你会为用户提供安全、有帮助、准确的回答。请始终以JSON格式返回数据。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            
            response_content = completion.choices[0].message.content
            
            try:
                # 尝试解析JSON响应
                parsed_response = json.loads(response_content)
                
                # 验证响应格式
                if not isinstance(parsed_response, dict):
                    logger.warning(f"API响应不是字典格式: {response_content}")
                    return self._get_default_response()
                
                return parsed_response
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {str(e)}, 原始响应: {response_content}")
                return self._get_default_response()
                
        except Exception as e:
            logger.error(f"Kimi API请求失败: {str(e)}")
            return self._get_default_response()
    
    def _get_default_response(self) -> Dict[str, Any]:
        """返回默认响应"""
        return {
            "text": "抱歉，我暂时无法处理您的请求。请稍后再试。",
            "image": "",
            "url": ""
        }
    
    async def select_dataset(self, datasets_info: Dict[str, Any]) -> Dict[str, Any]:
        """选择合适的数据集"""
        prompt = f"""请根据以下数据集信息，选择一个最适合进行因子挖掘的数据集。
数据集信息：
{json.dumps(datasets_info, indent=2, ensure_ascii=False)}

请考虑以下因素：
1. 数据集的覆盖率
2. 数据集的使用频率（userCount）
3. 数据集的alpha数量
4. 数据集的类别和子类别

请以以下JSON格式返回你的选择：
{{
    "selected_dataset": "数据集ID",
    "reason": "选择该数据集的原因",
    "analysis": {{
        "coverage": "覆盖率分析",
        "usage": "使用频率分析",
        "alpha_count": "alpha数量分析",
        "category": "类别分析"
    }}
}}
"""
        response = await self._make_request(prompt)
        if 'selected_dataset' in response and 'reason' in response:
            return response
        return {
            'selected_dataset': datasets_info['results'][0]['id'],
            'reason': '无法解析AI响应，使用第一个数据集作为默认选择',
            'analysis': {
                'coverage': '默认选择',
                'usage': '默认选择',
                'alpha_count': '默认选择',
                'category': '默认选择'
            }
        }
    
    async def generate_initial_factor(self, dataset_info: Dict[str, Any], operators: List[str], fields_json: str) -> str:
        """生成初始因子表达式"""
        # 解析字段信息
        fields_data = json.loads(fields_json)
        
        prompt = f"""请根据以下信息生成一个初始的因子表达式。

数据集信息：
{json.dumps(dataset_info, indent=2, ensure_ascii=False)}

可用字段信息（请使用字段的id作为字段名）：
{json.dumps(fields_data, indent=2, ensure_ascii=False)}

可用运算符：
{json.dumps(operators, indent=2, ensure_ascii=False)}

请以以下JSON格式返回你的因子表达式：
{{
    "factor_expression": "因子表达式",
    "explanation": "因子解释",
    "components": {{
        "fields": ["使用的字段id列表"],
        "operators": ["使用的运算符列表"]
    }},
    "expected_behavior": "预期的因子行为",
    "field_analysis": {{
        "selected_fields": [
            {{
                "id": "字段id",
                "description": "字段描述",
                "type": "字段类型",
                "coverage": "覆盖率",
                "usage": "使用频率"
            }}
        ],
        "selection_reason": "选择这些字段的原因"
    }}
}}

要求：
1. 使用字段的id作为字段名，不要使用字段的描述或其他属性
2. 使用提供的运算符，运算符名必须完全匹配
3. 具有合理的金融与经济学意义
4. 长度适中
5. 因子表达式必须使用 WorldQuant BRAIN 支持的语法
6. 不要使用 TS_VAL 函数，直接使用字段id
7. 确保所有括号都正确配对
8. 确保所有运算符和字段名之间都有正确的空格
9. 选择字段时考虑其覆盖率和使用频率

示例格式：
- ts_rank(divide(close, open), 20)
- ts_mean(volume, 5) / ts_std(volume, 5)
- ts_corr(close, volume, 10)
"""
        response = await self._make_request(prompt)
        if 'factor_expression' in response:
            return response['factor_expression']
        return ""
    
    async def generate_next_factor(self, 
                                 dataset_info: Dict[str, Any], 
                                 operators: List[str],
                                 previous_factors: List[Dict[str, Any]],
                                 fields_json: str) -> str:
        """根据之前的因子表现生成下一个因子表达式"""
        # 解析字段信息
        fields_data = json.loads(fields_json)
        
        prompt = f"""请根据以下信息生成下一个因子表达式。

数据集信息：
{json.dumps(dataset_info, indent=2, ensure_ascii=False)}

可用字段信息（请使用字段的id作为字段名）：
{json.dumps(fields_data, indent=2, ensure_ascii=False)}

可用运算符：
{json.dumps(operators, indent=2, ensure_ascii=False)}

之前的因子表现：
{json.dumps(previous_factors, indent=2, ensure_ascii=False)}

请以以下JSON格式返回你的因子表达式：
{{
    "factor_expression": "因子表达式",
    "explanation": "因子解释",
    "improvements": {{
        "previous_issues": "之前因子的不足",
        "solutions": "本因子的改进方案"
    }},
    "correlation_avoidance": "如何避免与之前因子高度相关",
    "components": {{
        "fields": ["使用的字段id列表"],
        "operators": ["使用的运算符列表"]
    }},
    "field_analysis": {{
        "selected_fields": [
            {{
                "id": "字段id",
                "description": "字段描述",
                "type": "字段类型",
                "coverage": "覆盖率",
                "usage": "使用频率"
            }}
        ],
        "selection_reason": "选择这些字段的原因",
        "improvement_over_previous": "相比之前因子使用的字段的改进"
    }}
}}

要求：
1. 避免与之前因子高度相关
2. 尝试改进之前因子的不足
3. 使用字段的id作为字段名，不要使用字段的描述或其他属性
4. 使用提供的运算符，运算符名必须完全匹配
5. 具有合理的金融意义
6. 因子表达式必须使用 WorldQuant BRAIN 支持的语法
7. 不要使用 TS_VAL 函数，直接使用字段id
8. 确保所有括号都正确配对
9. 确保所有运算符和字段名之间都有正确的空格
10. 选择字段时考虑其覆盖率和使用频率
11. 分析之前因子使用的字段，尝试使用不同的字段组合

示例格式：
- ts_rank(divide(close, open), 20)
- ts_mean(volume, 5) / ts_std(volume, 5)
- ts_corr(close, volume, 10)
"""
        response = await self._make_request(prompt)
        if 'factor_expression' in response:
            return response['factor_expression']
        return "" 