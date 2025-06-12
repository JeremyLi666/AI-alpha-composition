import asyncio
import json
import os
import traceback
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import wqb
from wqb import WQBSession, FilterRange
from ai_client import AIClient
from config_local import FACTOR_MINING_CONFIG, DATASET_SELECTION_CONFIG, FACTOR_GENERATION_CONFIG, WQB_CONFIG

class ExtendedWQBSession(WQBSession):
    """扩展的WQBSession类，添加了获取数据字段的功能"""
    
    def get_datafields(self, dataset_id: str) -> pd.DataFrame:
        """获取指定数据集的所有字段信息
        
        Args:
            dataset_id: 数据集ID，例如 'fundamental6'
            
        Returns:
            pd.DataFrame: 包含字段信息的DataFrame
        """
        # 构建基础URL和参数
        base_url = "https://api.worldquantbrain.com/data-fields"
        params = {
            'dataset.id': dataset_id,
            'delay': 1,
            'instrumentType': 'EQUITY',
            'limit': 50,  # 每页获取50条记录
            'offset': 0,
            'region': 'USA',
            'universe': 'TOP3000'
        }
        
        # 获取第一页以确定总数
        first_page = self.get(base_url, params=params)
        if not first_page.ok:
            self._log_error(f"获取数据集字段失败: {first_page.text}")
            return pd.DataFrame()
            
        total_count = first_page.json()['count']
        all_results = first_page.json()['results']
        
        # 获取剩余页面的数据
        for offset in range(50, total_count, 50):
            params['offset'] = offset
            page = self.get(base_url, params=params)
            if page.ok:
                all_results.extend(page.json()['results'])
            else:
                self._log_error(f"获取数据集字段分页失败: {page.text}")
                break
        
        # 转换为DataFrame
        return pd.DataFrame(all_results).loc[:,['id','description','type','coverage','userCount','alphaCount']].to_json(orient='records')
    
    def _log_error(self, message: str):
        """记录错误信息"""
        if hasattr(self, 'logger') and self.logger:
            self.logger.error(message)

class FactorMiningSystem:
    def __init__(self, wqbs: ExtendedWQBSession, logger=None):
        self.wqbs = wqbs
        self.logger = logger
        self.datasets_info = {}
        self.operators_info = []
        self.mined_factors = []
        self.current_dataset = None
        self.ai_client = AIClient()
        
    def _log_error(self, error_msg: str, error: Exception = None, context: Dict = None):
        """统一的错误日志记录方法"""
        if not self.logger:
            return
            
        log_parts = [f"错误: {error_msg}"]
        
        if error:
            log_parts.append(f"异常类型: {type(error).__name__}")
            log_parts.append(f"异常信息: {str(error)}")
            log_parts.append(f"堆栈跟踪:\n{traceback.format_exc()}")
            
        if context:
            log_parts.append(f"上下文信息: {json.dumps(context, indent=2, ensure_ascii=False)}")
            
        self.logger.error("\n".join(log_parts))
        
    def _log_warning(self, warning_msg: str, context: Dict = None):
        """统一的警告日志记录方法"""
        if not self.logger:
            return
            
        log_parts = [f"警告: {warning_msg}"]
        
        if context:
            log_parts.append(f"上下文信息: {json.dumps(context, indent=2, ensure_ascii=False)}")
            
        self.logger.warning("\n".join(log_parts))
        
    def _log_info(self, info_msg: str, context: Dict = None):
        """统一的信息日志记录方法"""
        if not self.logger:
            return
            
        log_parts = [f"信息: {info_msg}"]
        
        if context:
            log_parts.append(f"上下文信息: {json.dumps(context, indent=2, ensure_ascii=False)}")
            
        self.logger.info("\n".join(log_parts))
        
    async def initialize(self):
        """初始化系统，获取数据集和运算符信息"""
        # 获取数据集信息
        region = 'USA'
        delay = 1
        universe = 'TOP3000'
        resp = self.wqbs.search_datasets_limited(region, delay, universe)
        self.datasets_info = resp.json()
        
        # 获取运算符信息
        resp = self.wqbs.search_operators()
        self.operators_info = [item['name'] for item in resp.json()]
        
        if self.logger:
            self.logger.info(f"系统初始化完成，获取到 {len(self.datasets_info['results'])} 个数据集和 {len(self.operators_info)} 个运算符")
    
    async def select_dataset(self) -> Dict[str, Any]:
        """选择合适的数据集进行因子挖掘"""
        # 使用AI选择数据集
        selection = await self.ai_client.select_dataset(self.datasets_info)
        selected_id = selection['selected_dataset']
        
        # 找到选中的数据集
        for dataset in self.datasets_info['results']:
            if dataset['id'] == selected_id:
                self.current_dataset = dataset
                if self.logger:
                    self.logger.info(f"AI选择数据集: {dataset['name']}, 原因: {selection['reason']}")
                return dataset
        
        # 如果没有找到选中的数据集，返回第一个数据集
        self.current_dataset = self.datasets_info['results'][0]
        return self.current_dataset
    
    async def generate_initial_factor(self) -> str:
        """生成初始因子表达式"""
        # 获取数据集字段信息
        fields_json = self.wqbs.get_datafields(self.current_dataset['id'])
        if not fields_json:
            self._log_error("获取数据集字段失败", context={
                "dataset_id": self.current_dataset['id']
            })
            return ""
            
        # 将字段信息传递给AI
        return await self.ai_client.generate_initial_factor(
            self.current_dataset,
            self.operators_info,
            fields_json
        )
    
    async def simulate_factor(self, factor_expr: str) -> Dict[str, Any]:
        """模拟因子表现"""
        # 获取数据集中的字段名列表
        try:
            # 使用get_datafields获取字段信息
            fields_json = self.wqbs.get_datafields(self.current_dataset['id'])
            if not fields_json:
                self._log_error("获取数据集字段失败", context={
                    "dataset_id": self.current_dataset['id']
                })
                return {}
                
            # 解析字段信息
            fields_data = json.loads(fields_json)
            self._log_info("获取到数据集字段", {
                "dataset_id": self.current_dataset['id'],
                "total_fields": len(fields_data)
            })
            
            # 构建字段映射
            field_map = {field['id'].lower(): field for field in fields_data}
            
            # 检查并转换字段名大小写
            for field_id, field_info in field_map.items():
                if field_id in factor_expr.lower():
                    factor_expr = factor_expr.replace(field_id, field_info['id'])
            
            alpha = {
                'type': 'REGULAR',
                'settings': {
                    'instrumentType': 'EQUITY',
                    'region': 'USA',
                    'universe': 'TOP3000',
                    'delay': 1,
                    'decay': 13,
                    'neutralization': 'INDUSTRY',
                    'truncation': 0.13,
                    'pasteurization': 'ON',
                    'unitHandling': 'VERIFY',
                    'nanHandling': 'OFF',
                    'language': 'FASTEXPR',
                    'visualization': False
                },
                'regular': factor_expr,
            }
            
            self._log_info("开始模拟因子", {
                "factor_expression": factor_expr,
                "alpha_settings": alpha
            })
            
            resp = await self.wqbs.simulate(alpha)
            
            if not resp.ok:
                self._log_error("因子模拟失败", context={
                    "status_code": resp.status_code,
                    "response_text": resp.text,
                    "factor_expression": factor_expr
                })
                return {}
            
            result = resp.json()
            
            self._log_info("模拟响应", {"response": result})
            
            # 验证响应格式
            if not isinstance(result, dict):
                self._log_error("响应格式错误", context={
                    "expected_type": "dict",
                    "actual_type": str(type(result)),
                    "response": result
                })
                return {}
            
            # 检查必要字段
            if 'alpha' not in result:
                self._log_error("响应中缺少必要字段", context={
                    "missing_field": "alpha",
                    "response": result
                })
                return {}
            
            self._log_info("因子模拟成功", {"alpha_id": result['alpha']})
            return result
            
        except json.JSONDecodeError as e:
            self._log_error("JSON解析失败", error=e, context={
                "response_text": resp.text if 'resp' in locals() else 'No response',
                "factor_expression": factor_expr
            })
            return {}
        except Exception as e:
            self._log_error("因子模拟过程中发生错误", error=e, context={
                "factor_expression": factor_expr,
                "alpha_settings": alpha if 'alpha' in locals() else None
            })
            return {}
    
    async def check_factor(self, alpha_id: str) -> Dict[str, Any]:
        """检查因子表现"""
        try:
            resp = await self.wqbs.check(alpha_id)
            if not resp.ok:
                if self.logger:
                    self.logger.error(f"因子检查失败: {resp.text}")
                return {}
            
            result = resp.json()
            if self.logger:
                self.logger.info(f"因子检查成功: {result.get('status', '')}")
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"因子检查过程中发生错误: {str(e)}")
            return {}
    
    async def generate_next_factor(self, previous_factors: List[Dict[str, Any]]) -> str:
        """根据之前的因子表现生成下一个因子表达式"""
        # 获取数据集字段信息
        fields_json = self.wqbs.get_datafields(self.current_dataset['id'])
        if not fields_json:
            self._log_error("获取数据集字段失败", context={
                "dataset_id": self.current_dataset['id']
            })
            return ""
            
        # 将字段信息传递给AI
        return await self.ai_client.generate_next_factor(
            self.current_dataset,
            self.operators_info,
            previous_factors,
            fields_json
        )
    
    def save_factors(self):
        """保存已挖掘的因子"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'factors_{timestamp}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.mined_factors, f, indent=2, ensure_ascii=False)
        
        if self.logger:
            self.logger.info(f"因子已保存到文件: {filename}")
    
    async def mine_factors(self, max_iterations: int = None, min_sharpe: float = None):
        """主循环：挖掘因子"""
        if max_iterations is None:
            max_iterations = FACTOR_MINING_CONFIG['max_iterations']
        if min_sharpe is None:
            min_sharpe = FACTOR_MINING_CONFIG['min_sharpe']
        
        await self.initialize()
        
        while len(self.mined_factors) < FACTOR_MINING_CONFIG['max_factors']:
            try:
                # 选择数据集
                dataset = await self.select_dataset()
                
                # 生成初始因子
                factor_expr = await self.generate_initial_factor()
                if not factor_expr:
                    self._log_warning("生成初始因子失败，跳过当前迭代")
                    continue
                    
                self._log_info("生成初始因子表达式", {"expression": factor_expr})
                
                # 模拟因子
                sim_result = await self.simulate_factor(factor_expr)
                if not sim_result or 'alpha' not in sim_result:
                    self._log_warning("因子模拟失败，跳过当前迭代", context={
                        "factor_expression": factor_expr,
                        "simulation_result": sim_result
                    })
                    continue
                    
                alpha_id = sim_result['alpha']
                
                # 检查因子
                check_result = await self.check_factor(alpha_id)
                if not check_result:
                    self._log_warning("因子检查失败，跳过当前迭代", context={
                        "alpha_id": alpha_id,
                        "check_result": check_result
                    })
                    continue
                
                # 保存因子信息
                factor_info = {
                    'expression': factor_expr,
                    'simulation': sim_result,
                    'check': check_result,
                    'dataset': dataset
                }
                self.mined_factors.append(factor_info)
                
                # 迭代优化因子
                iteration = 0
                while iteration < max_iterations:
                    try:
                        # 生成下一个因子
                        next_factor = await self.generate_next_factor(self.mined_factors)
                        if not next_factor:
                            self._log_warning("生成下一个因子失败，结束当前迭代")
                            break
                            
                        self._log_info("生成下一个因子表达式", {"expression": next_factor})
                        
                        # 模拟和检查新因子
                        sim_result = await self.simulate_factor(next_factor)
                        if not sim_result or 'alpha' not in sim_result:
                            self._log_warning("新因子模拟失败，结束当前迭代", context={
                                "factor_expression": next_factor,
                                "simulation_result": sim_result
                            })
                            break
                            
                        alpha_id = sim_result['alpha']
                        check_result = await self.check_factor(alpha_id)
                        if not check_result:
                            self._log_warning("新因子检查失败，结束当前迭代", context={
                                "alpha_id": alpha_id,
                                "check_result": check_result
                            })
                            break
                        
                        # 保存新因子信息
                        factor_info = {
                            'expression': next_factor,
                            'simulation': sim_result,
                            'check': check_result,
                            'dataset': dataset
                        }
                        self.mined_factors.append(factor_info)
                        
                        # 检查是否达到目标
                        if check_result.get('is', {}).get('checks', [{}])[0].get('value', 0) >= min_sharpe:
                            self._log_info("找到满足条件的因子", {
                                "expression": next_factor,
                                "sharpe": check_result.get('is', {}).get('checks', [{}])[0].get('value', 0)
                            })
                            break
                        
                        iteration += 1
                        
                    except Exception as e:
                        self._log_error("因子优化迭代过程中发生错误", error=e, context={
                            "iteration": iteration,
                            "max_iterations": max_iterations,
                            "current_factor_count": len(self.mined_factors)
                        })
                        break
                
                # 定期保存因子
                if len(self.mined_factors) % FACTOR_MINING_CONFIG['save_interval'] == 0:
                    self.save_factors()
                
                # 如果已经达到最大因子数量，退出循环
                if len(self.mined_factors) >= FACTOR_MINING_CONFIG['max_factors']:
                    break
                    
            except Exception as e:
                self._log_error("因子挖掘主循环中发生错误", error=e, context={
                    "current_factor_count": len(self.mined_factors),
                    "max_factors": FACTOR_MINING_CONFIG['max_factors']
                })
                break
        
        # 最后保存一次因子
        self.save_factors()

async def main():
    # 创建logger
    logger = wqb.wqb_logger()
    
    # 创建ExtendedWQBSession
    wqbs = ExtendedWQBSession(
        (WQB_CONFIG['email'], WQB_CONFIG['password']), 
        logger=logger
    )
    
    # 创建因子挖掘系统
    mining_system = FactorMiningSystem(wqbs, logger)
    
    try:
        # 开始挖掘因子
        await mining_system.mine_factors()
    except Exception as e:
        if logger:
            logger.error(f"因子挖掘过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序发生错误: {str(e)}") 