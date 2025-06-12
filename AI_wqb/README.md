# WorldQuant Brain 因子挖掘系统

这是一个基于 WorldQuant Brain 平台的自动化因子挖掘系统，使用 AI 技术来生成和优化因子表达式。

## 功能特点

- 自动选择合适的数据集
- 使用 AI 生成初始因子表达式
- 智能优化和迭代因子
- 自动评估因子表现
- 详细的日志记录
- 支持断点续传

## 安装步骤

1. 克隆仓库：
```bash
git clone [repository_url]
cd [repository_name]
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置设置：
   - 复制 `config.py` 并重命名为 `config_local.py`
   - 在 `config_local.py` 中填入你的配置信息：
     - WQB 登录信息
     - AI API 密钥
     - 其他参数设置

## 使用方法

1. 确保已正确配置 `config.py`

2. 运行程序：
```bash
python factor_mining.py
```

## 配置说明

在 `config_local.py` 中可以配置以下参数：

### WQB配置
```python
WQB_CONFIG = {
    'email': 'your_email@example.com',
    'password': 'your_password'
}
```

### AI API配置
```python
AI_API_CONFIG = {
    'api_key': 'your_api_key',
    'api_url': 'https://api.moonshot.cn/v1',
    'model': 'moonshot-v1-8k',
    'temperature': 0.7
}
```

### 因子挖掘配置
```python
FACTOR_MINING_CONFIG = {
    'max_iterations': 10,  # 每个因子的最大迭代次数
    'min_sharpe': 1.5,    # 最小夏普比率要求
    'max_factors': 100,   # 最大因子数量
    'save_interval': 10   # 每挖掘多少个因子保存一次
}
```

## 注意事项

1. 请确保你的 WQB 账号有足够的权限
2. 建议使用虚拟环境运行程序
3. 请妥善保管你的登录信息和 API 密钥
4. 程序运行过程中会自动保存因子，可以随时中断

## 日志说明

程序会生成详细的日志，包括：
- 因子生成过程
- 因子评估结果
- 错误和警告信息
- 系统状态信息

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进这个项目。

## 许可证

MIT License 