# AI接口配置
AI_API_CONFIG = {
    'api_key': '<kimi_api_key>',
    'api_url': 'https://api.moonshot.cn/v1',
    'model': 'moonshot-v1-auto',
    'temperature': 0.3
}

# WQB登录配置
WQB_CONFIG = {
    'email': '1064223436@qq.com',
    'password': 'Ljw040309'
}

# 因子挖掘配置
FACTOR_MINING_CONFIG = {
    'max_iterations': 10,  # 每个数据集的迭代次数
    'min_sharpe': 1.5,    # 最小夏普比率要求
    'max_factors': 100,   # 最大因子数量
    'save_interval': 10,  # 保存间隔（因子数量）
}

# 数据集选择配置
DATASET_SELECTION_CONFIG = {
    'min_coverage': 0.7,  # 最小覆盖率
    'min_user_count': 100,  # 最小用户数
    'min_alpha_count': 1000,  # 最小alpha数量
}

# 因子生成配置
FACTOR_GENERATION_CONFIG = {
    'max_expression_length': 100,  # 最大表达式长度
    'max_operators': 5,  # 最大运算符数量
    'preferred_operators': [  # 优先使用的运算符
        'ts_mean',
        'ts_std_dev',
        'ts_corr',
        'ts_covariance',
        'ts_delay',
        'ts_delta',
        'ts_rank',
        'ts_quantile',
        'ts_zscore',
        'ts_scale'
    ]
} 