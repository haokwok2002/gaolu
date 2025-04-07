import pandas as pd
import pyodbc

# 数据库连接参数（替换为你的数据库配置）
server = '172.16.132.241'       # 数据库服务器名称或地址
database = 'ysdx'                # 数据库名称
username = 'ysdx'                # 数据库用户名
password = 'ysdx'                # 数据库密码

# 要查询的视图名称列表
view_names = [
    'v_5#_lqb_all',
    'v_5#_lqb',
    'v_5#_szcw',
    'v_5#_tscf_g',
    'v_5#_tscf_t',
    'v_T_SWC2',
    'v_T_SWC2_ll',
    'v_swc',
    'v_szcw'
]

# 查询时间点
start_time = '2024-12-12 20:00:00.000'

# 查询时间范围
start_time = '2025-02-01 20:00:00.000'  # 开始时间
end_time = '2025-03-01 20:00:00.000'    # 结束时间



# 建立数据库连接
conn_str = (
    f'DRIVER={{SQL Server}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    f'UID={username};'
    f'PWD={password}'
)
connection = pyodbc.connect(conn_str)

# 循环遍历每个视图，查询数据并保存为 CSV 文件
for view_name in view_names:
    # SQL 查询语句
    query = f"""
    SELECT * 
    FROM [ysdx].[dbo].[{view_name}] 
    WHERE [时间] BETWEEN '{start_time}' AND '{end_time}'
    ORDER BY [时间] DESC;
    """

    # 执行查询并加载到 DataFrame
    df = pd.read_sql(query, connection)

    # 保存数据到 CSV 文件
    output_file = f"C:\\Users\\admin\\Desktop\\gh\\{view_name}.csv"  # 输出文件名
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"数据已保存到 {output_file}")

# 关闭连接
connection.close()
