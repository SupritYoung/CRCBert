import json

path1 = '结直肠癌2021年以前.json'
path2 = '结直肠癌2021年以后.json'

output_path = '结直肠癌_all_0329.json'


with open(path1, 'r', encoding='utf-8') as f:
    datas1 = json.load(f)
    
with open(path2, 'r', encoding='utf-8') as f:
    datas2 = json.load(f)

merged_datas = datas2
index = 1
for id, value in datas1.items():
    idx = '21q'+str(index)
    value['patient_SN'] = id
    # 删除 id 字段
    value.pop('id')

    merged_datas[idx] = value

    index += 1

# 统一标准化处理
badian_notnull = 0
for id, value in merged_datas.items():
    if value['性别'] == '男性':
        value['性别'] = '男'
    elif value['性别'] == '女性':
        value['性别'] = '女'

    if value['年龄']:
        value['年龄'] = int(value['年龄'])
    else:
        value['年龄'] = '不详'
    
    if '靶点' in value.keys() and len(value['靶点']) > 0:
        badian_notnull += 1

# 统计情况
print('合并后的数据量:', len(merged_datas))
print('靶点不为空的数据量:', badian_notnull)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(merged_datas, f, ensure_ascii=False, indent=4)

