import pandas as pd
import json

# 对所有的原始数据进行处理，包括数据清洗、数据预处理等
path = '结直肠癌靶点预测分类模型/datas/结直肠癌_all_414.csv'

data = pd.read_csv(path, encoding='utf-8')
output_data = pd.DataFrame(columns=['patient_SN', 'record', 'MRI', 'CT', 'Ki-67', 'MSI', 'CK', 'P53'])

def check_badian(value, name):
    # ki-67≥25%、MSI 有就算、P53 有就算、CK-20 有就算
    if '%' in value:
        # 提取 % 前面的 2 位数字，转换为 int
        v = int(value.split('%')[0][-2:].replace('约', ''))
        if name == 'Ki-67' and v >= 25:
            return True
        elif name == 'MSI' and v >= 0:
            return True
        elif name == 'P53' and v >= 0:
            return True
        elif name == 'CK-20' and v >= 0:
            return True
        else:
            return False
        
    elif '+' in value or '-' in value:
        if '+' in value:
            return True
        else:
            return False

record_template = '''性别：{gender}
年龄：{age}
现病史：{history}
'''

jiancha_template = '''检查项目：{project}
检查所见：{result}
检查结论：{conclusion}
'''

for i in range(0, len(data)):
    new_row = pd.Series(index=['patient_SN', 'record', 'MRI', 'CT', 'Ki-67', 'MSI', 'CK', 'P53'])

    new_row['patient_SN'] = data.loc[i, 'patient_SN']
    record = record_template.format(gender=data.loc[i, '性别'], age=data.loc[i, '年龄'], history=data.loc[i, '现病史'])
    new_row['record'] = record

    # 字符串转 json 字典
    jiancha = []
    try:
        jiancha = json.loads(data.loc[i, '检查'].replace("'", '"'))
    except json.JSONDecodeError:
        # print('Error: ', data.loc[i, '检查'])
        continue

    # MRI 和 CT 都各取一个最早的日期
    CT, MRI = '', ''
    ct_date, mri_date = '9999-99-99', '9999-99-99'
    for j in jiancha:
        project = j['检查项目'] if '检查项目' in j.keys() else j['检查分类']
        date = j['检查日期'][:10] if '检查日期' in j.keys() else j['检查时间'][:10]

        if project == 'MRI' and j['检查所见'] != '':
            if mri_date == '' or date < mri_date:
                mri_date = date
                MRI = jiancha_template.format(project=project, result=j['检查所见'], conclusion=j['检查结论'])
        elif project == 'CT' and j['检查所见'] != '':
            if ct_date == '' or date < ct_date:
                ct_date = date
                CT = jiancha_template.format(project=project, result=j['检查所见'], conclusion=j['检查结论'])

    new_row['MRI'] = MRI
    new_row['CT'] = CT

    # 处理靶点信息
    if isinstance(data.loc[i, '靶点'], float) or len(data.loc[i, '靶点']) <= 2:
        continue

    try:
        badian = json.loads(data.loc[i, '靶点'].replace("'", '"'))
    except json.JSONDecodeError:
        print('Error: ', data.loc[i, '靶点'])
        continue

    for b in badian:
        name = b['分子病理子项名称']
        if check_badian(b['检验结果（原值）'], name) is True:
            new_row[name] = 1
        else:
            new_row[name] = 0

    output_data = output_data.append(new_row, ignore_index=True)

output_path = '结直肠癌靶点预测分类模型/datas/'
output_data.to_csv(output_path+'all_datas.csv', encoding='utf-8', index=False)

# 按照 7-2-1 的比例划分训练集、测试集和验证集
train_data = output_data.sample(frac=0.7, random_state=0)
test_data = output_data.drop(train_data.index).sample(frac=0.2, random_state=0)
valid_data = output_data.drop(train_data.index).drop(test_data.index)

train_data.to_csv(output_path+'train_datas.csv', encoding='utf-8', index=False)
test_data.to_csv(output_path+'test_datas.csv', encoding='utf-8', index=False)
valid_data.to_csv(output_path+'valid_datas.csv', encoding='utf-8', index=False)
