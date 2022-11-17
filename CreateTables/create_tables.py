import re


def get_tree(file_path):
    """根据txt文件的信息获取各table的column信息"""
    dic_list = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

        while '\n' in lines:  # 删除所有'\n'
            lines.remove('\n')

        for line in lines:
            # 将字符串信息转为词典信息
            line = line.strip('\n').strip().replace('\'', '')
            line = re.split(': | \t ', line)
            line_dic = {line[1]: line[2], line[3]: line[4], line[5]: line[6]}
            dic_list.append(line_dic)

    table_list = [item['表名'] for item in dic_list]  # 所有的table名
    table_dic = {}  # 用于统计各table下的column和对应的comment

    for table in table_list:
        # 将[column, comment]存入到对应的table_dic[table]
        temp_list = [[item['列名'], item['comments']] for item in dic_list if item['表名'] == table]
        table_dic.update({table: temp_list})

    return table_dic


if __name__ == '__main__':
    table_dic = get_tree(file_path=r'/Users/GongLi/Desktop/Satellite_PowerSystem/PowerSystem.txt')

    with open('create_tables.sql', 'a') as f:
        """该文档写入生成数据库的sql语句"""
        for key, val in table_dic.items():
            f.writelines('create table `%s`\n(\n' % key.split('_')[2])
            for item in val:
                item_tuple = (item[0].replace('P0X', '').replace('WW', '_'), item[1])
                if item[0] == 'CSSJ':  # 时间列的数据类型不一样
                    f.writelines('    %s timestamp(6) null comment \'%s\',\n' % item_tuple)
                elif item == val[-1]:  # 最后一行不需要','
                    f.writelines('    %s varchar(32) null comment \'%s\'\n' % item_tuple)
                else:  # 普通行格式一致
                    f.writelines('    %s varchar(32) null comment \'%s\',\n' % item_tuple)
            f.writelines(');\n\n\n')
