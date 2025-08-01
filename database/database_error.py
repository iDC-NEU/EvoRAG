'''
Author: fzb fzb0316@163.com
Date: 2024-09-16 16:01:35
LastEditors: fzb fzb0316@163.com
LastEditTime: 2024-09-20 14:12:16
FilePath: /RAGWebUi_demo/database/database_error.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''



class DatabaseError(Exception):
    code = -200


class DatabaseUrlFormatError(DatabaseError):
    code = -201

    def __init__(self, url):
        super().__init__(f"Database url格式错误: {url}")


class DatabaseAPIUnsupportedError(DatabaseError):
    code = -202

    def __init__(self, dbname):
        super().__init__(f"不支持如下的数据库: {dbname}")