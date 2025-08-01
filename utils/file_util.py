'''
Author: fzb0316 fzb0316@163.com
Date: 2024-09-21 19:23:18
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-10-18 15:12:12
FilePath: /BigModel/RAGWebUi_demo/utils/file_util.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import datetime
import os


def clear_files_by_timediff(root_dir, time_interval: int):
    """
    根据文件创建时间删除指定时间间隔之外的文件
    :param root_dir: 指定要删除文件的根目录
    :param time_interval: 文件创建时间与当前时间的时间间隔,单位秒
    :return:
    """
    for file in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file)

        creation_time = os.path.getctime(file_path)
        current_time = datetime.datetime.now().timestamp()

        if current_time - creation_time > time_interval:

            if os.path.isfile(file_path):
                os.unlink(file_path)



def file_exist(path):
    return os.path.exists(path)


def isfile(path):
    return os.path.isfile(path)


def create_dir(path=None):
    if path and not file_exist(path):
        os.makedirs(path)