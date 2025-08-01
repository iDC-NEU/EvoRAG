'''
Author: fzb fzb0316@163.com
Date: 2024-09-15 17:14:55
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-10-25 11:00:37
FilePath: /LLMKG/LLMKG/logger.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os.path
from typing import Optional

from loguru import logger


if not os.path.exists(os.path.join(os.getcwd(), "logs")):
    os.makedirs(os.path.join(os.getcwd(), "logs"))

class Logger:
    """
    定义一个全局日志工具类，设置好路径、rotation等信息
    """

    def __init__(self, name: Optional[str] = None):
        """
        初始化日志工具类
        """
        # 设置日志文件路径
        self.logger = logger
        self.log_name = os.path.join(os.getcwd(), "logs", f"{name if name else 'file_{time}'}.log")

        # 如果日志文件存在，则清除其内容
        if os.path.exists(self.log_name):
            with open(self.log_name, 'w') as f:
                f.truncate(0)  # 清空文件内容

        self.logger.add(self.log_name
                        # , rotation='10 MB'
                        # , retention='30 days'
                        )

    def info(self, message):
        """
        输出INFO级别的日志
        :param message: 日志信息
        """
        self.logger.info(message)

    def debug(self, message):
        """
        输出DEBUG级别的日志
        :param message: 日志信息
        """
        self.logger.debug(message)

    def warning(self, message):
        """
        输出WARNING级别的日志
        :param message: 日志信息
        """
        self.logger.warning(message)

    def error(self, message):
        """
        输出ERROR级别的日志
        :param message: 日志信息
        """
        self.logger.error(message)

    def log(self, message):
        with open(self.log_name, 'a+') as f:
            f.write(message + '\n')