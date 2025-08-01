'''
Author: fzb fzb0316@163.com
Date: 2024-09-15 17:19:40
LastEditors: fzb fzb0316@163.com
LastEditTime: 2024-09-15 17:36:30
FilePath: /LLMKG/LLMKG/utils/schedule.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from apscheduler.schedulers.background import BackgroundScheduler

__scheduler = BackgroundScheduler()


def get_scheduler() -> BackgroundScheduler:
    return __scheduler
