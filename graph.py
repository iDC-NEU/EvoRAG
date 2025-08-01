'''
Author: lpz 1565561624@qq.com
Date: 2024-09-22 20:25:29
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-09-27 18:16:54
FilePath: /RAG_demo/RAGWebUi_demo/graph.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''




from flask import Flask, render_template, request, jsonify
import json
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('graph.html')

@app.route('/update_graph', methods=['POST'])
def update_graph():
    try:
        with open('./templates/triplet.json', 'r', encoding='utf-8') as f:
            triplet = json.load(f)  # 解析 JSON 数据为 Python 对象
        print(triplet)
        return triplet
    except Exception as e:
        return {'error': f"Error parsing graph data: {str(e)}"}



with open('./templates/triplet.json', 'r', encoding='utf-8') as f:
    triplet = json.load(f)  # 解析 JSON 数据为 Python 对象
    print(triplet)
app.run(host='0.0.0.0', port=8085)