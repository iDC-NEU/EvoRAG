'''
Author: fzb fzb0316@163.com
Date: 2024-09-15 17:14:37
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2024-09-22 22:31:03
FilePath: /RAGWebUi_demo/webui_chat.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
import gradio as gr

from config.config import Config
from llmragenv.llmrag_env import LLMRAGEnv


__AVATAR = (os.path.join(os.getcwd(), "resource/avatar/user.jpeg"),
            os.path.join(os.getcwd(), "resource/avatar/logo.jpg"))




def run_webui():
    chat_env = LLMRAGEnv()

    with gr.Blocks() as chat_app:
        gr.Markdown("# Chat iDC-RAG")

        # 定义各个输入组件
        # model_dropdown = gr.Dropdown(
        #     choices=["llama2:7b", "llama2:13b", "llama2:70b","qwen:7b","qwen:14b","qwen:72b"],
        #     label="选择大模型",
        #     value="qwen:14b",  # 默认模型
        #     interactive=True
        # )

        service_option = gr.Radio(
            choices=["Without RAG", "Graph RAG", "Vector RAG"],
            label="选择一个服务功能选项",
            value="Without RAG",  # 默认选项
            interactive=True
        )

        # graph_database = gr.Radio(
        #     choices=["nebulagraph", "neo4j"],
        #     label="选择使用哪个图数据库后端",
        #     value="nebulagraph",  # 默认选项
        #     interactive=True
        # )

        # vector_database = gr.Radio(
        #     choices=["MilvusDB"],
        #     label="选择使用哪个向量数据库后端",
        #     value="MilvusDB",  # 默认选项
        #     interactive=True
        # )


        # 创建 ChatInterface (仅作展示，可根据需求修改)
        # gr.ChatInterface(
        #     chat_env.web_chat,
        #     description="生成式语言模型, graph RAG, Vector RAG",
        #     theme="default",
        #     retry_btn="重试",
        #     submit_btn="发送",
        #     stop_btn="停止",
        #     undo_btn="删除当前问题",
        #     clear_btn="清除所有问题",
        #     concurrency_limit=4,
        #     additional_inputs=[model_dropdown, service_option, graph_database, vector_database],
        # )

        gr.ChatInterface(
            chat_env.low_chat,
            description="生成式AI: GraphRAG, VectorRAG",
            theme="default",
            retry_btn="重试",
            submit_btn="发送",
            stop_btn="停止",
            undo_btn="删除当前问题",
            clear_btn="清除所有问题",
            concurrency_limit=4,
            additional_inputs=[service_option],
        )

        # 检索结果按钮
        display_button = gr.Button("显示检索到的内容")

        # HTML 显示检索结果
        json_output = gr.HTML(label="检索到的内容")

        display_button.click(
            fn = chat_env.get_resulturl,
            inputs=[service_option],
            outputs=[json_output],
        )
    
    chat_app.launch(server_name="0.0.0.0"
                    , server_port=int(Config.get_instance().get_with_nested_params("server", "ui_port"))
                    , share=Config.get_instance().get_with_nested_params("server", "ui_share")
                    , max_threads=10)


if __name__ == "__main__":
    run_webui()


    # message = "你好"
    # history = []

    # from LLM.client.client_factory import ClientFactory
    # answers = ClientFactory().get_client().chat_with_ai_stream(message, history)

    # # print()
    # # print(answers.response)

    # for chunk in answers:
    #     result = chunk.choices[0].delta.content or ""
    #     print(result, end = "")

    # a = ChatAnyTextBot()

    # a.chat_any(message, history)





