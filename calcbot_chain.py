# !/usr/bin/env python

import argparse
import logging
from dingtalk_stream import AckMessage
import dingtalk_stream

import os
import dashscope
from dashscope import TextEmbedding
from dashvector import Client, Doc
from dashvector import Doc
import dashvector

import sys

from langchain import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain
import torch

global c_p

# memory
mem1='暂无'
mem2='之前问的问题：'
# mem=f'以下是流程:{mem_l1}'
MEM_MAX=200

class GLM(LLM):
    max_token: int = 2048
    temperature: float = 0.001
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    
    history_len: int = 1024
    
    def __init__(self):
        super().__init__()
        
    @property
    def _llm_type(self) -> str:
        return "GLM"
            
    def load_model(self, llm_device="gpu",model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=model_config, trust_remote_code=True).half().cuda()

    def _call(self,prompt:str,history:List[str] = [],stop: Optional[List[str]] = None):
        response, _ = self.model.chat(
                    self.tokenizer,prompt,
                    history=history[-self.history_len:] if self.history_len > 0 else [],
                    max_length=self.max_token,temperature=self.temperature,
                    top_p=self.top_p)
        return response

modelpath = "./ChatGLM3/chatglm3-6b/"
sys.path.append(modelpath)
llm = GLM()
llm.load_model(model_name_or_path = modelpath)


def setup_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('%(asctime)s %(name)-8s %(levelname)-8s %(message)s [%(filename)s:%(lineno)d]'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def define_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--client_id', dest='client_id', required=True,
        help='app_key or suite_key from https://open-dev.digntalk.com'
    )
    parser.add_argument(
        '--client_secret', dest='client_secret', required=True,
        help='app_secret or suite_secret from https://open-dev.digntalk.com'
    )
    options = parser.parse_args()
    return options


my_shared.c_p={}
def prepare_data(path, size):
    batch_docs = []
    for file in os.listdir(path):
        with open(path + '/' + file, 'r', encoding='utf-8') as f:
            content=f.read()
            my_shared.c_p[content]=file
            print(f'key:{content},value:{my_shared.c_p[content]}')
            # print(f'用于向量化的文本:{content}')
            # batch_docs.append(file)
            batch_docs.append(content)
            if len(batch_docs) == size:
                yield batch_docs[:]
                batch_docs.clear()

    if batch_docs:
        yield batch_docs
        
def generate_embeddings(text):
    rsp = TextEmbedding.call(model=TextEmbedding.Models.text_embedding_v1,
                             input=text)
    
    embeddings = [record['embedding'] for record in rsp.output['embeddings']]
    return embeddings if isinstance(text, list) else embeddings[0]


#############################################################
# chain test
from langchain import PromptTemplate, LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain

# template_string1 = '''请基于```内的内容简洁地回答问题，不要有赘余和废话。"
# ```
# 以下问答与之后的问题有关联，请据此回答以下问题，你需要提炼answer对应的重要信息（包括电话，联系人，名字，流程，专业知识等）
# 背景知识：{context}
# ```
# 我的问题是：{question}。
# '''
# prompt1 = PromptTemplate(template=template_string1, input_variables=["context", "question"])

template_string1 = '''
```
请详细总结，但不要编造内容：{context}
```
请保证```中的信息不丢失，并将```中的内容结合成一句或几句话
'''
prompt1 = PromptTemplate(template=template_string1, input_variables=["context"])
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key='summary')

template_string2 = '''
```
已知信息：{summary}
```
请只根据```中的已知信息，具体的回答用户的问题。若你认为信息不足，只需要提供点到为止的答案即可，但不允许在答案中添加任何已知信息中没有的编造成分。
如果你根据已知信息不知道答案，请直接返回文本："我不清楚答案，我咨询专家后再来回答您。"
用户的问题是：{question}。
'''
prompt2 = PromptTemplate(template=template_string2, input_variables=["summary", "question"])
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key='raw_answer')

template_string3 = '''你是一位财务部门的主管，负责回答报销中遇到的问题
```内是一个问题，我还提供了对应这个问题的比较啰嗦的答案，你要在保证关键信息完整的情况下，删除‘根据所提供知识’，
‘抱歉’，等没意义的内容，也不要添加任何新的内容。但数字等信息量大信息不要删除。
如果原始答案中提到不知道答案，请直接返回文本："我不清楚答案，我咨询专家后再来回答您。"
原始的回答:{raw_answer}
```
原始的回答对应的问题是：{question}。
'''
prompt3 = PromptTemplate(template=template_string3, input_variables=["raw_answer", "question"])
chain3 = LLMChain(llm=llm, prompt=prompt3, output_key='concise_answer')

# print(prompt.input_variables)

def answer_question(_question):
    _context=get_info(_question)
    global llm
    overall_chain = SequentialChain(
        chains=[chain1, chain2, chain3],
            input_variables=["context","question"],
            output_variables=["summary","raw_answer", "concise_answer"],
            verbose=True
        )
    # result=overall_chain.predict(context=str(_context),question=str(_question))
    result=overall_chain({'context':str(_context),'question':str(_question)})
    return result
#############################################################




def get_info(question):
    global collection
    rets = collection.query(generate_embeddings(question), output_fields=['question'], topk=1)
    inf = ''
    print('ok1')
    for doc in rets.output:
        print(f"id: {doc.id}, title: {doc.fields['question']}, score: {doc.score}")
        print('------------'+doc.fields['question'])
        print(my_shared.c_p[doc.fields['question']])
        with open('data/' + my_shared.c_p[doc.fields['question']], 'r', encoding='utf-8') as f:
            print('ok2')
            inf+=f.read()
            inf+='\n'
            print('ok3')

    return inf
        

# ########################################################

class CalcBotHandler(dingtalk_stream.ChatbotHandler):
    def __init__(self, logger: logging.Logger = None):
        super(dingtalk_stream.ChatbotHandler, self).__init__()
        if logger:
            self.logger = logger

    async def process(self, callback: dingtalk_stream.CallbackMessage):
        
        incoming_message = dingtalk_stream.ChatbotMessage.from_dict(callback.data)
        expression = incoming_message.text.content.strip()

        try:
            # result = eval(expression)
            ######################################
            # result = answer_question(expression, pipe, collection)
            result = answer_question(expression)
            print(result)
            result=result
            ######################################
        except Exception as e:
            result = 'Error: %s' % e
        self.logger.info('%s = %s' % (expression, result))
        # response = 'Q: %s\nA: %s' % (expression, result)
        response = '%s' % (str(result))
        # response = 'Q: %s\nA: %s' % (expression, result)
        self.reply_text(response, incoming_message)

        return AckMessage.STATUS_OK, 'OK'
    
############################

        
############################

def main():

    logger = setup_logger()
    options = define_options()

    credential = dingtalk_stream.Credential(options.client_id, options.client_secret)
    client = dingtalk_stream.DingTalkStreamClient(credential)
    client.register_callback_hanlder(dingtalk_stream.chatbot.ChatbotMessage.TOPIC, CalcBotHandler(logger))
    client.start_forever()


if __name__ == '__main__':
    ######################################
    API_KEY='sk-XFZW0LLP5B7Io2Iw0JSvOwrzYOpI922BEBAF2552511EE8D3462DF4573CB4A'
    dashscope.api_key='sk-be6079a33be24834923ae6e1ebfdcdfd'

    client = dashvector.Client(api_key=API_KEY)
    assert client
    
    client.delete('quickstart')
    client.create(name='quickstart', dimension=1536)

    collection = client.get('quickstart')
    assert collection
    # 通过dashvector.Doc对象，插入单条数据
    # collection.insert(Doc(id='1', vector=[0.1, 0.2, 0.3, 0.4]))
    cnt=1
    batch_size=1
    # prepare_data('./data', batch_size)
    for docs in list(prepare_data('./data', batch_size)):
        for doc in docs:
            collection.insert(Doc(id=f'{cnt}', vector=generate_embeddings(doc), fields={"question":doc}))
            print(cnt)
            # print(generate_embeddings(doc))
            cnt+=1

    response=llm('你好')
    print(response)

    ######################################
    # init()
    main()
    
