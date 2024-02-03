import json
import os

import requests


def get_qianfan_token():
    """
    qianfan token
    https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Dlkm79mnx
    """
    ak = os.environ['QIANFAN_AK']
    sk = os.environ['QIANFAN_SK']
    if ak is None or sk is None:
        ValueError('QIANFAN_AK and QIANFAN_SK are not defined.')

    resp = requests.post("https://aip.baidubce.com/oauth/2.0/token", params={
        "grant_type": "client_credentials",
        "client_id": ak,
        "client_secret": sk
    })

    body = json.loads(resp.content)
    return body["access_token"]


def ernie_embedding_v1(contents):
    """
        qianfan bge-large-zh Embedding-V1
        https://cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu
    """
    if len(contents) > 16:
        ValueError("max contents length is 16")

    token = get_qianfan_token()

    resp = requests.post("https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1",
                         params={"access_token": token}, json={"input": contents})
    respBody = json.loads(resp.text)

    vectors = []
    for i in respBody["data"]:
        vectors.append(i["embedding"])
    return vectors

