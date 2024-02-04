import json
import os
import time

import requests
import jwt


def generate_token():
    """
    glm token https://open.bigmodel.cn/dev/api#nosdk
    :return: token: str
    """
    apikey = os.environ["GLM_API_KEY"]
    if apikey is None:
        ValueError('')

    try:
        exp_seconds = int(os.environ["GLM_API_KEY"])
    except ValueError or KeyError:
        exp_seconds = 60
        pass

    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


def gle_embedding_v2(sentence: str):
    """
    glm embedding-v2 https://open.bigmodel.cn/dev/api#text_embedding
    """
    resp = requests.post("https://open.bigmodel.cn/api/paas/v4/embeddings",
                         headers={'Authorization': 'Bearer ' + generate_token()},
                         json={'input': sentence,'mode': 'embedding-2'})
    body = json.loads(resp.text)
    return {'tokens': body['usage']['completion_tokens'], 'vector': body['data'][0]['embedding']}