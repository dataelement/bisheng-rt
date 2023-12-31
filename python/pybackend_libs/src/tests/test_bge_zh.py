# flake8: noqa
import numpy as np
from pybackend_libs.dataelem.model.embedding.bge import BGEZhEmbedding


def test_bge_zh():
    params = {
        'pretrain_path': '/home/public/llm/bge-large-zh',
        'devices': '8',
        'gpu_memory': 3
    }

    model = BGEZhEmbedding(**params)

    input_texts = [
        'how much protein should a female eat', '南瓜的家常做法',
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        '1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅'
    ]

    inp = {'model': 'bge_zh', 'type': 'query', 'texts': input_texts[:2]}
    query_embs = model.predict(inp)

    inp = {'model': 'bge_zh', 'type': 'doc', 'texts': input_texts[2:]}
    doc_embs = model.predict(inp)

    embs = query_embs['embeddings'] + doc_embs['embeddings']
    embeddings = np.asarray(embs)
    print(embeddings.shape)

    scores = (embeddings[:2] @ embeddings[2:].T) * 100
    print(scores)
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 0] < scores[1, 1]

    print(np.linalg.norm(embeddings, axis=1))


test_bge_zh()
