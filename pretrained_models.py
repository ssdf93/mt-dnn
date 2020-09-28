from transformers import *
from module.san_model import SanModel
MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "xlm": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
    "san": (BertConfig, SanModel, BertTokenizer),
}

# Support DeBERTa
try:
    from DeBERTa.deberta import DeBERTa
    from DeBERTa.deberta.config import ModelConfig as DeBERTaConfig
    from data_utils.tokenizer import DeBERTaTokenizer
    MODEL_CLASSES["deberta"] = (DeBERTaConfig, DeBERTa, DeBERTaTokenizer)
except Exception as e:
    print("Please install DeBERTa: https://github.com/microsoft/DeBERTa")
