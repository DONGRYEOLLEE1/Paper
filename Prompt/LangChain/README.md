## 🦜️🔗LangChain

LangChain is a framework for developing applications powered by language models.

### Installation

```
pip install langchain
pip install chromadb 
pip install tiktoken
```

‼️ `chromadb` 설치시 dependency problem 생길 수 있음

### API Key

```python
import os
YOUR_API_KEY = 'sk...'
os.environ['OPENAI_API_KEY'] = f'{YOUR_API_KEY}'
```

### Reference

- [LangChain Docs](https://python.langchain.com/en/latest/)
- [LangChain Github](https://github.com/hwchase17/langchain)
