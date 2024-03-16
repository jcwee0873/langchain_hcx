import json
import aiohttp
import http.client
from typing import (
    Any,
    Dict,
    List,
    Union,
)
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    SecretStr,
    root_validator
)
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env
)

class HCXEmbeddings(BaseModel, Embeddings):
    """Naver HyperClovaX embedding models.

    To use, you should have the
    environment variable ``NCP_CLOVASTUDIO_API_KEY``,  ``NCP_APIGW_API_KEY``, ``NCP_EMB_APP_ID``` set with your API key or pass it
    as a named parameter to the constructor.

    Parameters
    ----------
    model : str 
        Default: "clir-emb-dolphin". {"clir-emb-dolphin", "clir-sts-dolphin"}
    api_key : str
        Naver Cloud Platform API Key
    api_gw_key : str
        Naver Cloud Platform API Gateway Key
    app_id : str
        Naver Cloud Platform Application ID

    Example
    -------
    ```python
        hcx_emb = HCXEmbeddings(model="clir-emb-dolphin")
    ```
    """
    
    model: str = "clir-emb-dolphin"
    api_key: SecretStr = Field(default=None, alias="api_key")
    api_gw_key: SecretStr = Field(default=None, alias="apigw_key")
    app_id: SecretStr = Field(default=None, alias="app_id")

    ncp_api_base: str = "clovastudio.apigw.ntruss.com"
    
    
    @root_validator()
    def check_api_keys(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        api_key = get_from_dict_or_env(values, "api_key", "NCP_CLOVASTUDIO_API_KEY")
        api_gw_key = get_from_dict_or_env(values, "api_gw_key", "NCP_APIGW_API_KEY")
        app_id = get_from_dict_or_env(values, "app_id", "NCP_EMB_APP_ID")
        
        values["api_key"] = convert_to_secret_str(api_key)
        values["api_gw_key"] = convert_to_secret_str(api_gw_key)
        values["app_id"] = convert_to_secret_str(app_id)

        if values['model'] not in ["clir-emb-dolphin", "clir-sts-dolphin"]:
            raise ValueError("Invalid model. Choose from: {'clir-emb-dolphin', 'clir-sts-dolphin'}")
        

        return values


    def _send_request(self, messages : Union[str, Dict[str, str]]) -> List[float]:
        if isinstance(messages, str):
            messages = {
                "text": messages
            }
        if not isinstance(messages, dict):
            raise TypeError("Messages should be a string or a dictionary with 'text' key")
        
        headers = {
            'Content-Type': 'application/json',
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key.get_secret_value(),
            'X-NCP-APIGW-API-KEY': self.api_gw_key.get_secret_value(),
        }

        conn = http.client.HTTPSConnection(self.ncp_api_base)
        conn.request('POST', f"/testapp/v1/api-tools/embedding/{self.model}/{self.app_id.get_secret_value()}", json.dumps(messages), headers=headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()

        if result['status']['code'] == '20000':
            return result['result']['embedding']

        raise Exception(f"Error: {str(result)}")
    
    
    async def _asend_request(self, messages : Union[str, Dict[str, str]]) -> List[float]:
        headers = {
            'Content-Type': 'application/json',
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key.get_secret_value(),
            'X-NCP-APIGW-API-KEY': self.api_gw_key.get_secret_value(),
        }
        url = f"https://{self.ncp_api_base}/testapp/v1/api-tools/embedding/{self.model}/{self.app_id.get_secret_value()}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=json.dumps(messages), headers=headers) as response:
                response_text = await response.text()
                result = json.loads(response_text)
                
            if result['status']['code'] == '20000':
                return result['result']['embedding']

            raise Exception(f"Error: {str(result)}")
        
    
    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        embeddings = []

        for text in texts:
            embeddings.append(self._send_request(text))

        return embeddings

    async def _aget_embedding(self, texts: List[str]) -> List[List[float]]:
        embeddings = []

        for text in texts:
            embeddings.append(await self._asend_request(text))
    
        return embeddings


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_embedding(texts)
    

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self._aget_embedding(texts)
    

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
    

    async def aembed_query(self, text: str) -> List[float]:
        return await self.aembed_documents([text])[0]
    