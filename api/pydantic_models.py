from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
import os


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class ModelName(str, Enum):
    COMPOUND_BETA_MINI = "compound-beta-mini"
    LLAMA3 = "llama3-70b-8192"
    

class QueryInput(BaseModel):
    question:str
    session_id:str=Field(default=None)
    model : ModelName = Field(default=ModelName.LLAMA3)
    
    
class QueryResponse(BaseModel):
    answer:str
    session_id:str
    model:ModelName
    
class DocumentInfo(BaseModel):
    id:int
    filename:str
    upload_timestamp:datetime
    
class DeleteFileRequest(BaseModel):
    file_id:int