from pydantic import BaseModel

# Request schema
class SignupRequest(BaseModel):
    username: str
    user_id: str

# Response schema
class SignupResponse(BaseModel):
    status: str
    message: str
    file_path: str = None
