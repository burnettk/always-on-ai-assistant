from pydantic import BaseModel
from typing import Optional
import inspect
import json

class User(BaseModel):
    name: str
    age: int
    city: Optional[str] = None

def get_user_info(user_id: int) -> User:
    """Fetches user information from the database."""
    # Simulate database lookup
    if user_id == 1:
        return User(name="John Doe", age=30, city="New York")
    elif user_id == 2:
         return User(name="Jane Smith", age=25)
    else:
        raise ValueError("User not found")

def get_function_schema(func):
    """Generates a JSON schema for a function using Pydantic."""
    signature = inspect.signature(func)
    params = {}
    for name, param in signature.parameters.items():
        annotation = param.annotation
        if annotation == inspect._empty:
            raise ValueError(f"Missing type annotation for parameter '{name}'")
        params[name] = (annotation, ...)
    
    return {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": json.loads(BaseModel.model_creator(func.__name__, **params).schema_json())
    }

function_schema = get_function_schema(get_user_info)
print(json.dumps(function_schema, indent=2))

