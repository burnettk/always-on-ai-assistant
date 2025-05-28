from pydantic import BaseModel, create_model
from typing import Optional
import inspect
import json
import sys

class User(BaseModel):
    name: str
    age: int
    city: Optional[str] = None

# Decorator to mark functions as excellent tools
def excellent_tool(func):
    """Marks a function as an excellent tool."""
    func._is_excellent_tool = True
    return func

@excellent_tool
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
        "parameters": create_model(func.__name__ + 'Params', **params).model_json_schema()
    }

if __name__ == "__main__":
    tool_schemas = []
    current_module = sys.modules[__name__]
    for name, member in inspect.getmembers(current_module):
        if inspect.isfunction(member) and hasattr(member, '_is_excellent_tool'):
            schema = get_function_schema(member)
            tool_schemas.append(schema)

    print(json.dumps(tool_schemas, indent=2))

# Expected output (assuming get_user_info is the only decorated function):
# [
#   {
#     "name": "get_user_info",
#     "description": "Fetches user information from the database.",
#     "parameters": {
#       "properties": {
#         "user_id": {
#           "title": "User Id",
#           "type": "integer"
#         }
#       },
#       "required": [
#         "user_id"
#       ],
#       "title": "get_user_infoParams",
#       "type": "object"
#     }
#   }
# ]
