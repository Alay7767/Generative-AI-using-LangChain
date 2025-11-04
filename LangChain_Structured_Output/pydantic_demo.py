from pydantic import BaseModel, EmailStr, Field 
from typing import Optional

class Student(BaseModel):
    name : str = 'Alay'
    age : Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=4, default=0)
    
new_student = {'age' : 23, 'email': 'abc@gmail.com', 'cgpa': 3.78}

student = Student(**new_student)
print(student) 