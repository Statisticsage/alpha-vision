from pydantic import BaseModel, EmailStr, Field

class UserCreate(BaseModel):
    """Schema for user creation requests."""
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)  # ✅ Corrected Field reference

class UserLogin(BaseModel):
    """Schema for user login requests."""
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

class UserResponse(BaseModel):
    """Schema for returning user data."""
    id: int
    email: EmailStr

    class Config:
        """Pydantic v1 & v2 ORM compatibility"""
        from_attributes = True  # Correct for Pydantic v2
        orm_mode = True  # Needed for Pydantic v1 compatibility

class Token(BaseModel):
    """Schema for authentication tokens."""
    access_token: str
    token_type: str
