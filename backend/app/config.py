import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    FRONTEND_URL: str
    OPENAI_API_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30  # Default: 30 minutes

    class Config:
        """Ensure .env file is always loaded & validation occurs."""
        env_file = ".env"  
        validate_assignment = True  # ✅ Enforces validation on change
        case_sensitive = True  # ✅ Prevents accidental case mismatches

# Create an instance of Settings
settings = Settings()

# Debugging: Ensure settings are loaded properly
if __name__ == "__main__":
    print(f"✅ Database URL: {settings.DATABASE_URL}")
