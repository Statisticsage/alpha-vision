from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.config import settings
from app.db_base import Base  # ✅ Import Base from a separate module

# Database connection URL
DATABASE_URL = settings.DATABASE_URL

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency function to get the database session."""
    db = SessionLocal()
    try:
        yield db  # ✅ Corrected
    finally:
        db.close()
