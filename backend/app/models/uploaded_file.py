from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import datetime
from app.db_base import Base  # ✅ Import from new module

class UploadedFile(Base):
    """Model for uploaded files."""
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    task = Column(String, nullable=False)  # "analyze" or "predict"
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Foreign Key & Relationship
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="files")  # ✅ **Fixed indentation**
