from sqlalchemy.orm import Session
from app.models.uploaded_file import UploadedFile  # ✅ Correct
 # ✅ Ensure `File` model exists

def get_file_from_db(db: Session, file_id: int):
    """Fetches a file record from the database by its ID."""
    file_record = db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
    if not file_record:
        return None  # Or raise an error
    return file_record