from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.database import get_db
from core.dependencies import get_current_user
from core.storage import STORAGE_BUCKET, get_supabase, object_key
from models.document import Document
from models.user import User

router = APIRouter()

# Registration and login are handled entirely by Supabase Auth on the client
# (email + password with an emailed 6-digit code, or Google OAuth). The API only
# verifies the resulting access token (see core/dependencies.py).


@router.delete("/profile")
def delete_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete the account: Storage objects, DB rows, and the Supabase auth user."""
    supabase = get_supabase()

    # 1. Remove every stored file for this user (best effort).
    doc_keys = [
        object_key(d.user_id, d.id, d.filename)
        for d in db.query(Document).filter(Document.user_id == current_user.id).all()
    ]
    if doc_keys:
        try:
            supabase.storage.from_(STORAGE_BUCKET).remove(doc_keys)
        except Exception:
            pass

    # 2. Remove the Supabase Auth user (best effort — the DB row goes regardless).
    try:
        supabase.auth.admin.delete_user(current_user.supabase_uid)
    except Exception:
        pass

    # 3. Remove the local rows (ORM cascade drops documents / chunks / sessions / messages).
    db.delete(current_user)
    db.commit()
    return {"message": "User and all associated data deleted successfully"}
