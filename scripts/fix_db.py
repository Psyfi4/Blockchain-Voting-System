#!/usr/bin/env python3
# scripts/fix_db.py
import sqlite3
from datetime import datetime

DB = "aadhaar_face_db.db"

def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # 1) Add column if it does not exist (no default to avoid "non-constant default" error)
    cur.execute("PRAGMA table_info(persons);")
    cols = [r[1] for r in cur.fetchall()]
    print("persons columns:", cols)
    if "registration_date" not in cols:
        print("Adding registration_date column (nullable, no default)...")
        cur.execute("ALTER TABLE persons ADD COLUMN registration_date TEXT;")
        conn.commit()
    else:
        print("registration_date already present.")

    # 2) Populate registration_date for existing rows that are NULL
    # We'll set them to the current UTC time (ISO format)
    now = datetime.utcnow().isoformat()
    print("Updating NULL registration_date to", now)
    cur.execute("UPDATE persons SET registration_date = ? WHERE registration_date IS NULL;", (now,))
    conn.commit()

    # 3) OPTIONAL: remove rows with no aadhaar_number (they cause ambiguous recognition)
    # If you want to keep them, change this to an update instead of delete.
    cur.execute("SELECT COUNT(*) FROM persons WHERE aadhaar_number IS NULL OR aadhaar_number = '';")
    count = cur.fetchone()[0]
    print("Rows with NULL/empty aadhaar_number:", count)
    if count > 0:
        # Uncomment next two lines to delete them. Use with caution.
        # print("Deleting rows with NULL/empty aadhaar_number...")
        # cur.execute("DELETE FROM persons WHERE aadhaar_number IS NULL OR aadhaar_number = '';")
        # conn.commit()
        print("(Not deleting automatically: remove them manually if you confirm.)")

    cur.close()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()
