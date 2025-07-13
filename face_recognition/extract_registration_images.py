# extract_registration_images.py
# Extract some embeddings back to create test images (if possible)

import mysql.connector
import numpy as np
import os

def check_registration_data():
    """
    Check what users are registered in your database
    """
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root", 
            password="",
            database="payment_facial"
        )
        cursor = db.cursor()
        
        cursor.execute("SELECT DISTINCT user_id FROM face_embeddings")
        users = cursor.fetchall()
        
        print("🗄️  REGISTERED USERS IN DATABASE:")
        print("=" * 40)
        
        for user_tuple in users:
            user_id = user_tuple[0]
            cursor.execute("SELECT COUNT(*) FROM face_embeddings WHERE user_id = %s", (user_id,))
            count = cursor.fetchone()[0]
            print(f"👤 {user_id}: {count} embeddings")
        
        cursor.close()
        db.close()
        
        return [user[0] for user in users]
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return []

def main():
    """Check registration data"""
    print("🗄️  CHECKING REGISTRATION DATABASE")
    print("=" * 50)
    
    users = check_registration_data()
    
    if users:
        print(f"\n✅ Found {len(users)} registered users")
        print("💡 You need to capture NEW test images for evaluation")
        print("   (Can't extract original images from embeddings)")
    else:
        print("❌ No registered users found or database error")

if __name__ == "__main__":
    main()