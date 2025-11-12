"""
Database models for CAIretaker Fall Detection System
Handles incident logging with SQLite
"""

import sqlite3
from datetime import datetime
import os
from contextlib import contextmanager

class Database:
    def __init__(self, db_path="backend/fall_incidents.db"):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create incidents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fall_incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    location TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT DEFAULT 'Active',
                    resolved_at DATETIME,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_person_id 
                ON fall_incidents(person_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON fall_incidents(timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_status 
                ON fall_incidents(status)
            ''')
            
            print("✓ Database initialized successfully")
    
    def log_fall_incident(self, person_id, confidence, location="Live Camera"):
        """Log a new fall incident"""
        now = datetime.now()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fall_incidents 
                (person_id, timestamp, date, time, location, confidence, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                person_id,
                now.isoformat(),
                now.strftime('%Y-%m-%d'),
                now.strftime('%H:%M:%S'),
                location,
                confidence,
                'Active'
            ))
            
            incident_id = cursor.lastrowid
            
            print(f"✓ Fall incident logged: ID={incident_id}, Person={person_id}, Time={now.strftime('%H:%M:%S')}")
            
            return incident_id
    
    def get_active_fall_for_person(self, person_id):
        """Check if person has an active fall incident"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM fall_incidents
                WHERE person_id = ? AND status = 'Active'
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (person_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def resolve_fall_incident(self, incident_id):
        """Mark fall incident as resolved"""
        now = datetime.now()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE fall_incidents
                SET status = 'Resolved', resolved_at = ?
                WHERE id = ?
            ''', (now.isoformat(), incident_id))
            
            print(f"✓ Fall incident resolved: ID={incident_id}")
    
    def resolve_fall_for_person(self, person_id):
        """Resolve all active falls for a person (when they stand up)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE fall_incidents
                SET status = 'Resolved', resolved_at = ?
                WHERE person_id = ? AND status = 'Active'
            ''', (datetime.now().isoformat(), person_id))
            
            if cursor.rowcount > 0:
                print(f"✓ Resolved {cursor.rowcount} fall incident(s) for Person ID {person_id}")
    
    def delete_incident(self, incident_id):
        """Delete a specific incident"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM fall_incidents
                WHERE id = ?
            ''', (incident_id,))
            
            if cursor.rowcount > 0:
                print(f"✓ Deleted incident ID {incident_id}")
            else:
                print(f"⚠ Incident ID {incident_id} not found")
    
    def get_all_incidents(self, limit=100, status=None):
        """Get all incidents with optional status filter"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute('''
                    SELECT * FROM fall_incidents
                    WHERE status = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (status, limit))
            else:
                cursor.execute('''
                    SELECT * FROM fall_incidents
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_incidents_by_date_range(self, start_date, end_date):
        """Get incidents within a date range"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM fall_incidents
                WHERE date BETWEEN ? AND ?
                ORDER BY timestamp DESC
            ''', (start_date, end_date))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_statistics(self):
        """Get incident statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total incidents
            cursor.execute('SELECT COUNT(*) as total FROM fall_incidents')
            total = cursor.fetchone()['total']
            
            # Active incidents
            cursor.execute('SELECT COUNT(*) as active FROM fall_incidents WHERE status = "Active"')
            active = cursor.fetchone()['active']
            
            # Resolved incidents
            cursor.execute('SELECT COUNT(*) as resolved FROM fall_incidents WHERE status = "Resolved"')
            resolved = cursor.fetchone()['resolved']
            
            # Today's incidents
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute('SELECT COUNT(*) as today FROM fall_incidents WHERE date = ?', (today,))
            today_count = cursor.fetchone()['today']
            
            return {
                'total': total,
                'active': active,
                'resolved': resolved,
                'today': today_count
            }
    
    def clear_all_incidents(self):
        """Clear all incidents (use with caution)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM fall_incidents')
            print(f"✓ Cleared all incidents from database")


# Singleton database instance
_db_instance = None

def get_database():
    """Get database singleton instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance