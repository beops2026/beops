import pickle
import numpy as np
import requests
import sqlite3  
from datetime import datetime
from flask import Flask, request, jsonify
import os

def is_weekend(dt):
    """Check if date is weekend (Sat/Sun).
    
    Input:
        dt (str or datetime): Date as ISO string (YYYY-MM-DD) or datetime object
    
    Output:
        bool: True if Saturday or Sunday, False otherwise
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return dt.weekday() >= 5


delhi_holidays_2026 = [
    "2026-01-01",  # New Year's Day
    "2026-01-26",  # Republic Day
    "2026-03-04",  # Maha Shivaratri
    "2026-03-21",  # Holi
    "2026-03-31",  # Eid-ul-Fitr *
    "2026-04-02",  # Ram Navami
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-04-18",  # Good Friday
    "2026-05-25",  # Buddha Purnima
    "2026-06-17",  # Eid-ul-Zuha (Bakrid) *
    "2026-08-15",  # Independence Day
    "2026-08-28",  # Janmashtami
    "2026-09-17",  # Milad-un-Nabi *
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-01",  # Diwali
    "2026-11-15",  # Guru Nanak Jayanti
    "2026-12-25"   # Christmas
]
def is_delhi_holiday(date_str):
    return date_str in delhi_holidays_2026

print(is_delhi_holiday("2026-02-27"))
print(is_weekend("2026-02-27"))