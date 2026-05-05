from fastapi import APIRouter, Query
from datetime import datetime

router = APIRouter()

@router.get("/")
def health_check():
    return {"status": "ok", "message": "SentinelAI Backend is running."}

@router.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Insider Threat Detection"
    }

@router.get("/sensor-data")
def sensor_data(limit: int = Query(60, ge=1, le=1000)):
    """Returns recent sensor data"""
    return {
        "status": "ok",
        "count": limit,
        "data": [],
        "timestamp": datetime.now().isoformat()
    }

@router.get("/alerts")
def alerts(limit: int = Query(30, ge=1, le=1000)):
    """Returns recent alerts"""
    return {
        "status": "ok",
        "count": limit,
        "alerts": [],
        "timestamp": datetime.now().isoformat()
    }

@router.get("/logs")
def logs(limit: int = Query(30, ge=1, le=1000)):
    """Returns recent system logs"""
    return {
        "status": "ok",
        "count": limit,
        "logs": [],
        "timestamp": datetime.now().isoformat()
    }