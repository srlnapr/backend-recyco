from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routers import prediction
import uvicorn
from decouple import config

app = FastAPI(
    title="Waste Classification API",
    description="API untuk klasifikasi sampah menggunakan EfficientNet",
    version="1.0.0"
)

# CORS middleware untuk Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk development, ganti dengan domain frontend di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router, prefix="/api/v1", tags=["prediction"])

@app.get("/")
async def root():
    return {
        "message": "Waste Classification API",
        "version": "1.0.0",
        "status": "active",
        "supported_categories": ["plastic", "organic", "paper", "glass", "metal", "cardboard"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )