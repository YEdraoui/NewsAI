import uvicorn

if __name__ == "__main__":
    print("Starting NewsAI API server...")
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0", 
        port=8000,
        reload=False
    )