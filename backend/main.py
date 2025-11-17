from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes.transactions import transactions_router
from backend.api.routes.auth import auth_router
from backend.api.routes.analytics import analytics_router
from backend.api.routes.budget import budget_router
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.models import SecuritySchemeType

app = FastAPI()

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

#Include routers
app.include_router(transactions_router, prefix="/transactions", tags=["transactions"])
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(analytics_router, prefix="/analytics", tags=["analytics"])
app.include_router(budget_router, prefix="/budget", tags=["budget"])

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Your API",
        version="1.0.0",
        description="API description",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    # Apply the security globally to all endpoints
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            openapi_schema["paths"][path][method]["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi