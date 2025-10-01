from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.get("/")
async def docs():
    return RedirectResponse(url="/docs")
