from fastapi import APIRouter

from .endpoints import synrad, nowcast

router = APIRouter()
router.include_router(nowcast.router, prefix="/users", tags=["Nowcast"])
router.include_router(synrad.router, tags =["Synrad"])