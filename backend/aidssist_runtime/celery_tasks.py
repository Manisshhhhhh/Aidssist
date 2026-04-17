from __future__ import annotations

from .celery_app import celery_app
from .data_intelligence import prepare_asset_intelligence
from .import_orchestrator import process_kaggle_import_job


if celery_app is not None:

    @celery_app.task(name="aidssist.kaggle_import")
    def run_kaggle_import_job(job_id: str):
        return process_kaggle_import_job(job_id)


    @celery_app.task(name="aidssist.refresh_asset_intelligence")
    def refresh_asset_intelligence_job(asset_id: str):
        return prepare_asset_intelligence(asset_id, force_refresh=True)

