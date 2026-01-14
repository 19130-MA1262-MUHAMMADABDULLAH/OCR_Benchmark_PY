import os
import time
import asyncio
import traceback
import requests
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai

from .base import BaseModel
from ..types.model import ExtractionResult
from ..utils import get_mime_type


class DocumentAI(BaseModel):

    def __init__(self, model: str, output_dir: str | None = None):
        super().__init__(model, output_dir)
        self.client = documentai.DocumentProcessorServiceClient(
            client_options=ClientOptions(api_endpoint="us-documentai.googleapis.com")
        )
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "743013682142")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us")
        processor_id = os.getenv("GOOGLE_DOCUMENTAI_PROCESSOR_ID", "cf72efa4495c72a1")
        self.processor_name = self.client.processor_path(
            project_id, location, processor_id
        )

    async def ocr(self, image_path: str) -> ExtractionResult:
        t0 = time.perf_counter()

        # Read the image file as bytes either from the filesystem or a URL
        def _fetch_bytes(path: str):
            if Path(path).exists():
                return Path(path).read_bytes()
            resp = requests.get(path, timeout=60)
            resp.raise_for_status()
            return resp.content
        
        try:
            content_img = await asyncio.to_thread(_fetch_bytes, image_path)
            
            # Load the image mime_type from either the metadata provided or infer from file extension
            mime_img = get_mime_type(image_path)

            # Make a documentai.ProcessRequest, processor_name, and raw_document required. Get them first!
            process_request = documentai.ProcessRequest(
                name=self.processor_name,
                raw_document=documentai.RawDocument(content=content_img, mime_type=mime_img)
            )

            # Call the client to process the document via process_document
            response = self.client.process_document(request=process_request)
            text = response.document.text
            num_pg = response.document.pages.__len__()

            input_cost = num_pg * 1.5 / 1000  # Example cost calculation
            # num_pg * 1.5 / 1000 is better formula than 1.5 / 1000 * num_pg for accuracy

            usage_payload = {
                "duration": time.perf_counter() - t0,
                "input_tokens": num_pg,
                "output_tokens": 0,
                "total_tokens": num_pg,
                "input_cost": input_cost,
                "output_cost": 0.0,
                "total_cost": input_cost,
            }

            return {"text": text, "usage": usage_payload}  # pyright: ignore[reportArgumentType, reportReturnType] # 
        except Exception as e:
            print("DocumentAI OCR error:", e)
            print(traceback.format_exc())
            raise

