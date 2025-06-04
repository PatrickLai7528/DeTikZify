import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from detikzify.infer import DetikzifyPipeline
from detikzify.model import load as load_model

app = FastAPI(title="DeTikZify API")

MODEL_NAME = os.getenv("MODEL_NAME", "nllg/detikzify-v2-8b")

model, processor = load_model(MODEL_NAME, device_map="auto")
pipe = DetikzifyPipeline(model=model, processor=processor)

@app.get("/")
def read_root():
    """Health check endpoint.

    Returns
    -------
    dict
        ``{"status": "ok"}`` if the service is running.
    """
    return {"status": "ok"}

@app.post("/generate")
async def generate(file: UploadFile = File(...), preprocess: bool = True):
    """Convert an image to TikZ code.

    Parameters
    ----------
    file : UploadFile
        The image uploaded via multipart form data.
    preprocess : bool, optional
        Apply built-in preprocessing before inference. Defaults to ``True``.

    Returns
    -------
    fastapi.responses.JSONResponse
        JSON object with a ``code`` field containing the generated TikZ.
    """
    image_bytes = await file.read()
    tikzdoc = pipe.sample(image=image_bytes, preprocess=preprocess)
    return JSONResponse({"code": tikzdoc.code})
