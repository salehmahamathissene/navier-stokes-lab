from __future__ import annotations

from pathlib import Path
import inspect

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .detections import detect_bruteforce
from .report import generate_soc_report

router = APIRouter(prefix="/soc", tags=["soc"])

# Render: ALWAYS write to /tmp (guaranteed writable)
OUT_DIR = Path("/tmp/risklab_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEST_PDF = OUT_DIR / "soc_report.pdf"


def _generate_pdf(finding) -> None:
    """
    Supports generate_soc_report that either:
    - writes to a path, OR
    - returns bytes, OR
    - returns a path
    """
    sig = None
    try:
        sig = inspect.signature(generate_soc_report)
    except Exception:
        sig = None

    # Try calling with (finding, output_path) if supported
    if sig is not None:
        params = list(sig.parameters.values())
        if len(params) >= 2:
            try:
                out = generate_soc_report(finding, LATEST_PDF)
                # if it returned bytes
                if isinstance(out, (bytes, bytearray)):
                    LATEST_PDF.write_bytes(out)
                # if it returned a path-like
                elif isinstance(out, (str, Path)):
                    p = Path(out)
                    if p.exists() and p != LATEST_PDF:
                        LATEST_PDF.write_bytes(p.read_bytes())
                return
            except TypeError:
                pass

    # Fallback: call with only (finding)
    out = generate_soc_report(finding)
    if isinstance(out, (bytes, bytearray)):
        LATEST_PDF.write_bytes(out)
    elif isinstance(out, (str, Path)):
        p = Path(out)
        if not p.exists():
            raise RuntimeError("generate_soc_report returned a path, but it does not exist.")
        if p != LATEST_PDF:
            LATEST_PDF.write_bytes(p.read_bytes())


@router.post("/upload")
async def soc_upload(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        text = raw.decode("utf-8", errors="ignore")

        finding = detect_bruteforce(text)

        # Generate PDF
        _generate_pdf(finding)

        if not LATEST_PDF.exists():
            raise RuntimeError("SOC PDF not created.")

        # Return JSON always
        return {
            "ok": True,
            "severity": getattr(finding, "severity", "UNKNOWN"),
            "summary": getattr(finding, "summary", "SOC report generated."),
            "report_url": "/soc/report/latest",
        }

    except Exception as e:
        # Return JSON error (not plain text)
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": str(e),
            },
        )


@router.get("/report/latest")
def soc_report_latest():
    if not LATEST_PDF.exists():
        raise HTTPException(404, "No SOC report yet. POST /soc/upload first.")
    return FileResponse(
        str(LATEST_PDF),
        media_type="application/pdf",
        filename="soc_report.pdf",
    )
