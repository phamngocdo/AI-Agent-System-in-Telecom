import asyncio
import shutil
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from .constants import MAX_UPLOAD_BYTES, PDF_CONTENT_TYPES
from .utils import safe_pdf_filename


class MarkerPDFProcessor:
    async def read_and_validate_pdf(self, upload: UploadFile) -> bytes:
        filename = upload.filename or ""
        content_type = (upload.content_type or "").lower()
        if not filename.lower().endswith(".pdf") or (content_type and content_type not in PDF_CONTENT_TYPES):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Chỉ hỗ trợ upload file PDF.",
            )

        data = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File PDF tối đa 3MB.",
            )
        if not data.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File upload không phải PDF hợp lệ.",
            )
        return data

    async def to_markdown(self, data: bytes, filename: str) -> str:
        marker_cli = shutil.which("marker_single")
        if not marker_cli:
            raise RuntimeError("marker-pdf chưa được cài hoặc marker_single không có trong PATH.")

        with tempfile.TemporaryDirectory(prefix="telcollm-marker-") as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / safe_pdf_filename(filename)
            output_dir = temp_path / "out"
            output_dir.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(data)

            process = await asyncio.create_subprocess_exec(
                marker_cli,
                str(input_path),
                "--output_format",
                "markdown",
                "--output_dir",
                str(output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await process.communicate()
            except asyncio.CancelledError:
                await self._terminate_process(process)
                raise

            if process.returncode != 0:
                error_text = (stderr or stdout or b"").decode("utf-8", errors="ignore").strip()
                raise RuntimeError(f"marker-pdf OCR thất bại: {error_text or 'không rõ lỗi'}")

            markdown_files = sorted(output_dir.rglob("*.md"), key=lambda path: path.stat().st_size, reverse=True)
            if not markdown_files:
                raise RuntimeError("marker-pdf không tạo file markdown đầu ra.")

            return markdown_files[0].read_text(encoding="utf-8", errors="ignore").strip()

    @staticmethod
    async def _terminate_process(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
