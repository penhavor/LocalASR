"""Сетевые утилиты: HTTP-загрузки с прогрессом и доозагрузкой, API Hugging Face."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import quote, urlparse

from ..constants import HF_API_BASE, HF_RESOLVE_BASE, HTTP_CHUNK, HTTP_TIMEOUT, USER_AGENT
from ..logutils import get_logger

log = get_logger("net")

ProgressCallback = Callable[[int, int, float, str], None]  # done, total, speed, filename
CancelToken = Callable[[], bool]


class NetworkError(RuntimeError):
    pass


class DownloadCancelled(RuntimeError):
    pass


def _requests():
    try:
        import requests  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise NetworkError(
            "Не установлен пакет requests. Запусти main.py — он доустановит зависимости."
        ) from exc
    return requests


def session():
    requests = _requests()
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    return sess


# --------------------------------------------------------------------------- #
#  Простые запросы
# --------------------------------------------------------------------------- #


def fetch_text(url: str, timeout: int = HTTP_TIMEOUT) -> str:
    sess = session()
    resp = sess.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise NetworkError(f"HTTP {resp.status_code} для {url}")
    return resp.text


def head_size(url: str, timeout: int = HTTP_TIMEOUT) -> Optional[int]:
    sess = session()
    try:
        resp = sess.head(url, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except Exception as exc:  # noqa: BLE001
        log.debug("HEAD не удался для %s: %s", url, exc)
    return None


# --------------------------------------------------------------------------- #
#  Hugging Face
# --------------------------------------------------------------------------- #


def is_hf_repo_id(text: str) -> bool:
    """Похоже ли на идентификатор репозитория HF (`owner/name` или `owner/name/tree/rev/sub`)."""
    text = (text or "").strip()
    if not text or text.startswith(("http://", "https://", "/", "\\")):
        return False
    if os.path.isabs(text) or Path(text).exists():
        return False
    parts = [p for p in text.split("/") if p]
    return 2 <= len(parts) <= 6 and all(re.fullmatch(r"[\w.\-]+", p) for p in parts)


def hf_resolve_url(repo_id: str, filename: str, revision: str = "main") -> str:
    filename = filename.lstrip("/")
    return f"{HF_RESOLVE_BASE}/{repo_id}/resolve/{revision}/{quote(filename)}"


def hf_repo_page(repo_id: str) -> str:
    return f"{HF_RESOLVE_BASE}/{repo_id}"


@dataclass
class RemoteFile:
    path: str
    size: int
    is_dir: bool = False


def hf_tree(repo_id: str, subfolder: str = "", revision: str = "main", recursive: bool = True) -> List[RemoteFile]:
    """Список файлов репозитория HF с размерами (пагинацию обрабатываем)."""
    sess = session()
    path_part = f"/tree/{quote(revision)}"
    if subfolder:
        path_part += f"/{quote(subfolder.strip('/'))}"
    url = f"{HF_API_BASE}/{quote(repo_id, safe='/')}{path_part}"
    params = {"recursive": "true"} if recursive else {}

    files: List[RemoteFile] = []
    for page in range(20):  # защита от бесконечной пагинации
        log.debug("hf_tree: GET %s (params=%s, страница %d)", url, params, page + 1)
        try:
            resp = sess.get(url, params=params, timeout=HTTP_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - requests и его зависимости
            raise NetworkError(f"Не удалось получить список файлов {repo_id}: {exc}") from exc
        if resp.status_code == 404:
            raise NetworkError(f"Репозиторий или папка не найдены: {repo_id}/{subfolder}")
        if resp.status_code == 401:
            raise NetworkError(f"Репозиторий закрыт (gated): {repo_id}. Нужен токен HF.")
        if resp.status_code != 200:
            raise NetworkError(f"HF API вернул HTTP {resp.status_code} для {repo_id}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise NetworkError(f"HF API вернул не-JSON: {exc}") from exc
        if not isinstance(payload, list):
            raise NetworkError("Неожиданный ответ HF API")

        for item in payload:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            item_path = str(item.get("path") or "")
            if not item_path:
                continue
            size = int(item.get("size") or 0)
            if item_type == "directory":
                files.append(RemoteFile(item_path, 0, True))
            else:
                lfs = item.get("lfs") or {}
                if isinstance(lfs, dict) and lfs.get("size"):
                    size = int(lfs["size"]) or size
                files.append(RemoteFile(item_path, size, False))

        link = resp.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        if not match:
            break
        url = match.group(1)
        params = {}
    log.debug("hf_tree: %s/%s — файлов: %d, байт: %d",
              repo_id, subfolder or "main",
              sum(1 for f in files if not f.is_dir),
              sum(f.size for f in files if not f.is_dir))
    return files


def hf_read_json(repo_id: str, filename: str, revision: str = "main", subfolder: str = "") -> Optional[Dict]:
    """Прочитать маленький JSON (config.json) из репозитория HF. None — если нет.

    Используется, чтобы определить движок модели ДО скачивания.
    """
    rel = f"{subfolder.strip('/')}/{filename}" if subfolder else filename
    url = hf_resolve_url(repo_id, rel, revision=revision)
    try:
        sess = session()
        resp = sess.get(url, timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:  # noqa: BLE001 - нет сети/битый JSON: просто None
        return None


# --------------------------------------------------------------------------- #
#  Скачивание файлов
# --------------------------------------------------------------------------- #


def _sleep_cancelable(seconds: float, cancel: Optional[CancelToken] = None) -> None:
    """Спит мелкими порциями, чтобы отмена срабатывала мгновенно.

    Без этого закрытие окна во время паузы ретрая (до 20 с) оставляло поток
    живым — отсюда «QThread: Destroyed while thread is still running».
    """
    deadline = time.time() + max(0.0, seconds)
    while time.time() < deadline:
        if cancel is not None and cancel():
            raise DownloadCancelled("Скачивание отменено")
        time.sleep(min(0.2, max(0.0, deadline - time.time())))


def download_file(
    url: str,
    dest: Path,
    progress: Optional[ProgressCallback] = None,
    cancel: Optional[CancelToken] = None,
    resume: bool = True,
    timeout: int = 120,
    max_attempts: int = 5,
    backoff_cap: int = 20,
) -> Path:
    """Качает файл с доозагрузкой (Range), прогрессом и отменой.

    Пишет в `dest.part`, по успеху переименовывает — «половинок» в models/ не будет.
    После скачивания сверяет размер с Content-Length: «тихо обрезанный» файл
    (сервер закрыл соединение без ошибки) докачивается заново, а не сохраняется.
    Паузы между попытками слушают отмену — закрыть окно можно в любой момент.
    """
    requests = _requests()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    existing = part.stat().st_size if (resume and part.exists()) else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    log.debug("download_file: %s -> %s (доозагрузка с %d байт)", url, dest, existing)

    class _IncompleteError(RuntimeError):
        """Размер не совпал с Content-Length — файл оборван."""

    attempt = 0
    while True:
        attempt += 1
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as resp:
                if existing and resp.status_code == 416:
                    # файл уже полностью скачан
                    existing = 0
                    headers = {}
                    continue
                if resp.status_code not in (200, 206):
                    hint = ""
                    if resp.status_code in (401, 403):
                        hint = (
                            " — репозиторий закрытый (gated) или нужен токен Hugging Face: "
                            "открой страницу модели в браузере, войди в аккаунт и нажми "
                            "«Agree and access», либо используй открытый зеркальный репозиторий."
                        )
                    raise NetworkError(f"HTTP {resp.status_code} при скачивании {url}{hint}")
                log.debug(
                    "download_file: HTTP %s, content-length=%s, попытка %d",
                    resp.status_code, resp.headers.get("Content-Length"), attempt,
                )

                if resp.status_code == 200 and existing:
                    # сервер не поддержал Range — качаем заново
                    existing = 0
                    mode = "wb"
                else:
                    mode = "ab" if existing else "wb"

                total_header = resp.headers.get("Content-Length")
                total = (int(total_header) + existing) if total_header else 0

                started = time.time()
                done = existing
                last_report = 0.0

                with open(part, mode) as fh:
                    for block in resp.iter_content(chunk_size=HTTP_CHUNK):
                        if cancel is not None and cancel():
                            raise DownloadCancelled("Скачивание отменено")
                        if not block:
                            continue
                        fh.write(block)
                        done += len(block)
                        now = time.time()
                        if progress is not None and (now - last_report) >= 0.1:
                            elapsed = max(now - started, 1e-6)
                            speed = (done - existing) / elapsed
                            progress(done, total, speed, dest.name)
                            last_report = now

                # верификация: оборванное соединение может закрыть поток без
                # исключения — сверяем фактический размер с заявленным
                actual = part.stat().st_size if part.exists() else 0
                if total and actual != total:
                    raise _IncompleteError(f"получено {actual} из {total} байт")
            break
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
            _IncompleteError,
        ) as exc:
            if cancel is not None and cancel():
                raise DownloadCancelled("Скачивание отменено") from exc
            if attempt >= max_attempts:
                raise NetworkError(f"Не удалось скачать {url} после {attempt} попыток: {exc}") from exc
            wait = min(2 ** attempt, backoff_cap)
            log.warning("Сеть моргнула (%s), повтор через %d с…", exc.__class__.__name__, wait)
            _sleep_cancelable(wait, cancel)
            # пересчитываем смещение для доозагрузки
            existing = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}

    if progress is not None:
        progress(done, total or done, 0.0, dest.name)

    if not part.exists() or part.stat().st_size == 0:
        raise NetworkError(f"Пустой файл после скачивания: {url}")

    final_size = part.stat().st_size
    elapsed = max(time.time() - started, 1e-6)
    os.replace(part, dest)
    log.debug(
        "download_file: готово %s (%s за %.1f с, %.1f МБ/с)",
        dest.name, f"{final_size} байт", elapsed, final_size / elapsed / (1024 * 1024),
    )
    return dest


def extract_zip(archive: Path, dest_dir: Path, progress: Optional[Callable[[float], None]] = None) -> List[Path]:
    """Аккуратно распаковывает zip (Vosk-модели), защищаясь от zip-slip."""
    import zipfile

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []

    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        total = max(1, len(members))
        # Vosk-архивы содержат общую корневую папку — срезаем её
        roots = {m.filename.split("/")[0] for m in members}
        strip_root = len(roots) == 1 and all("/" in m.filename for m in members)

        for index, member in enumerate(members, start=1):
            name = member.filename
            if strip_root:
                name = name.split("/", 1)[1]
            if not name:
                continue
            target = (dest_dir / name).resolve()
            try:
                target.relative_to(dest_dir.resolve())
            except ValueError as exc:
                raise NetworkError(f"Небезопасный путь в архиве: {member.filename}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                while True:
                    block = src.read(1024 * 512)
                    if not block:
                        break
                    dst.write(block)
            extracted.append(target)
            if progress is not None:
                progress(index / total)

    return extracted


def looks_like_url(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    parsed = urlparse(text)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = os.path.basename(path)
    return quote(name) if name else "model.bin"


def guess_model_kind(name: str) -> str:
    """Грубая оценка типа модели по имени файла/репозитория (для автоопределения)."""
    low = name.lower()
    if "vosk" in low or low.endswith(".zip"):
        return "vosk"
    if "ggml" in low or low.endswith(".bin") and "ggml" in low:
        return "whispercpp"
    return "faster-whisper"


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
