"""Скачивание и «установка» моделей в папку `models/` проекта.

Поддерживаются источники:
  * репозиторий Hugging Face (`owner/name`, опционально с подпапкой);
  * прямая ссылка на файл (`https://.../ggml-large-v3-turbo-q5_0.bin`, zip Vosk);
  * локальная папка/файл (копируются или подключаются на месте).

Каждая установленная модель получает `models/<id>/installed.json` — по нему GUI
потом понимает, что за модель лежит на диске.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from .. import paths
from ..constants import APP_VERSION, ENGINE_VOSK, ENGINE_WHISPERCPP, MODELS_DIRNAME
from ..logutils import get_logger
from . import net
from .models_catalog import META_FILENAME, ModelEntry, ct2_subfolder, detect_engine

log = get_logger("downloader")

ProgressCallback = Callable[[float, str], None]  # (доля 0..1, текст статуса)
CancelToken = Callable[[], bool]

# Файлы, которые не нужны для запуска (экономят гигабайты трафика)
SKIP_ALWAYS = (
    re.compile(r"\.gitattributes$", re.I),
    re.compile(r"^\.git/", re.I),
    re.compile(r"^onnx/", re.I),
    re.compile(r"^flax_model", re.I),
)
WEIGHT_EXTENSIONS = (".safetensors", ".pt", ".pth", ".msgpack", ".onnx", ".bin")
KEEP_ALWAYS = {
    "model.bin", "config.json", "tokenizer.json", "vocabulary.json", "vocabulary.txt",
    "preprocessor_config.json", "added_tokens.json", "special_tokens_map.json",
    "tokenizer_config.json", "vocab.json", "merges.txt", "normalizer.json",
    "generation_config.json", "processor_config.json", "tokenizer.model",
    "chat_template.json", "model.safetensors.index.json", "README.md",
}


@dataclass
class InstallResult:
    entry: ModelEntry
    directory: Path
    downloaded_bytes: int = 0
    skipped_files: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class InstallError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
#  Разбор источника
# --------------------------------------------------------------------------- #


def split_repo_and_subfolder(source: str) -> Tuple[str, str, str]:
    """Разбирает ссылку на HF-модель: (репозиторий, подпапка, revision).

    Поддерживает форматы:
      owner/name                                    -> (owner/name, "", main)
      owner/name/tree/main/ct2_int8_float16         -> (owner/name, ct2_int8_float16, main)
      owner/name/tree/v1.0/sub/dir                  -> (owner/name, sub/dir, v1.0)
      owner/name:subfolder                          -> (owner/name, subfolder, main)
      https://huggingface.co/owner/name/tree/main/x -> (owner/name, x, main)
    """
    text = (source or "").strip().rstrip("/")
    if text.startswith(("http://", "https://")):
        parsed = urlparse(text)
        if "huggingface.co" in parsed.netloc:
            text = parsed.path.lstrip("/")
            for prefix in ("models/", "api/models/"):
                if text.startswith(prefix):
                    text = text[len(prefix):]
        else:
            return text, "", "main"  # прямая ссылка на файл

    if ":" in text and "/tree/" not in text:
        repo, _, sub = text.partition(":")
        return repo.strip("/"), sub.strip("/"), "main"

    if "/tree/" in text:
        repo, _, rest = text.partition("/tree/")
        parts = [part for part in rest.split("/") if part]
        revision = parts[0] if parts else "main"
        sub = "/".join(parts[1:])
        return repo.strip("/"), sub, revision

    if "/blob/" in text:
        repo, _, rest = text.partition("/blob/")
        parts = [part for part in rest.split("/") if part]
        revision = parts[0] if parts else "main"
        return repo.strip("/"), "", revision

    return text.strip("/"), "", "main"


def resolve_source(entry: ModelEntry) -> Dict:
    """Возвращает план установки: что и откуда качать.

    {
      "kind": "hf_repo" | "url_file" | "local",
      "files": [(relpath, url, size)],
      "total_bytes": int,
    }
    """
    source = (entry.source or "").strip()
    if not source:
        raise InstallError(f"У модели «{entry.name}» не указан источник")

    local = Path(source).expanduser()
    if local.exists():
        return _plan_local(local, entry)

    if net.looks_like_url(source):
        return _plan_url(source, entry)

    repo_id, subfolder, revision = split_repo_and_subfolder(source)
    subfolder = entry.subfolder or subfolder
    revision = entry.revision or revision
    if net.is_hf_repo_id(repo_id):
        return _plan_hf(repo_id, subfolder, entry, revision)

    raise InstallError(
        f"Не понял источник модели: «{source}».\n"
        "Ожидается ID репозитория Hugging Face (owner/name), прямая ссылка https://… "
        "или путь к локальной папке/файлу."
    )


def _plan_hf(repo_id: str, subfolder: str, entry: ModelEntry, revision: str = "main") -> Dict:
    try:
        remote = net.hf_tree(repo_id, subfolder, revision=revision)
    except net.NetworkError as exc:
        raise InstallError(str(exc)) from exc

    files = [f for f in remote if not f.is_dir]
    if not files:
        raise InstallError(f"В {repo_id}/{subfolder or 'main'} нет файлов")

    single_file = entry.file
    if single_file:
        matches = [f for f in files if f.path.endswith(single_file) or f.path.split("/")[-1] == single_file]
        if not matches:
            raise InstallError(f"Файл {single_file} не найден в {repo_id}")
        files = matches[:1]
    else:
        files = _filter_needed(files)

    plan = []
    for item in files:
        rel = item.path
        if subfolder and rel.startswith(subfolder.rstrip("/") + "/"):
            rel = rel[len(subfolder.rstrip("/")) + 1:]
        plan.append((rel, net.hf_resolve_url(repo_id, item.path, revision=revision), item.size))

    total = sum(size for _, _, size in plan)
    return {
        "kind": "hf_repo",
        "files": plan,
        "total_bytes": total,
        "repo": repo_id,
        "subfolder": subfolder,
        "revision": revision,
    }


def _filter_needed(files) -> List:
    """Отсекает лишнее: дубли комплектов весов и служебный мусор.

    Правило: оставляем ТОЛЬКО ОДИН комплект весов — тот, который лежит в
    репозитории (CT2 model.bin → safetensors → pytorch_model.bin), иначе
    transformers-модели (Qwen, GigaAM) скачались бы без весов.
    """
    names = [f.path.split("/")[-1].lower() for f in files]
    has_model_bin = "model.bin" in names
    has_safetensors = any(n.endswith(".safetensors") for n in names)

    result = []
    for item in files:
        name = item.path.split("/")[-1]
        low = name.lower()
        if any(pattern.search(item.path) for pattern in SKIP_ALWAYS):
            continue
        if low.endswith(WEIGHT_EXTENSIONS) and low not in KEEP_ALWAYS:
            if has_model_bin and low != "model.bin":
                continue  # есть CT2-веса — остальные комплекты не качаем
            if not has_model_bin and has_safetensors and not low.endswith(".safetensors"):
                continue  # есть safetensors — bin/pt-дубликаты не качаем
        result.append(item)
    return result or list(files)


def _plan_url(url: str, entry: ModelEntry) -> Dict:
    name = unquote(os.path.basename(urlparse(url).path)) or net.filename_from_url(url)
    size = entry.size_bytes or net.head_size(url) or 0
    return {"kind": "url_file", "files": [(name, url, size)], "total_bytes": size, "filename": name}


def _plan_local(local: Path, entry: ModelEntry) -> Dict:
    if local.is_file():
        size = local.stat().st_size
        return {"kind": "local", "files": [(local.name, str(local), size)], "total_bytes": size, "local": str(local)}
    total = paths.dir_size(local)
    files = []
    for child in sorted(local.rglob("*")):
        if child.is_file():
            files.append((str(child.relative_to(local)), str(child), child.stat().st_size))
    return {"kind": "local", "files": files, "total_bytes": total, "local": str(local)}


# --------------------------------------------------------------------------- #
#  Установка
# --------------------------------------------------------------------------- #


def install_model(
    entry: ModelEntry,
    root: Optional[Path] = None,
    progress: Optional[ProgressCallback] = None,
    cancel: Optional[CancelToken] = None,
) -> InstallResult:
    """Скачивает модель в models/<id>/ и пишет installed.json."""
    started = time.time()
    root = Path(root) if root else paths.ROOT_DIR
    models_root = root / MODELS_DIRNAME
    models_root.mkdir(parents=True, exist_ok=True)

    plan = resolve_source(entry)
    files: List[Tuple[str, str, int]] = plan["files"]
    total_bytes = plan.get("total_bytes") or sum(size for _, _, size in files) or entry.size_bytes or 0

    target_dir = models_root / sanitize_id(entry.id)
    if plan["kind"] == "local" and Path(plan["local"]).is_dir():
        source_dir = Path(plan["local"]).resolve()
        if _is_inside(source_dir, target_dir.resolve()):
            raise InstallError("Нельзя установить модель саму в себя")
        target_dir = source_dir if _is_inside(source_dir, models_root.resolve()) else target_dir

    paths.ensure_free_space(models_root, total_bytes or 0, margin_bytes=256 * 1024 * 1024)

    if progress:
        progress(0.0, f"Подготовка: {entry.name} ({paths.human_size(total_bytes)})")

    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped: List[str] = []
    done_bytes = 0

    def report(current: int, total: int, speed: float, filename: str) -> None:
        nonlocal done_bytes
        done_bytes = current
        if progress and total_bytes:
            fraction = min(0.999, (downloaded + current) / max(1, total_bytes))
            speed_text = f" · {paths.human_size(speed)}/с" if speed else ""
            progress(fraction, f"{filename} · {paths.human_size(downloaded + current)} из {paths.human_size(total_bytes)}{speed_text}")

    try:
        if plan["kind"] == "local":
            for rel, src, size in files:
                if cancel and cancel():
                    raise net.DownloadCancelled("Установка отменена")
                dest = target_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if Path(src).resolve() != dest.resolve():
                    shutil.copy2(src, dest)
                downloaded += size
                if progress and total_bytes:
                    progress(min(0.999, downloaded / total_bytes), f"Копирую {rel}")
        elif plan["kind"] == "url_file" and str(files[0][0]).lower().endswith(".zip"):
            archive = target_dir / files[0][0]
            net.download_file(files[0][1], archive, progress=report, cancel=cancel)
            downloaded = archive.stat().st_size
            if progress:
                progress(0.999, "Распаковываю архив…")
            net.extract_zip(archive, target_dir, progress=lambda p: None)
            archive.unlink(missing_ok=True)
            # Vosk распаковывается в подпапку вида vosk-model-ru-0.42/* — выравниваем
            _flatten_single_subdir(target_dir)
        else:
            for rel, url, size in files:
                if cancel and cancel():
                    raise net.DownloadCancelled("Установка отменена")
                dest = target_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                net.download_file(url, dest, progress=report, cancel=cancel)
                downloaded += size

        engine = detect_engine(target_dir) or entry.engine
        if not engine:
            raise InstallError(
                f"Скачанные файлы не похожи на модель ни одного поддерживаемого движка.\n"
                f"Папка: {target_dir}\n"
                "Ожидается: model.bin+config.json (faster-whisper), ggml-*.bin (whisper.cpp) "
                "или am/final.mdl (Vosk)."
            )

        entry.engine = engine
        entry.install_dir = target_dir
        entry.installed = True
        entry.installed_size_bytes = paths.dir_size(target_dir)
        if not entry.size_bytes:
            entry.size_bytes = total_bytes or entry.installed_size_bytes

        meta = {
            "id": entry.id,
            "name": entry.name,
            "engine": engine,
            "source": entry.source,
            "subfolder": plan.get("subfolder", entry.subfolder),
            "revision": plan.get("revision", entry.revision),
            "size_bytes": entry.size_bytes,
            "installed_size_bytes": entry.installed_size_bytes,
            "ram_mb": entry.ram_mb,
            "vram_mb": entry.vram_mb,
            "languages": entry.languages,
            "description": entry.description,
            "url": entry.url or (net.hf_repo_page(plan["repo"]) if plan.get("repo") else ""),
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "app_version": APP_VERSION,
            "custom": bool(entry.custom),
            "subdir": ct2_subfolder(target_dir),
        }
        net.write_json(target_dir / META_FILENAME, meta)

    except net.DownloadCancelled:
        _cleanup_partial(target_dir)
        raise
    except InstallError:
        raise
    except net.NetworkError as exc:
        raise InstallError(f"Ошибка сети: {exc}") from exc
    except OSError as exc:
        raise InstallError(f"Ошибка диска: {exc}") from exc

    duration = time.time() - started
    log.info(
        "Модель установлена: %s -> %s (%s за %.1f с)",
        entry.id, target_dir, paths.human_size(downloaded), duration,
    )
    if progress:
        progress(1.0, f"Готово: {entry.name} установлена")
    return InstallResult(
        entry=entry,
        directory=target_dir,
        downloaded_bytes=downloaded,
        skipped_files=skipped,
        duration_seconds=duration,
    )


def _flatten_single_subdir(folder: Path) -> None:
    """Если после распаковки получилась одна вложенная папка — поднимаем её содержимое."""
    try:
        children = [p for p in folder.iterdir() if not p.name.startswith(".")]
    except OSError:
        return
    if len(children) != 1 or not children[0].is_dir():
        return
    inner = children[0]
    for item in list(inner.iterdir()):
        shutil.move(str(item), str(folder / item.name))
    try:
        inner.rmdir()
    except OSError:
        pass


def _cleanup_partial(folder: Path) -> None:
    """Удаляет недокачанные .part-файлы (сама папка остаётся для доозагрузки)."""
    try:
        for part in folder.rglob("*.part"):
            part.unlink(missing_ok=True)
    except OSError:
        pass


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def sanitize_id(model_id: str) -> str:
    """Имя папки модели: безопасно для всех ФС."""
    cleaned = re.sub(r"[^\w.\-]+", "_", model_id.strip(), flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned[:80] or "model"


def uninstall_model(entry: ModelEntry, root: Optional[Path] = None) -> bool:
    """Удаляет модель с диска (если она лежит внутри models/)."""
    root = Path(root) if root else paths.ROOT_DIR
    folder = entry.install_dir or (root / MODELS_DIRNAME / sanitize_id(entry.id))
    folder = Path(folder)
    if not folder.exists():
        return False
    models_root = (root / MODELS_DIRNAME).resolve()
    try:
        folder.resolve().relative_to(models_root)
    except ValueError:
        log.warning("Отказано: %s лежит вне models/ — не удаляю", folder)
        return False
    shutil.rmtree(folder, ignore_errors=True)
    return not folder.exists()


# --------------------------------------------------------------------------- #
#  Автоопределение модели по произвольной ссылке/пути (для «своей модели»)
# --------------------------------------------------------------------------- #


def inspect_custom_source(source: str, root: Optional[Path] = None) -> ModelEntry:
    """Создаёт ModelEntry из произвольного источника: путь, HF-репо или URL.

    Никаких скачиваний — только определение типа, размера и имени.
    """
    source = (source or "").strip()
    if not source:
        raise InstallError("Пустой источник модели")

    model_id = ""
    name = ""
    engine = ""
    size_bytes: Optional[int] = None
    subfolder = ""
    revision = "main"
    single_file = ""
    url = ""
    languages: List[str] = []

    local = Path(source).expanduser()
    if local.exists():
        engine = detect_engine(local)
        if local.is_dir():
            name = local.name
            model_id = f"local-{local.name}"
            size_bytes = paths.dir_size(local)
            subfolder = ct2_subfolder(local)
        else:
            name = local.stem
            model_id = f"local-{local.stem}"
            size_bytes = local.stat().st_size
            engine = engine or net.guess_model_kind(local.name)
            single_file = local.name
        url = ""
    elif net.looks_like_url(source):
        parsed = urlparse(source)
        if "huggingface.co" in parsed.netloc and "/resolve/" not in parsed.path:
            repo_id, subfolder, revision = split_repo_and_subfolder(source)
            name = repo_id
            model_id = repo_id
            url = net.hf_repo_page(repo_id)
            try:
                files = net.hf_tree(repo_id, subfolder)
                size_bytes = sum(f.size for f in files if not f.is_dir)
                names = [f.path.split("/")[-1].lower() for f in files]
                if any(n.startswith("ggml") for n in names):
                    engine = "whispercpp"
                elif "model.bin" in names:
                    engine = "faster-whisper"
                elif any(n == "final.mdl" for n in names):
                    engine = ENGINE_VOSK
                else:
                    engine = "faster-whisper"
            except net.NetworkError as exc:
                raise InstallError(str(exc)) from exc
        else:
            filename = unquote(os.path.basename(parsed.path)) or "model.bin"
            name = Path(filename).stem
            model_id = Path(filename).stem
            url = source
            try:
                size_bytes = net.head_size(source)
            except Exception:  # noqa: BLE001 - нет сети / нет requests: размер уточним при скачивании
                size_bytes = None
            low = filename.lower()
            if low.endswith(".zip"):
                engine = ENGINE_VOSK
            elif low.startswith("ggml") or "ggml" in low:
                engine = ENGINE_WHISPERCPP
            else:
                engine = "faster-whisper"
            single_file = filename
    else:
        repo_id, subfolder, revision = split_repo_and_subfolder(source)
        if not net.is_hf_repo_id(repo_id):
            raise InstallError(
                f"Не понял источник «{source}».\n"
                "Поддерживаются: путь на диске, ID репозитория Hugging Face (owner/name), "
                "прямая ссылка https://…"
            )
        name = repo_id
        model_id = repo_id
        url = net.hf_repo_page(repo_id)
        try:
            files = net.hf_tree(repo_id, subfolder, revision=revision)
        except net.NetworkError as exc:
            raise InstallError(str(exc)) from exc
        names = [f.path.split("/")[-1].lower() for f in files]
        if any(n.startswith("ggml") for n in names):
            engine = ENGINE_WHISPERCPP
            ggml = [f for f in files if f.path.lower().endswith(".bin") and "ggml" in f.path.lower()]
            if len(ggml) == 1:
                single_file = ggml[0].path.split("/")[-1]
            size_bytes = sum(f.size for f in files if not f.is_dir and "ggml" in f.path.lower())
        elif "model.bin" in names:
            engine = "faster-whisper"
            size_bytes = sum(f.size for f in _filter_needed(files))
        elif any(n.endswith(".mdl") for n in names):
            engine = ENGINE_VOSK
            size_bytes = sum(f.size for f in files if not f.is_dir)
        else:
            engine = "faster-whisper"
            size_bytes = sum(f.size for f in files if not f.is_dir)

    entry = ModelEntry(
        id=sanitize_id(model_id or name or "custom-model"),
        name=name or model_id,
        engine=engine or "faster-whisper",
        source=source,
        subfolder=subfolder,
        revision=revision or "main",
        file=single_file,
        size_bytes=size_bytes,
        languages=languages or ["ru", "en"],
        languages_note="определяется моделью",
        description="Модель добавлена пользователем.",
        url=url,
        custom=True,
        tags=["своя модель"],
        quality=3,
        speed=3,
    )
    _guess_ram(entry)
    return entry


def _guess_ram(entry: ModelEntry) -> None:
    """Грубая оценка потребления памяти по размеру модели (если в каталоге её нет)."""
    if entry.ram_mb or not entry.size_bytes:
        return
    size_mb = entry.size_bytes / (1024 * 1024)
    if entry.engine == ENGINE_WHISPERCPP:
        entry.ram_mb = int(size_mb * 1.3 + 200)
    elif entry.engine == ENGINE_VOSK:
        entry.ram_mb = int(size_mb * 1.1 + 200)
    else:
        entry.ram_mb = int(size_mb * 1.25 + 350)


def load_meta(folder: Path) -> Dict:
    meta_file = Path(folder) / META_FILENAME
    if not meta_file.is_file():
        return {}
    try:
        with open(meta_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}
