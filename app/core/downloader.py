"""Model download and "installation" into the project `models/` folder.

Supported sources: Hugging Face repo (`owner/name`, optional subfolder),
direct file URL (ggml-*.bin, Vosk zip), local folder/file. Each installed
model gets `models/<id>/installed.json` so the GUI knows what is on disk.
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
from ..constants import (
    APP_VERSION,
    ENGINE_FASTER_WHISPER,
    ENGINE_VOSK,
    ENGINE_WHISPERCPP,
    MODELS_DIRNAME,
)
from ..i18n import _
from ..logutils import get_logger
from . import net
from .models_catalog import (
    META_FILENAME,
    ModelEntry,
    ct2_subfolder,
    detect_engine,
    engine_from_config,
)

log = get_logger("downloader")

ProgressCallback = Callable[[float, str], None]  # (fraction 0..1, status text)
CancelToken = Callable[[], bool]

#: marker of an INCOMPLETE install: created when the download starts and
#: removed only after full verification. While it exists (and installed.json
#: does not), the catalog does NOT treat the model as installed — a network
#: interruption must never leave a "half model" that the GUI shows as ready.
PARTIAL_MARKER = ".localasr-partial.json"

# Files not needed at runtime (saves gigabytes of traffic)
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
#  Source parsing
# --------------------------------------------------------------------------- #


def split_repo_and_subfolder(source: str) -> Tuple[str, str, str]:
    """Parse an HF model link into (repo, subfolder, revision).

    Supported: owner/name; owner/name/tree/<rev>/<sub>; owner/name:sub;
    https://huggingface.co/owner/name/tree/<rev>/<sub>.
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
            return text, "", "main"  # direct file URL

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
    """Return the install plan: what to download and from where.

    {"kind": "hf_repo"|"url_file"|"local", "files": [(rel, url, size)],
     "total_bytes": int}
    """
    source = (entry.source or "").strip()
    if not source:
        raise InstallError(_("Model \"{name}\" has no source specified").format(name=entry.name))

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

    raise InstallError(_(
        "Could not understand the model source: \"{source}\".\n"
        "Expected a Hugging Face repo ID (owner/name), a direct https://… URL "
        "or a path to a local folder/file."
    ).format(source=source))


def _plan_hf(repo_id: str, subfolder: str, entry: ModelEntry, revision: str = "main") -> Dict:
    try:
        remote = net.hf_tree(repo_id, subfolder, revision=revision)
    except net.NetworkError as exc:
        raise InstallError(str(exc)) from exc

    files = [f for f in remote if not f.is_dir]
    if not files:
        raise InstallError(_("No files in {repo}/{sub}").format(repo=repo_id, sub=subfolder or "main"))

    single_file = entry.file
    if single_file:
        matches = [f for f in files if f.path.endswith(single_file) or f.path.split("/")[-1] == single_file]
        if not matches:
            raise InstallError(_("File {f} not found in {repo}").format(f=single_file, repo=repo_id))
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
    """Drop redundant weight sets and service junk.

    Keep ONLY ONE weight set — the one the repo actually uses
    (CT2 model.bin -> safetensors -> pytorch_model.bin).
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
                continue  # CT2 weights present — skip the other sets
            if not has_model_bin and has_safetensors and not low.endswith(".safetensors"):
                continue  # safetensors present — skip bin/pt duplicates
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
#  Installation
# --------------------------------------------------------------------------- #


def _write_partial_marker(target_dir: Path, entry: ModelEntry, plan: Dict) -> None:
    """Mark the folder as "install incomplete" (see PARTIAL_MARKER)."""
    try:
        payload = {
            "id": entry.id,
            "name": entry.name,
            "source": entry.source,
            "kind": plan.get("kind"),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "app_version": APP_VERSION,
            "files": [rel for rel, _url, _size in plan.get("files") or []][:200],
        }
        net.write_json(target_dir / PARTIAL_MARKER, payload)
        log.debug("Partial-install marker written: %s", target_dir / PARTIAL_MARKER)
    except OSError as exc:
        log.debug("Failed to write the partial-install marker: %s", exc)


def _remove_partial_marker(target_dir: Path) -> None:
    try:
        (target_dir / PARTIAL_MARKER).unlink(missing_ok=True)
    except OSError:
        pass


def _verify_plan(target_dir: Path, files: List[Tuple[str, str, int]]) -> List[str]:
    """Compare downloaded files with the plan: list of problems (empty = OK)."""
    problems: List[str] = []
    for rel, _url, size in files:
        dest = target_dir / rel
        if not dest.is_file():
            problems.append(_("{rel}: missing").format(rel=rel))
            continue
        actual = dest.stat().st_size
        if size > 0 and actual != size:
            problems.append(_("{rel}: size {actual} != expected {size}").format(
                rel=rel, actual=actual, size=size))
    return problems


def install_model(
    entry: ModelEntry,
    root: Optional[Path] = None,
    progress: Optional[ProgressCallback] = None,
    cancel: Optional[CancelToken] = None,
) -> InstallResult:
    """Download the model into models/<id>/ and write installed.json.

    Resilient to interruptions: PARTIAL_MARKER until verified; fully
    downloaded files are skipped on reinstall; every file is size-checked
    against the plan at the end (a truncated file is an install error).
    """
    started = time.time()
    root = Path(root) if root else paths.ROOT_DIR
    models_root = root / MODELS_DIRNAME
    models_root.mkdir(parents=True, exist_ok=True)

    plan = resolve_source(entry)
    files: List[Tuple[str, str, int]] = plan["files"]
    total_bytes = plan.get("total_bytes") or sum(size for _, _, size in files) or entry.size_bytes or 0
    log.debug(
        "install_model: %s | kind=%s repo=%s subfolder=%r revision=%s | files: %d | size: %s",
        entry.id, plan["kind"], plan.get("repo", "—"), plan.get("subfolder", ""),
        plan.get("revision", "—"), len(files), paths.human_size(total_bytes),
    )
    for rel, _url, size in files[:80]:
        log.debug("  plan: %s (%s)", rel, paths.human_size(size))

    target_dir = models_root / sanitize_id(entry.id)
    if plan["kind"] == "local" and Path(plan["local"]).is_dir():
        source_dir = Path(plan["local"]).resolve()
        if _is_inside(source_dir, target_dir.resolve()):
            raise InstallError(_("A model cannot be installed into itself"))
        target_dir = source_dir if _is_inside(source_dir, models_root.resolve()) else target_dir

    paths.ensure_free_space(models_root, total_bytes or 0, margin_bytes=256 * 1024 * 1024)

    if progress:
        progress(0.0, _("Preparing: {name} ({size})").format(
            name=entry.name, size=paths.human_size(total_bytes)))

    target_dir.mkdir(parents=True, exist_ok=True)
    is_local_source = plan["kind"] == "local"
    if not is_local_source:
        _write_partial_marker(target_dir, entry, plan)
    downloaded = 0
    skipped: List[str] = []
    done_bytes = 0

    def report(current: int, total: int, speed: float, filename: str) -> None:
        nonlocal done_bytes
        done_bytes = current
        if progress and total_bytes:
            fraction = min(0.999, (downloaded + current) / max(1, total_bytes))
            speed_text = _(" · {}/s").format(paths.human_size(speed)) if speed else ""
            progress(fraction, _("{file} · {done} of {total}{speed}").format(
                file=filename, done=paths.human_size(downloaded + current),
                total=paths.human_size(total_bytes), speed=speed_text))

    try:
        if plan["kind"] == "local":
            for rel, src, size in files:
                if cancel and cancel():
                    raise net.DownloadCancelled(_("Install cancelled"))
                dest = target_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if Path(src).resolve() != dest.resolve():
                    shutil.copy2(src, dest)
                downloaded += size
                if progress and total_bytes:
                    progress(min(0.999, downloaded / total_bytes), _("Copying {rel}").format(rel=rel))
        elif plan["kind"] == "url_file" and str(files[0][0]).lower().endswith(".zip"):
            archive = target_dir / files[0][0]
            net.download_file(files[0][1], archive, progress=report, cancel=cancel)
            downloaded = archive.stat().st_size
            if progress:
                progress(0.999, _("Extracting the archive…"))
            net.extract_zip(archive, target_dir, progress=lambda p: None)
            archive.unlink(missing_ok=True)
            # Vosk extracts into a vosk-model-* subfolder — flatten it
            _flatten_single_subdir(target_dir)
        else:
            for rel, url, size in files:
                if cancel and cancel():
                    raise net.DownloadCancelled(_("Install cancelled"))
                dest = target_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                # resuming an interrupted install: skip complete files
                if size > 0 and dest.is_file() and dest.stat().st_size == size:
                    downloaded += size
                    skipped.append(rel)
                    log.debug("install_model: skipping already downloaded %s (%s)", rel, paths.human_size(size))
                    if progress and total_bytes:
                        progress(min(0.999, downloaded / total_bytes), _("Already downloaded: {rel}").format(rel=rel))
                    continue
                net.download_file(url, dest, progress=report, cancel=cancel)
                downloaded += size

            # verification: every planned file exists with the right size
            problems = _verify_plan(target_dir, files)
            if problems:
                listing = "\n".join("  • " + p for p in problems[:10])
                raise InstallError(_(
                    "Downloading \"{name}\" did not finish cleanly — "
                    "{n} file(s) are corrupted or incomplete:\n{listing}\n\n"
                    "The network was probably interrupted. Run the install again — "
                    "complete files will be skipped, only missing ones re-downloaded."
                ).format(name=entry.name, n=len(problems), listing=listing))

        engine = detect_engine(target_dir) or entry.engine
        if not engine:
            raise InstallError(_(
                "The downloaded files do not look like a model of any supported engine.\n"
                "Folder: {dir}\n"
                "Expected: model.bin+config.json (faster-whisper), ggml-*.bin (whisper.cpp) "
                "or am/final.mdl (Vosk)."
            ).format(dir=target_dir))

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
            "files": [rel for rel, _url, _size in files][:200],
        }
        net.write_json(target_dir / META_FILENAME, meta)
        # install complete and verified — the partial marker is no longer needed
        _remove_partial_marker(target_dir)

    except net.DownloadCancelled:
        _cleanup_partial(target_dir)
        log.warning(_("Install of %s cancelled by user (folder marked incomplete)"), entry.id)
        raise
    except InstallError:
        log.error(_("Install of %s did not finish: folder remains marked incomplete"), entry.id)
        raise
    except net.NetworkError as exc:
        log.error(_("Install of %s interrupted by a network error: %s"), entry.id, exc)
        raise InstallError(_("Network error: {exc}").format(exc=exc)) from exc
    except OSError as exc:
        log.error(_("Install of %s interrupted by a disk error: %s"), entry.id, exc)
        raise InstallError(_("Disk error: {exc}").format(exc=exc)) from exc

    duration = time.time() - started
    log.info(
        _("Model installed: %s -> %s (%s in %.1f s)"),
        entry.id, target_dir, paths.human_size(downloaded), duration,
    )
    if progress:
        progress(1.0, _("Done: {name} installed").format(name=entry.name))
    return InstallResult(
        entry=entry,
        directory=target_dir,
        downloaded_bytes=downloaded,
        skipped_files=skipped,
        duration_seconds=duration,
    )


def _flatten_single_subdir(folder: Path) -> None:
    """If extraction produced a single nested folder — lift its contents up."""
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
    """Delete incomplete .part files (the folder stays for resuming)."""
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
    """Model folder name: safe on all file systems."""
    cleaned = re.sub(r"[^\w.\-]+", "_", model_id.strip(), flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned[:80] or "model"


def uninstall_model(entry: ModelEntry, root: Optional[Path] = None) -> bool:
    """Delete the model from disk (only if it is inside models/)."""
    root = Path(root) if root else paths.ROOT_DIR
    folder = entry.install_dir or (root / MODELS_DIRNAME / sanitize_id(entry.id))
    folder = Path(folder)
    if not folder.exists():
        return False
    models_root = (root / MODELS_DIRNAME).resolve()
    try:
        folder.resolve().relative_to(models_root)
    except ValueError:
        log.warning(_("Refused: %s is outside models/ — not deleting"), folder)
        return False
    shutil.rmtree(folder, ignore_errors=True)
    return not folder.exists()


# --------------------------------------------------------------------------- #
#  Model autodetection from an arbitrary link/path ("custom model")
# --------------------------------------------------------------------------- #


def _engine_from_hf_listing(repo_id: str, subfolder: str, revision: str, names: List[str]) -> str:
    """Engine guess from an HF repo file listing (+ remote config.json).

    This is how "Custom model" understands ANY Hugging Face ASR model:
    an unknown config.json with an audio architecture goes to the generic
    hf-transformers engine.
    """
    if any(n.startswith("ggml") for n in names):
        return ENGINE_WHISPERCPP
    if "model.bin" in names:
        return ENGINE_FASTER_WHISPER
    if any(n == "final.mdl" or n.endswith(".mdl") for n in names):
        return ENGINE_VOSK
    if "config.json" in names:
        cfg = net.hf_read_json(repo_id, "config.json", revision=revision, subfolder=subfolder)
        engine = engine_from_config(cfg)
        if engine:
            return engine
    return ENGINE_FASTER_WHISPER




def inspect_custom_source(source: str, root: Optional[Path] = None) -> ModelEntry:
    """Build a ModelEntry from an arbitrary source: path, HF repo or URL.

    Nothing is downloaded — only type, size and name detection.
    """
    source = (source or "").strip()
    if not source:
        raise InstallError(_("Empty model source"))

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
                files = net.hf_tree(repo_id, subfolder, revision=revision)
                names = [f.path.split("/")[-1].lower() for f in files]
                engine = _engine_from_hf_listing(repo_id, subfolder, revision, names)
                if engine == ENGINE_WHISPERCPP:
                    size_bytes = sum(
                        f.size for f in files if not f.is_dir and "ggml" in f.path.lower()
                    )
                else:
                    size_bytes = sum(f.size for f in _filter_needed(files))
            except net.NetworkError as exc:
                raise InstallError(str(exc)) from exc
        else:
            filename = unquote(os.path.basename(parsed.path)) or "model.bin"
            name = Path(filename).stem
            model_id = Path(filename).stem
            url = source
            try:
                size_bytes = net.head_size(source)
            except Exception:  # noqa: BLE001 - offline / no requests: size refined during download
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
            raise InstallError(_(
                "Could not understand the source \"{source}\".\n"
                "Supported: a disk path, a Hugging Face repo ID (owner/name), "
                "a direct https://… URL"
            ).format(source=source))
        name = repo_id
        model_id = repo_id
        url = net.hf_repo_page(repo_id)
        try:
            files = net.hf_tree(repo_id, subfolder, revision=revision)
        except net.NetworkError as exc:
            raise InstallError(str(exc)) from exc
        names = [f.path.split("/")[-1].lower() for f in files]
        engine = _engine_from_hf_listing(repo_id, subfolder, revision, names)
        if engine == ENGINE_WHISPERCPP:
            ggml = [f for f in files if f.path.lower().endswith(".bin") and "ggml" in f.path.lower()]
            if len(ggml) == 1:
                single_file = ggml[0].path.split("/")[-1]
            size_bytes = sum(f.size for f in files if not f.is_dir and "ggml" in f.path.lower())
        elif engine in (ENGINE_FASTER_WHISPER,):
            size_bytes = sum(f.size for f in _filter_needed(files))
        else:
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
        languages_note="detected by the model",
        description="Model added by the user.",
        url=url,
        custom=True,
        tags=["custom model"],
        quality=3,
        speed=3,
    )
    _guess_ram(entry)
    return entry


def _guess_ram(entry: ModelEntry) -> None:
    """Rough RAM estimate from the model size (when the catalog has none)."""
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
