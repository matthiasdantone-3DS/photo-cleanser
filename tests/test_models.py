import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from config import ModelIntegrityError, verify_model, write_lock_checksum, write_lock_url
from models import download_models as dm


# Crea un lock file di prova con un modello e il checksum indicato
def make_lock(tmp_path: Path, sha256: str = "") -> Path:
    lock = tmp_path / "models.lock.json"
    lock.write_text(
        json.dumps({"models": {"demo": {"file": "demo.bin", "url": "https://example.org/demo.bin", "sha256": sha256}}}),
        encoding="utf-8",
    )
    return lock


# Un modello senza checksum viene rifiutato, salvo --allow-unverified
def test_verify_requires_checksum(tmp_path: Path) -> None:
    model = tmp_path / "demo.bin"
    model.write_bytes(b"pesi")
    lock = make_lock(tmp_path)
    with pytest.raises(ModelIntegrityError):
        verify_model(model, "demo", lock_path=lock)
    verify_model(model, "demo", allow_unverified=True, lock_path=lock)


# Il checksum giusto passa, quello sbagliato no
def test_verify_checks_hash(tmp_path: Path) -> None:
    model = tmp_path / "demo.bin"
    model.write_bytes(b"pesi")
    good = hashlib.sha256(b"pesi").hexdigest()
    verify_model(model, "demo", lock_path=make_lock(tmp_path, good))
    with pytest.raises(ModelIntegrityError):
        verify_model(model, "demo", lock_path=make_lock(tmp_path, "0" * 64))


# Un modello assente solleva FileNotFoundError
def test_verify_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_model(tmp_path / "manca.bin", "demo", lock_path=make_lock(tmp_path))


# Checksum e URL si scrivono nel lock senza perdere gli altri campi
def test_write_lock(tmp_path: Path) -> None:
    lock = make_lock(tmp_path)
    write_lock_checksum("demo", "a" * 64, "nota", lock_path=lock)
    write_lock_url("demo", "https://example.org/nuovo.bin", lock_path=lock)
    entry = json.loads(lock.read_text(encoding="utf-8"))["models"]["demo"]
    assert entry["sha256"] == "a" * 64
    assert entry["sha256_note"] == "nota"
    assert entry["url"] == "https://example.org/nuovo.bin"
    assert entry["file"] == "demo.bin"


# Le coppie di --pin e --url vengono controllate
def test_parse_pairs() -> None:
    known = {"yunet", "logo"}
    assert dm.parse_pairs(["yunet=" + "A" * 64], known, "sha256") == {"yunet": "a" * 64}
    assert dm.parse_pairs(["logo=https://example.org/logo.pt"], known, "url") == {
        "logo": "https://example.org/logo.pt"
    }
    for bad in (["yunet"], ["altro=" + "a" * 64], ["yunet=corto"]):
        with pytest.raises(ValueError):
            dm.parse_pairs(bad, known, "sha256")
    with pytest.raises(ValueError):
        dm.parse_pairs(["logo=http://example.org/logo.pt"], known, "url")


# Prepara il download simulato: scrive contenuto noto e restituisce il suo SHA-256
def patch_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: bytes) -> list[str]:
    calls: list[str] = []

    def fake_download(url: str, target: Path) -> str:
        calls.append(url)
        target.write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    monkeypatch.setattr(dm, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(dm, "download_to", fake_download)
    return calls


# Download con checksum giusto: il file arriva, nessun residuo temporaneo
def test_fetch_model_verified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_download(monkeypatch, tmp_path, b"pesi")
    entry = {"file": "demo.bin", "url": "https://example.org/demo.bin", "sha256": hashlib.sha256(b"pesi").hexdigest()}
    dm.fetch_model("demo", entry, allow_unverified=False)
    assert (tmp_path / "demo.bin").read_bytes() == b"pesi"
    assert not list(tmp_path.glob("*.part"))


# Checksum sbagliato: il download fallisce e il file viene eliminato
def test_fetch_model_mismatch_deletes_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_download(monkeypatch, tmp_path, b"pesi")
    entry = {"file": "demo.bin", "url": "https://example.org/demo.bin", "sha256": "0" * 64}
    with pytest.raises(RuntimeError):
        dm.fetch_model("demo", entry, allow_unverified=False)
    assert not list(tmp_path.iterdir())


# Senza checksum il download viene rifiutato prima ancora di partire
def test_fetch_model_refuses_without_checksum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_download(monkeypatch, tmp_path, b"pesi")
    entry = {"file": "demo.bin", "url": "https://example.org/demo.bin", "sha256": ""}
    with pytest.raises(RuntimeError):
        dm.fetch_model("demo", entry, allow_unverified=False)
    assert calls == []


# Con --allow-unverified il checksum calcolato viene registrato nel lock
def test_fetch_model_trust_on_first_use(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_download(monkeypatch, tmp_path, b"pesi")
    recorded: dict[str, str] = {}
    monkeypatch.setattr(dm, "write_lock_checksum", lambda key, checksum, note: recorded.update({key: checksum}))
    entry = {"file": "demo.bin", "url": "https://example.org/demo.bin", "sha256": ""}
    dm.fetch_model("demo", entry, allow_unverified=True)
    assert recorded == {"demo": hashlib.sha256(b"pesi").hexdigest()}
    assert (tmp_path / "demo.bin").is_file()


# Il lock del progetto contiene le voci attese e nessun checksum finto
def test_project_lock_is_well_formed() -> None:
    lock = json.loads(config.LOCK_FILE.read_text(encoding="utf-8"))
    assert {"yunet", "logo"} <= set(lock["models"])
    assert lock["models"]["yunet"]["url"].startswith("https://")
