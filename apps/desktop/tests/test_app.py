import json
import threading

import pytest
from PIL import Image

from khmerclipper.errors import Cancelled, KhmerClipperError
from khmerclipper.history import HistoryStore
from khmerclipper.ocr import trocr_models
from khmerclipper.prefs import Prefs
from khmerclipper.ui import app as app_module
from khmerclipper.ui.app import Controller, TrayApp, preferred_qpa


@pytest.fixture
def controller(qtbot, tmp_path):
    copied = []
    prefs = Prefs.load(tmp_path / "prefs.json")
    prefs.engine = "tesseract"  # the TrOCR default would prompt for (or start) a real model download
    c = Controller(prefs, HistoryStore(tmp_path / "h.json"), set_clipboard=copied.append)
    c.copied = copied
    qtbot.addWidget(c.popup)
    return c


def test_handle_text_copies_records_and_shows(controller):
    controller.handle_text("ក  \nខ")
    assert controller.copied == ["ក  \nខ"]
    assert controller.history.records[0].text == "ក  \nខ"
    assert controller.popup.isVisible() and controller.popup.label.text() == "ក  \nខ"
    assert controller.popup.title.text() == "Copied"


def test_handle_text_respects_prefs(controller):
    controller.prefs.preserve_linebreaks = False
    controller.prefs.auto_copy = False
    controller.prefs.show_result = False
    controller.handle_text("ក\nខ")
    assert controller.copied == []
    assert controller.history.records[0].text == "ក ខ"
    assert not controller.popup.isVisible()


def test_empty_result_is_not_recorded(controller):
    controller.handle_text("  \n ")
    assert controller.history.records == []
    assert controller.popup.kind == "notext" and controller.popup.title.text() == "No text found"


def test_errors(controller):
    controller.handle_error(Cancelled())
    assert not controller.popup.isVisible()
    controller.handle_error(KhmerClipperError("Tesseract was not found."))
    assert "Tesseract was not found." in controller.popup.label.text()
    assert controller.popup.kind == "error" and controller.popup.feedback_button.isVisibleTo(controller.popup)


def test_run_ocr_end_to_end_off_thread(controller, qtbot, monkeypatch):
    monkeypatch.setattr(app_module, "recognize", lambda image, engine, progress=None, cancel=None: "សួស្តី")
    controller.run_ocr(Image.new("RGB", (20, 20), "white"))
    qtbot.waitUntil(lambda: bool(controller.history.records), timeout=3000)
    assert controller.history.records[0].text == "សួស្តី"
    assert controller.copied == ["សួស្តី"]


def test_submit_delivers_results_on_gui_thread(controller, qtbot):
    # Callbacks touch widgets, so they must run on the GUI thread, never the pool thread.
    seen = []
    controller.submit(lambda progress, cancel: (progress("p"), 1)[1],
                      on_done=lambda r: seen.append(threading.current_thread() is threading.main_thread()))
    qtbot.waitUntil(lambda: bool(seen), timeout=3000)
    assert seen == [True]


def test_preferred_qpa():
    assert preferred_qpa("linux", {"XDG_SESSION_TYPE": "wayland", "DISPLAY": ":0"}) == "xcb"
    assert preferred_qpa("linux", {"DISPLAY": ":0", "QT_QPA_PLATFORM": "wayland"}) is None  # user choice wins
    assert preferred_qpa("linux", {"WAYLAND_DISPLAY": "wayland-0"}) is None  # no XWayland available
    assert preferred_qpa("win32", {"DISPLAY": ":0"}) is None


def test_trocr_unpublished_explains_instead_of_crashing(controller, monkeypatch):
    monkeypatch.delenv("KHMERCLIPPER_TROCR_DIR", raising=False)
    monkeypatch.setattr(trocr_models, "load_manifest", lambda path=None: trocr_models.Manifest("khmer-v1", ()))
    controller.prefs.engine = "trocr"
    controller.run_ocr(Image.new("RGB", (20, 20), "white"))
    assert "isn't published" in controller.popup.label.text()
    assert controller.history.records == []


def test_startup_downloads_missing_trocr_model_once(controller, qtbot, monkeypatch):
    monkeypatch.delenv("KHMERCLIPPER_TROCR_DIR", raising=False)
    manifest = trocr_models.Manifest("khmer-v3", (trocr_models.RemoteFile("a", "u", "h", 5_000_000),))
    monkeypatch.setattr(trocr_models, "load_manifest", lambda path=None: manifest)
    monkeypatch.setattr(trocr_models, "is_installed", lambda m, root=None: False)
    release, installs = threading.Event(), []

    def install(m, root=None, progress=None, cancel=None):
        installs.append(m)
        release.wait(3)
        raise KhmerClipperError("offline")  # a failed background download stays silent

    monkeypatch.setattr(trocr_models, "install", install)
    assert controller.download_trocr_model() == 0  # Tesseract chosen: nothing to fetch
    controller.prefs.engine = "trocr"
    assert controller.download_trocr_model() == 5
    assert controller.download_trocr_model() == 0  # already running
    controller.run_ocr(Image.new("RGB", (20, 20), "white"))
    assert "still downloading" in controller.popup.label.text()
    release.set()
    qtbot.waitUntil(lambda: controller.model_download is None, timeout=3000)
    assert len(installs) == 1 and "still downloading" in controller.popup.label.text()


def test_tray_menu_and_commands(controller, monkeypatch):
    monkeypatch.setattr(app_module.HotkeyListener, "start", lambda self, *a, **k: False)
    controller.prefs.check_updates = False
    controller.prefs.welcomed = True
    captures = []
    monkeypatch.setattr(controller, "start_capture", lambda: captures.append(1))
    tray = TrayApp(controller)
    texts = [a.text() for a in tray.menu.actions() if a.text()]
    assert texts == ["Capture Text", "OCR Image File…", "History…", "Send Feedback…", "Preferences…",
                     "Check for Updates…", "Quit KhmerClipper"]
    tray.on_command("capture")
    assert captures == [1]
    tray.on_command("show")
    assert tray.prefs_window is not None and tray.prefs_window.isVisible()


def test_keep_captures_saves_image_with_ocr_details(controller):
    image = Image.new("RGB", (20, 20), "white")
    controller.handle_text("ក\nខ", image=image)
    assert controller.history.has_capture(controller.history.records[0].id)  # on by default

    controller.prefs.keep_captures = False
    controller.handle_text("ក\nខ", image=image)
    assert not controller.history.has_capture(controller.history.records[0].id)

    controller.prefs.keep_captures = True
    controller.prefs.preserve_linebreaks = False
    controller.handle_text("ក\nខ", image=image)
    record = controller.history.records[0]
    assert controller.history.has_capture(record.id)
    meta = json.loads((controller.history.captures_dir / f"{record.id}.json").read_text(encoding="utf-8"))
    assert meta["ocr_text"] == "ក\nខ" and record.text == "ក ខ"  # raw OCR kept, not the formatted text
    assert meta["engine"] == "tesseract" and meta["preserve_linebreaks"] is False


def test_unexpected_error_is_recorded(controller, monkeypatch):
    seen = []
    monkeypatch.setattr(app_module.crash, "record", lambda exc, engine: seen.append(exc))
    controller.handle_error(ZeroDivisionError())
    assert len(seen) == 1
    controller.handle_error(KhmerClipperError("x"))
    controller.handle_error(Cancelled())
    assert len(seen) == 1
    controller.prefs.send_crash_reports = False
    controller.handle_error(ZeroDivisionError())
    assert len(seen) == 1


def test_error_popup_shows_even_if_recording_fails(controller, monkeypatch):
    monkeypatch.setattr(app_module.crash, "record", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    controller.handle_error(ZeroDivisionError())
    assert controller.popup.title.text() == "Capture failed" and "ZeroDivisionError" in controller.popup.label.text()
    assert "crash report" not in controller.popup.label.text()  # recording failed, so don't promise one


def test_safe_flush_swallows_errors(monkeypatch):
    monkeypatch.setattr(app_module.reports, "flush", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app_module._safe_flush() == 0
    monkeypatch.setattr(app_module.reports, "flush", lambda: 3)
    assert app_module._safe_flush() == 3


@pytest.mark.parametrize("error", [ValueError("corrupt png"), OSError("disk full")])
def test_report_send_failure_tells_the_user(controller, monkeypatch, error):
    monkeypatch.setattr(app_module.HotkeyListener, "start", lambda self, *a, **k: False)
    controller.prefs.check_updates = False
    controller.prefs.welcomed = True
    tray = TrayApp(controller)
    controller.handle_text("ក", image=Image.new("RGB", (20, 20), "white"))
    record = controller.history.records[0]
    monkeypatch.setattr(app_module.reports, "capture_payload", lambda *a, **k: (_ for _ in ()).throw(error))
    with pytest.raises(type(error)):  # the History window must not mark the row Reported
        tray._report(record, "")
    assert controller.popup.title.text() == "Couldn't save the report"
