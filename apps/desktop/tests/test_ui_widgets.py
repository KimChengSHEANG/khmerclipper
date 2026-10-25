from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QKeySequence
from PIL import Image
from PySide6.QtWidgets import QDialog, QLabel, QMessageBox, QPushButton

from helpers import FIXTURES
from khmerclipper.history import HistoryStore, ScanRecord
from khmerclipper.prefs import Prefs
from khmerclipper.ui.feedback_dialog import FeedbackDialog
from khmerclipper.ui.fonts import khmer_font
from khmerclipper.ui.history_window import HistoryWindow
from khmerclipper.ui.image_window import ImageOCRWindow
from khmerclipper.ui.popup import ResultPopup
from khmerclipper.ui.progress_panel import ProgressPanel
from khmerclipper.ui.report_dialog import ReportDialog
from khmerclipper.ui.prefs_window import WAYLAND_HINT, PrefsWindow
from khmerclipper.ui.worker import Task


def test_khmer_font_is_bundled_noto(qapp):
    assert khmer_font().families()[0] == "Noto Sans Khmer"


def test_worker_reports_progress_and_result(qtbot):
    # Connect before starting: a queued cross-thread signal emitted with no receiver is lost.
    seen = []
    task = Task(lambda progress, cancel: (progress("step"), 42)[1])
    task.signals.progress.connect(seen.append)
    with qtbot.waitSignal(task.signals.done, timeout=3000) as blocker:
        QThreadPool.globalInstance().start(task)
    assert blocker.args == [42]
    assert seen == ["step"]


def test_worker_reports_failure(qtbot):
    def boom(progress, cancel):
        raise ValueError("bad")
    task = Task(boom)
    with qtbot.waitSignal(task.signals.failed, timeout=3000) as blocker:
        QThreadPool.globalInstance().start(task)
    assert isinstance(blocker.args[0], ValueError)


def test_popup_shows_plain_text_and_copies_full_text(qtbot):
    popup = ResultPopup()
    qtbot.addWidget(popup)
    popup.show_text("<b>ក</b>")
    assert popup.label.text() == "<b>ក</b>"
    assert popup.label.textFormat() == Qt.TextFormat.PlainText   # OCR text is never rendered as HTML
    long_text = "ក" * 2000
    popup.show_text(long_text)
    with qtbot.waitSignal(popup.copy_requested) as blocker:
        popup.copy_button.click()
    assert blocker.args == [long_text]


def test_popup_states(qtbot):
    popup = ResultPopup()
    qtbot.addWidget(popup)
    popup.show_text("ក", copied=False)
    assert popup.title.text() == "Done" and popup.copy_button.isVisibleTo(popup)
    with qtbot.waitSignal(popup.copy_requested):
        popup.copy_button.click()
    assert popup.copy_button.text() == "Copied ✓"   # the click says it worked
    qtbot.waitUntil(lambda: popup.copy_button.text() == "Copy", timeout=3000)
    popup.show_error("boom", reported=True)
    assert popup.title.text() == "Capture failed" and "crash report" in popup.label.text()
    assert popup.feedback_button.isVisibleTo(popup) and not popup.copy_button.isVisibleTo(popup)
    with qtbot.waitSignal(popup.feedback_requested):
        popup.feedback_button.click()
    popup.show_no_text()
    assert popup.kind == "notext" and not popup.feedback_button.isVisibleTo(popup)


def test_progress_panel_waits_before_showing(qtbot):
    quick = ProgressPanel("Reading text…")
    qtbot.addWidget(quick)
    quick.close()  # finished within the delay
    qtbot.wait(ProgressPanel.SHOW_DELAY_MS + 100)
    assert not quick.isVisible()
    slow = ProgressPanel("Reading text…")
    qtbot.addWidget(slow)
    qtbot.waitUntil(slow.isVisible, timeout=2000)
    with qtbot.waitSignal(slow.canceled):
        slow.cancel_button.click()
    assert slow.label.text() == "Cancelling…" and not slow.cancel_button.isEnabled()


def test_popup_message_hides_copy(qtbot):
    popup = ResultPopup()
    qtbot.addWidget(popup)
    popup.show_message("No text found")
    assert popup.copy_button.isHidden()
    assert popup.isVisible()


def test_history_window_copy_delete_clear(qtbot, tmp_path, monkeypatch):
    store = HistoryStore(tmp_path / "h.json")
    store.add(ScanRecord("old"))
    store.add(ScanRecord("ថ្មី"))
    copied = []
    window = HistoryWindow(store, copied.append)
    qtbot.addWidget(window)
    window.refresh()
    assert window.list.count() == 2
    window.list.setCurrentRow(0)
    window.copy_selected()
    assert copied == ["ថ្មី"]
    window.delete_selected()
    assert [r.text for r in store.records] == ["old"]
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    window.clear_all()
    assert len(store.records) == 1
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window.clear_all()
    assert store.records == [] and window.list.count() == 0


def test_prefs_window_saves_changes(qtbot, tmp_path):
    prefs = Prefs.load(tmp_path / "prefs.json")
    window = PrefsWindow(prefs, trocr_available=True, hotkey_supported=True)
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.changed) as blocker:
        window.checks["auto_copy"].setChecked(False)
    assert blocker.args == ["auto_copy"]
    assert Prefs.load(tmp_path / "prefs.json").auto_copy is False
    window.engine.setCurrentIndex(window.engine.findData("tesseract"))
    assert Prefs.load(tmp_path / "prefs.json").engine == "tesseract"


def test_prefs_window_disables_unpublished_trocr_and_wayland_hotkey(qtbot, tmp_path):
    window = PrefsWindow(Prefs.load(tmp_path / "p.json"), trocr_available=False, hotkey_supported=False)
    qtbot.addWidget(window)
    assert not window.engine.model().item(window.engine.findData("trocr")).isEnabled()
    assert not window.hotkey.isEnabled()
    assert any(label.text() == WAYLAND_HINT for label in window.findChildren(QLabel))


def test_prefs_window_hotkey_validation(qtbot, tmp_path, monkeypatch):
    prefs = Prefs.load(tmp_path / "p.json")
    window = PrefsWindow(prefs, trocr_available=True, hotkey_supported=True)
    qtbot.addWidget(window)
    window.hotkey.setKeySequence(QKeySequence("Ctrl+Alt+K"))
    window._hotkey_edited()
    assert prefs.hotkey == "<ctrl>+<alt>+k"
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    window.hotkey.setKeySequence(QKeySequence("K"))
    window._hotkey_edited()
    assert prefs.hotkey == "<ctrl>+<alt>+k" and warned


def test_image_window_open_path(qtbot, tmp_path):
    window = ImageOCRWindow()
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.image_chosen) as blocker:
        window.open_path(str(FIXTURES / "1.png"))
    assert blocker.args[0].mode == "RGB"
    bad = tmp_path / "notes.txt"
    bad.write_text("hello", encoding="utf-8")
    with qtbot.waitSignal(window.failed) as blocker:
        window.open_path(str(bad))
    assert "notes.txt" in blocker.args[0]


def _store_with_capture(tmp_path):
    store = HistoryStore(tmp_path / "h.json")
    store.add(ScanRecord("plain")); rec = ScanRecord("ក")
    store.add(rec, image=Image.new("RGB", (8, 4)), meta={"ocr_text": "ក"})
    return store, rec


def _report_button(w, row):
    return w.list.itemWidget(w.list.item(row)).findChild(QPushButton, "report_button")


def test_history_rows_show_report_only_for_captures(qtbot, tmp_path):
    store, _ = _store_with_capture(tmp_path)
    w = HistoryWindow(store, lambda t: None, report=lambda r, c: None); qtbot.addWidget(w); w.refresh()
    assert _report_button(w, 0).text() == "Report" and _report_button(w, 1) is None


def test_report_flow_marks_reported(qtbot, tmp_path, monkeypatch):
    store, rec = _store_with_capture(tmp_path)
    reported = []
    w = HistoryWindow(store, lambda t: None, report=lambda r, c: reported.append((r.id, c)))
    qtbot.addWidget(w); w.refresh()
    monkeypatch.setattr(ReportDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(ReportDialog, "correction_text", lambda self: "ខ")
    w.report_record(rec)
    assert reported == [(rec.id, "ខ")] and store.is_flagged(rec.id)
    button = _report_button(w, 0)
    assert button.text() == "Reported" and not button.isEnabled()


def test_report_failure_leaves_row_unreported(qtbot, tmp_path, monkeypatch):
    store, rec = _store_with_capture(tmp_path)

    def fail(r, c):
        raise OSError("disk full")
    w = HistoryWindow(store, lambda t: None, report=fail); qtbot.addWidget(w); w.refresh()
    monkeypatch.setattr(ReportDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    w.report_record(rec)
    assert not store.is_flagged(rec.id) and _report_button(w, 0).text() == "Report"


def test_report_dialog_unchanged_text_means_no_correction(qtbot, tmp_path):
    png = tmp_path / "c.png"; Image.new("RGB", (8, 4)).save(png)
    d = ReportDialog(png, "ក"); qtbot.addWidget(d)
    assert d.correction_text() == ""
    d.correction.setPlainText("ខ")
    assert d.correction_text() == "ខ"


def test_feedback_dialog_requires_message(qtbot):
    d = FeedbackDialog(); qtbot.addWidget(d)
    assert not d.send_button.isEnabled()
    d.message.setPlainText("  ")
    assert not d.send_button.isEnabled()
    d.message.setPlainText("សូមបន្ថែម")
    assert d.send_button.isEnabled()
    d.email.setText(" a@b.c ")
    assert d.values() == ("សូមបន្ថែម", "a@b.c")
