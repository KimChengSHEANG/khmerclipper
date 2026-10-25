"""Tray app wiring: capture -> OCR (worker thread) -> clipboard/history/popup."""

import os
import sys
import time
from collections.abc import Callable, Mapping
from functools import partial

from PySide6.QtCore import QObject, QThreadPool, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication, QIcon
from PySide6.QtWidgets import QApplication, QDialog, QMenu, QMessageBox, QSystemTrayIcon

from khmerclipper import __version__, autostart, crash, reports, updater
from khmerclipper.capture import portal
from khmerclipper.capture.geometry import capture_backend
from khmerclipper.capture.overlay import RegionSelector
from khmerclipper.errors import Cancelled, KhmerClipperError
from khmerclipper.formatter import apply
from khmerclipper.history import HistoryStore, ScanRecord
from khmerclipper.hotkey import HotkeyListener, pynput_to_qt
from khmerclipper.ocr import Engine, recognize, trocr_models
from khmerclipper.paths import APP_NAME, resource_dir
from khmerclipper.prefs import Prefs
from khmerclipper.single_instance import CommandServer, send_command
from khmerclipper.ui.feedback_dialog import FeedbackDialog
from khmerclipper.ui.history_window import HistoryWindow
from khmerclipper.ui.image_window import ImageOCRWindow
from khmerclipper.ui.popup import ResultPopup
from khmerclipper.ui.prefs_window import PrefsWindow
from khmerclipper.ui.progress_panel import ProgressPanel
from khmerclipper.ui.worker import Task


def _update_progress(dialog: ProgressPanel, value: object) -> None:
    if isinstance(value, tuple):  # (bytes_done, bytes_total) from a download
        done, total = value
        dialog.setValue(int(done * 1000 / max(1, total)))
    else:
        dialog.setLabelText(str(value))


def _present(window) -> None:
    window.show()
    window.raise_()
    window.activateWindow()


def preferred_qpa(platform: str, env: Mapping[str, str]) -> str | None:
    """Run Qt under XWayland when available: a native-Wayland background app can't reliably
    set the clipboard (no focused surface) or position its popup. Capture still goes through
    the portal because capture_backend() reads XDG_SESSION_TYPE, not the Qt platform."""
    if platform.startswith("linux") and not env.get("QT_QPA_PLATFORM") and env.get("DISPLAY"):
        return "xcb"
    return None


class _Relay(QObject):
    """Receives one Task's signals on the GUI thread. Bound @Slot methods of a QObject that
    lives on the GUI thread are guaranteed queued delivery; bare closures are not."""

    def __init__(self, on_progress, on_done, on_failed):
        super().__init__()
        self._on_progress, self._on_done, self._on_failed = on_progress, on_done, on_failed

    @Slot(object)
    def progress(self, value) -> None:
        self._on_progress(value)

    @Slot(object)
    def done(self, result) -> None:
        self._on_done(result)

    @Slot(object)
    def failed(self, error) -> None:
        self._on_failed(error)


class Controller(QObject):
    capture_requested = Signal()  # emitted from the hotkey thread / IPC; delivered on the GUI thread

    def __init__(self, prefs: Prefs, history: HistoryStore, set_clipboard: Callable[[str], None] | None = None):
        super().__init__()
        self.prefs, self.history = prefs, history
        self._set_clipboard = set_clipboard or (lambda text: QGuiApplication.clipboard().setText(text))
        self.popup = ResultPopup()
        self.popup.copy_requested.connect(self.copy)
        self.selector = RegionSelector()
        self.selector.selected.connect(self.run_ocr)
        self._tasks: dict[Task, _Relay] = {}
        self.model_download: Task | None = None  # startup TrOCR download, while running
        self.capture_requested.connect(self.start_capture)

    # --- capture -------------------------------------------------------------
    def start_capture(self) -> None:
        if self.prefs.engine == Engine.TROCR:
            from khmerclipper.ocr import trocr

            self.submit(lambda progress, cancel: trocr.preload())  # warm up while the user selects
        if capture_backend() == "portal":
            self.submit(lambda progress, cancel: portal.screenshot(), on_done=self._portal_done)
        else:
            self.selector.start()

    def _portal_done(self, image) -> None:
        if image is not None:  # None = user cancelled the portal dialog
            self.run_ocr(image)

    # --- OCR -----------------------------------------------------------------
    def run_ocr(self, image) -> None:
        engine = Engine(self.prefs.engine)
        if engine is Engine.TROCR and not self._trocr_ready(image):
            return
        dialog = self._progress_dialog("Reading text…")
        self.submit(lambda progress, cancel: recognize(image, engine, progress, cancel),
                    on_done=lambda text: self.handle_text(text, image=image), dialog=dialog)

    def _trocr_ready(self, image) -> bool:
        """True if TrOCR can run now. Otherwise explains or offers the one-time download
        (which re-runs this OCR when finished) and returns False."""
        manifest = trocr_models.load_manifest()
        if os.environ.get("KHMERCLIPPER_TROCR_DIR") or trocr_models.is_installed(manifest):
            return True
        if self.model_download is not None:
            self.popup.show_message("The TrOCR model is still downloading. Try again in a few minutes.")
            return False
        if not manifest.available:
            self.popup.show_message("The TrOCR model isn't published yet. Switch to Tesseract in Preferences.")
            return False
        size_mb = max(1, manifest.total_size // 1_000_000)
        answer = QMessageBox.question(None, APP_NAME, f"TrOCR needs a one-time {size_mb} MB download. Download now?")
        if answer != QMessageBox.StandardButton.Yes:
            return False
        dialog = self._progress_dialog("Downloading TrOCR model…", maximum=1000)
        self.submit(lambda progress, cancel: trocr_models.install(
                        manifest, progress=lambda done, total: progress((done, total)), cancel=cancel),
                    on_done=lambda _dir: self.run_ocr(image), dialog=dialog)
        return False

    def download_trocr_model(self) -> int:
        """Start fetching the TrOCR model in the background (no prompt) when it's the chosen
        engine and missing. Returns the download size in MB, or 0 if nothing was started.
        A failure (e.g. offline) is silent: the next capture offers the download again."""
        manifest = trocr_models.load_manifest()
        if (self.prefs.engine != Engine.TROCR or self.model_download is not None or not manifest.available
                or os.environ.get("KHMERCLIPPER_TROCR_DIR") or trocr_models.is_installed(manifest)):
            return 0

        def fetch(progress, cancel) -> None:
            try:
                trocr_models.install(manifest, cancel=cancel)
            except KhmerClipperError:  # includes Cancelled
                pass

        def done(_result) -> None:
            self.model_download = None

        self.model_download = self.submit(fetch, on_done=done)
        return max(1, manifest.total_size // 1_000_000)

    def handle_text(self, text: str, kind: str = "text", image=None) -> None:
        if not text.strip():
            self.popup.show_no_text()
            return
        formatted = apply(text, self.prefs.preserve_linebreaks)
        if self.prefs.auto_copy:
            self.copy(formatted)
        record = ScanRecord(formatted, kind)
        if self.prefs.keep_captures and image is not None:
            self.history.add(record, image=image, meta=self._capture_meta(text, record))
        else:
            self.history.add(record)
        if self.prefs.show_result:
            self.popup.show_text(formatted, copied=self.prefs.auto_copy)

    def _capture_meta(self, text: str, record: ScanRecord) -> dict:
        meta = {"ocr_text": text, "engine": self.prefs.engine, "app_version": __version__,
                "date": record.date, "preserve_linebreaks": self.prefs.preserve_linebreaks}
        if self.prefs.engine == Engine.TROCR:
            meta["model_version"] = trocr_models.load_manifest().version
        return meta

    def handle_error(self, error: BaseException) -> None:
        if isinstance(error, Cancelled):
            return
        reported = False
        if isinstance(error, KhmerClipperError):
            message = str(error)
        else:
            message = f"Unexpected error: {error!r}"
            if self.prefs.send_crash_reports:
                try:
                    reported = crash.record(error, self.prefs.engine) is not None
                except Exception:  # a reporting failure must not hide the popup
                    pass
        self.popup.show_error(message, reported=reported)

    def copy(self, text: str) -> None:
        self._set_clipboard(text)

    # --- background work -----------------------------------------------------
    def _progress_dialog(self, label: str, maximum: int = 0) -> ProgressPanel:
        return ProgressPanel(label, maximum)

    def submit(self, fn, on_done: Callable[[object], None] | None = None,
               dialog: ProgressPanel | None = None) -> Task:
        task = Task(fn)

        def finish() -> None:
            self._tasks.pop(task, None)
            if dialog is not None:
                dialog.close()
                dialog.deleteLater()

        def progress(value) -> None:
            if dialog is not None:
                _update_progress(dialog, value)

        def done(result) -> None:
            finish()
            if on_done is not None:
                on_done(result)

        def failed(error) -> None:
            finish()
            self.handle_error(error)

        relay = _Relay(progress, done, failed)
        self._tasks[task] = relay  # keep runnable, signals and relay alive until it reports back
        task.signals.progress.connect(relay.progress)
        task.signals.done.connect(relay.done)
        task.signals.failed.connect(relay.failed)
        if dialog is not None:
            dialog.canceled.connect(task.cancel.set)
        QThreadPool.globalInstance().start(task)  # start only after every connection exists
        return task


class TrayApp(QObject):
    def __init__(self, controller: Controller):
        super().__init__()
        self.c = controller
        prefs = controller.prefs
        self.hotkeys = HotkeyListener()
        self.history_window: HistoryWindow | None = None
        self.prefs_window: PrefsWindow | None = None
        self.image_window: ImageOCRWindow | None = None

        self.menu = QMenu()
        for text, slot in (("Capture Text", controller.start_capture),
                           ("OCR Image File…", self.show_image_window),
                           ("History…", self.show_history),
                           ("Send Feedback…", self.show_feedback),
                           ("Preferences…", self.show_prefs),
                           ("Check for Updates…", partial(self.check_updates, True))):
            action = QAction(text, self.menu)
            action.triggered.connect(lambda _checked=False, s=slot: s())
            self.menu.addAction(action)
        self.menu.addSeparator()
        quit_action = QAction("Quit KhmerClipper", self.menu)
        quit_action.triggered.connect(QApplication.quit)
        self.menu.addAction(quit_action)

        self.tray = QSystemTrayIcon(QIcon(str(resource_dir() / "icon.png")))
        self.tray.setToolTip(APP_NAME)
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._on_activated)
        self.tray.show()
        controller.popup.feedback_requested.connect(self.show_feedback)

        self.hotkey_ok = self.register_hotkey()
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.show_prefs()  # e.g. GNOME without the AppIndicator extension: keep the app reachable
        if not prefs.welcomed:
            how = (f"Press {pynput_to_qt(prefs.hotkey)}" if self.hotkey_ok
                   else "Bind “khmerclipper --capture” to a keyboard shortcut")
            self.tray.showMessage(f"{APP_NAME} is running", f"{how} or use the tray icon to capture text.")
            prefs.welcomed = True
            prefs.save()
        if prefs.check_updates and updater.due(prefs.last_update_check, time.time()):
            self.check_updates(False)
        if size_mb := controller.download_trocr_model():
            self.tray.showMessage(f"Downloading the TrOCR model ({size_mb} MB)",
                                  "It runs in the background. Text capture uses it once the download finishes.")

    def register_hotkey(self) -> bool:
        return self.hotkeys.start(self.c.prefs.hotkey, self.c.capture_requested.emit)

    def on_command(self, command: str) -> None:
        if command == "capture":
            self.c.start_capture()
        elif command == "show":
            self.show_prefs()

    def _on_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.c.start_capture()

    def show_history(self) -> None:
        if self.history_window is None:
            self.history_window = HistoryWindow(self.c.history, self.c.copy, report=self._report)
        _present(self.history_window)

    def _report(self, record: ScanRecord, correction: str) -> None:
        """Queue a capture report (raises if it can't be built or saved), then try to send it."""
        history = self.c.history
        meta = history.capture_meta(record.id)
        try:
            payload = reports.capture_payload(history.captures_dir / f"{record.id}.png", meta,
                                              meta.get("ocr_text", record.text), correction, self.c.prefs.engine)
        except Exception:  # unreadable or corrupt image, not just OSError: never fail silently
            self._report_save_failed()
            raise  # tells the History window not to mark the row Reported
        self._queue_and_send(payload)

    def show_feedback(self) -> None:
        dialog = FeedbackDialog()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        message, email = dialog.values()
        try:
            self._queue_and_send(reports.feedback_payload(message, email, self.c.prefs.engine))
        except Exception:
            pass  # already told the user

    def _report_save_failed(self) -> None:
        self.c.popup.show_error("Please try again.", title="Couldn't save the report")

    def _queue_and_send(self, payload: dict) -> None:
        try:
            reports.enqueue(payload)
        except Exception:
            self._report_save_failed()
            raise

        def done(sent) -> None:
            self.c.popup.show_message("Thanks — your report was sent." if sent else
                                      "Report saved. It will be sent when you're online.")
        self.c.submit(lambda p, c: _safe_flush(), on_done=done)

    def show_prefs(self) -> None:
        if self.prefs_window is None:
            self.prefs_window = PrefsWindow(self.c.prefs, trocr_available=trocr_models.load_manifest().available,
                                            hotkey_supported=capture_backend() == "qt")
            self.prefs_window.changed.connect(self._pref_changed)
        _present(self.prefs_window)

    def show_image_window(self) -> None:
        if self.image_window is None:
            self.image_window = ImageOCRWindow()
            self.image_window.image_chosen.connect(self.c.run_ocr)
            self.image_window.failed.connect(lambda message: self.c.popup.show_error(message, title="Couldn't open the image"))
        _present(self.image_window)

    def _pref_changed(self, name: str) -> None:
        if name == "hotkey" and not self.register_hotkey():
            self.c.popup.show_error("Try another one.", title="That shortcut couldn't be registered")
        elif name == "keep_captures" and not self.c.prefs.keep_captures:
            self.c.history.purge_captures()
            if self.history_window is not None:
                self.history_window.refresh()
        elif name == "launch_at_login":
            try:
                autostart.set_enabled(self.c.prefs.launch_at_login)
            except (OSError, NotImplementedError) as e:
                self.c.popup.show_error(str(e), title="Couldn't change launch at login")

    def check_updates(self, manual: bool) -> None:
        def fetch(progress, cancel):
            payload = updater.fetch_latest()
            return payload is not None, updater.newer_release(payload, __version__)

        def done(result) -> None:
            reachable, release = result
            if reachable:
                self.c.prefs.last_update_check = time.time()
                self.c.prefs.save()
            if release is None:
                if manual:
                    self.c.popup.show_message(f"{APP_NAME} {__version__} is up to date." if reachable
                                              else "Couldn't reach the update server. Try again later.")
                return
            answer = QMessageBox.question(None, f"{APP_NAME} update",
                                          f"{APP_NAME} {release.version} is available (you have {__version__}). "
                                          "Open the download page?")
            if answer == QMessageBox.StandardButton.Yes:
                QDesktopServices.openUrl(QUrl(release.page_url))

        self.c.submit(fetch, on_done=done)


def _safe_flush() -> int:
    """Flush queued reports in the background; failures are never surfaced to the user."""
    try:
        return reports.flush()
    except Exception:
        return 0


def _start_crash_reporting(prefs: Prefs) -> None:
    try:
        crash.install(lambda: prefs.send_crash_reports, lambda: prefs.engine)
        if prefs.send_crash_reports:
            crash.collect_fault_log(prefs.engine)
        else:
            crash.discard_fault_log()
            reports.drop_type("crash")
    except Exception:  # crash reporting must never prevent launch
        pass


def run_gui(capture: bool = False) -> int:
    if qpa := preferred_qpa(sys.platform, os.environ):
        os.environ["QT_QPA_PLATFORM"] = qpa  # must precede QApplication
    qapp = QApplication.instance() or QApplication(sys.argv[:1])
    qapp.setApplicationName(APP_NAME)
    qapp.setQuitOnLastWindowClosed(False)  # tray app: closing a window must not quit
    qapp.setWindowIcon(QIcon(str(resource_dir() / "icon.png")))

    if send_command("capture" if capture else "show"):
        return 0  # another instance is running and took the command
    server = CommandServer(parent=qapp)
    server.listen()

    prefs = Prefs.load()
    _start_crash_reporting(prefs)
    controller = Controller(prefs, HistoryStore())
    controller.submit(lambda progress, cancel: _safe_flush())
    tray = TrayApp(controller)
    server.command.connect(tray.on_command)
    qapp.aboutToQuit.connect(tray.hotkeys.stop)
    if controller.model_download is not None:  # the pool waits for running tasks on exit
        qapp.aboutToQuit.connect(controller.model_download.cancel.set)
    qapp.aboutToQuit.connect(server.close)  # drop client sockets before Qt tears the server down
    if capture:
        QTimer.singleShot(0, controller.start_capture)
    return qapp.exec()
