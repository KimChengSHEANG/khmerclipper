from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QFont, QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from khmerclipper.paths import APP_NAME
from khmerclipper.ui.fonts import khmer_font

PREVIEW_CHARS = 1500
# Edge colour per state, matching the macOS HUD: green done, grey nothing found, red failed.
COLORS = {"copied": "#30a14e", "ready": "#30a14e", "info": "#3b82f6", "notext": "#8e8e93", "error": "#e5484d"}


def screen_at_cursor():
    return QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()


class ResultPopup(QWidget):
    """Transient bottom-right popup with the OCR result or a status message (the mac HUD's
    counterpart). A coloured edge and a bold heading say what happened."""

    copy_requested = Signal(str)
    feedback_requested = Signal()
    TIMEOUT_MS = 5000
    ERROR_TIMEOUT_MS = 8000

    def __init__(self):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setObjectName("popup")
        self.setMinimumWidth(280)
        self._text = ""
        self.kind = "info"
        self._timeout = self.TIMEOUT_MS
        self.title = QLabel()
        title_font = QFont()
        title_font.setBold(True)
        self.title.setFont(title_font)
        self.label = QLabel()
        self.label.setTextFormat(Qt.TextFormat.PlainText)  # OCR output must never render as HTML
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setMaximumWidth(420)
        self.copy_button = QPushButton("Copy")
        self.copy_button.clicked.connect(self._copy)
        self.feedback_button = QPushButton("Send Feedback…")
        self.feedback_button.clicked.connect(lambda: (self.hide(), self.feedback_requested.emit()))
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.hide)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.feedback_button)
        buttons.addWidget(self.copy_button)
        buttons.addWidget(close_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 12, 12)
        layout.addWidget(self.title)
        layout.addWidget(self.label)
        layout.addLayout(buttons)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)

    def show_text(self, text: str, copied: bool = True) -> None:
        """An OCR result; `copied` says whether it is already on the clipboard."""
        self._show("copied" if copied else "ready", "Copied" if copied else "Done", text, copyable=True,
                   timeout=min(15000, self.TIMEOUT_MS + 30 * len(text)))  # long results need reading time

    def show_message(self, message: str) -> None:
        self._show("info", APP_NAME, message)

    def show_no_text(self) -> None:
        self._show("notext", "No text found", "Try selecting a larger area.")

    def show_error(self, message: str, reported: bool = False, title: str = "Capture failed") -> None:
        if reported:
            message += "\nA crash report will be sent to the developer."
        self._show("error", title, message, timeout=self.ERROR_TIMEOUT_MS, feedback=True)

    def _show(self, kind: str, title: str, text: str, copyable: bool = False, timeout: int = TIMEOUT_MS,
              feedback: bool = False) -> None:
        self.kind = kind
        self._text = text
        self.setStyleSheet(f"QWidget#popup {{ background: palette(window); border: 1px solid palette(mid);"
                           f" border-left: 4px solid {COLORS[kind]}; }}")
        self.title.setText(title)
        self.label.setFont(khmer_font(13) if copyable else QFont())  # Khmer face for OCR text only
        self.label.setText(text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…")
        self._reset_copy()
        self.copy_button.setVisible(copyable)
        self.feedback_button.setVisible(feedback)
        self.adjustSize()
        area = screen_at_cursor().availableGeometry()
        self.move(area.right() - self.width() - 16, area.bottom() - self.height() - 16)
        self.show()
        self.raise_()
        self._timeout = timeout
        self.timer.start(timeout)

    def _copy(self) -> None:
        self.copy_requested.emit(self._text)
        self.copy_button.setText("Copied ✓")
        self.copy_button.setEnabled(False)
        # The button is the timer's context: Qt drops the call if the popup is gone by then.
        QTimer.singleShot(1500, self.copy_button, self._reset_copy)

    def _reset_copy(self) -> None:
        self.copy_button.setText("Copy")
        self.copy_button.setEnabled(True)

    def enterEvent(self, event):
        self.timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.timer.start(self._timeout)
        super().leaveEvent(event)
