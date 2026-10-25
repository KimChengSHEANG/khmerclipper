from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from khmerclipper.ui.popup import screen_at_cursor


class ProgressPanel(QWidget):
    """Small frameless progress box near the bottom of the screen (the mac panel's counterpart).
    Same calls as the QProgressDialog it replaces: setLabelText, setValue, canceled."""

    canceled = Signal()
    SHOW_DELAY_MS = 400  # quick Tesseract runs finish first and never flash the panel

    def __init__(self, label: str, maximum: int = 0):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setObjectName("panel")
        self.setStyleSheet("QWidget#panel { background: palette(window); border: 1px solid palette(mid); }")
        self.label = QLabel(label)
        self.bar = QProgressBar()
        self.bar.setRange(0, maximum)  # 0..0 = indeterminate
        self.bar.setTextVisible(maximum > 0)
        self.bar.setFixedWidth(300)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel)
        row = QHBoxLayout()
        row.addWidget(self.bar)
        row.addWidget(self.cancel_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 10, 10)
        layout.addWidget(self.label)
        layout.addLayout(row)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._appear)
        self._timer.start(self.SHOW_DELAY_MS)

    def setLabelText(self, text: str) -> None:
        self.label.setText(text)

    def setValue(self, value: int) -> None:
        self.bar.setValue(value)

    def _cancel(self) -> None:
        self.label.setText("Cancelling…")
        self.cancel_button.setEnabled(False)
        self.canceled.emit()

    def _appear(self) -> None:
        self.adjustSize()
        area = screen_at_cursor().availableGeometry()
        self.move(area.center().x() - self.width() // 2, area.bottom() - self.height() - 80)
        self.show()
        self.raise_()

    def close(self) -> bool:
        self._timer.stop()  # finished before the delay: never show
        return super().close()
