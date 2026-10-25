import AppKit

enum CaptureMode {
    case text
    case barcode
}

/// Orchestrates the full snip flow: overlay selection -> screenshot -> OCR/barcode
/// -> clipboard -> history -> HUD. One instance is owned by AppDelegate.
@MainActor
final class CaptureCoordinator {
    private var overlayWindows: [NSWindow] = []
    private let preferences: AppPreferences
    private let historyStore: ScanHistoryStore
    private var hudWindowController: HUDWindowController?
    private var recognitionTask: Task<Void, Never>?

    /// Drives the bottom-of-screen progress box; nil means idle (tear down).
    /// Only TrOCR reports phases -- Tesseract runs fast enough not to need it.
    var onOCRProgress: (@MainActor (OCRProgress?) -> Void)?

    /// Cancels an in-flight recognition (from the progress box's Cancel button).
    func cancelCapture() {
        recognitionTask?.cancel()
    }

    init(preferences: AppPreferences, historyStore: ScanHistoryStore) {
        self.preferences = preferences
        self.historyStore = historyStore
    }

    func beginCapture(mode: CaptureMode, saveFixture: Bool = false) {
        guard overlayWindows.isEmpty else { return }
        // Warm the TrOCR model while the user drags out their selection, so the
        // first-time compile/load doesn't stall after they release the mouse.
        if mode == .text, preferences.ocrEngine == .trocr {
            Task.detached { await TrOCREngine.preload(model: .khmer) }
        }
        for screen in NSScreen.screens {
            let window = SelectionOverlayWindow(screen: screen)
            let view = SelectionOverlayView(frame: screen.frame)
            view.onComplete = { [weak self] rect in
                self?.finishSelection(rect: rect, screen: screen, mode: mode, saveFixture: saveFixture)
            }
            window.contentView = view
            window.makeKeyAndOrderFront(nil)
            let locationInWindow = window.convertPoint(fromScreen: NSEvent.mouseLocation)
            view.primeMouseLocation(locationInWindow)
            overlayWindows.append(window)
        }
        NSApp.activate(ignoringOtherApps: true)
        NSCursor.hide()
    }

    private func finishSelection(rect: CGRect?, screen: NSScreen, mode: CaptureMode, saveFixture: Bool = false) {
        NSCursor.unhide()
        overlayWindows.forEach { $0.orderOut(nil) }
        overlayWindows.removeAll()

        guard let rect else { return }
        if preferences.playCaptureSound {
            NSSound(named: "Grab")?.play()
        }

        recognitionTask = Task { [weak self] in
            guard let self else { return }
            do {
                let image = try await ScreenCaptureManager.captureImage(of: rect, screen: screen)
                if saveFixture {
                    let text = try await TesseractEngine.recognizeText(in: image)
                    try self.writeFixture(image: image, text: text)
                    await self.handleResult(text: text, kind: .text, image: image, engine: .tesseract)
                } else {
                    try await self.recognize(image: image, mode: mode)
                }
            } catch is CancellationError {
                // User hit Cancel; the progress box already tore down, nothing to show.
            } catch {
                await self.handleFailure(error)
            }
        }
    }

    /// Runs OCR/barcode detection on an already-in-hand image (a screen capture,
    /// or a file dropped/opened via the Image OCR window) and reports the result.
    func recognize(image: CGImage, mode: CaptureMode) async throws {
        switch mode {
        case .text:
            defer { onOCRProgress?(nil) }
            onOCRProgress?(.preparing)   // box appears if this is still running after 0.4 s
            let text = try await OCRDispatcher.recognizeText(
                in: image,
                engine: preferences.ocrEngine,
                progress: { [onOCRProgress] info in onOCRProgress?(info) }
            )
            await handleResult(text: text, kind: .text, image: image, engine: preferences.ocrEngine)
        case .barcode:
            let payloads = try await BarcodeEngine.detectPayloads(in: image)
            await handleResult(text: payloads.joined(separator: "\n"), kind: .barcode)
        }
    }

    func reportFailure(_ error: Error) async {
        await handleFailure(error)
    }

    /// `image`/`engine` are only given for text scans; the image is kept with the
    /// history entry when "Keep capture images" is on.
    private func handleResult(text: String, kind: ScanRecord.Kind,
                              image: CGImage? = nil, engine: OCREngineKind? = nil) async {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            hudWindowController = HUDWindowController.show(.noText)
            return
        }
        let formatted = OCRTextFormatter.apply(text, preserveLinebreaks: preferences.preserveLinebreaks)
        if preferences.autoCopyToClipboard {
            ClipboardManager.copy(formatted)
        }
        let record = ScanRecord(text: formatted, kind: kind)
        if preferences.keepCaptureImages, let image, let engine {
            historyStore.add(record, image: image, meta: [
                "ocr_text": text,
                "engine": engine.rawValue,
                "app_version": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "",
                "date": ISO8601DateFormatter().string(from: record.date),
                "preserve_linebreaks": preferences.preserveLinebreaks,
            ])
        } else {
            historyStore.add(record)
        }
        if preferences.showResultHUD {
            hudWindowController = HUDWindowController.show(preferences.autoCopyToClipboard ? .copied(preview: formatted) : .ready(text: formatted))
        }
    }

    private func handleFailure(_ error: Error) async {
        hudWindowController = HUDWindowController.show(.error(message: error.localizedDescription))
    }

    /// Dev-only: dumps the capture plus its OCR text into the repo's OCR fixtures
    /// so a new regression case can be added with one hotkey. Uses #filePath to
    /// locate the source tree, so it only writes anything on the dev machine.
    /// Ordinal starts at 3 and continues past the highest existing `<n>.png`.
    private func writeFixture(image: CGImage, text: String) throws {
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // Sources/Capture
            .deletingLastPathComponent()   // Sources
            .deletingLastPathComponent()   // repo root
            .appendingPathComponent("Tests/Fixtures/ocr")
        let fm = FileManager.default
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        let existing = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
        let maxOrdinal = existing
            .filter { $0.hasSuffix(".png") }
            .compactMap { Int($0.dropLast(4)) }
            .max() ?? 0
        let index = max(3, maxOrdinal + 1)

        guard let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]) else {
            throw CocoaError(.fileWriteUnknown)
        }
        try png.write(to: dir.appendingPathComponent("\(index).png"))
        try text.write(to: dir.appendingPathComponent("\(index).txt"), atomically: true, encoding: .utf8)
    }
}
