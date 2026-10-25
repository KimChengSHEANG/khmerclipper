import SwiftUI
import AppKit

/// Floating progress box shown near the bottom of the screen while OCR runs,
/// with a phase label, percentage, and a Cancel button. TrOCR's decode loop can
/// take several seconds, so this gives live feedback plus a way to bail out.
@MainActor
final class OCRProgressController: NSWindowController {
    private static var current: OCRProgressController?
    private let model = OCRProgressModel()

    /// Drives the panel from a single callback: a value shows/updates the box,
    /// `nil` tears it down. `onCancel` is wired on first show.
    static func update(info: OCRProgress?, onCancel: @escaping () -> Void) {
        guard let info else {
            current?.close()
            current = nil
            return
        }
        if let current {
            current.model.info = info
        } else {
            let controller = OCRProgressController(onCancel: onCancel)
            current = controller
            controller.model.info = info
            // Fast scans (most of them, ~100 ms) finish before this and never flash the box.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                guard current === controller else { return }
                controller.place()
                controller.window?.orderFrontRegardless()
            }
        }
    }

    /// Bottom-centre of the screen, clear of the area the user just selected.
    private func place() {
        guard let window, let screen = NSScreen.main else { return }
        let area = screen.visibleFrame
        window.setFrameOrigin(NSPoint(x: area.midX - window.frame.width / 2, y: area.minY + 80))
    }

    private init(onCancel: @escaping () -> Void) {
        let hosting = NSHostingController(rootView: OCRProgressView(model: model, onCancel: onCancel))
        hosting.sizingOptions = []   // sized once below: continuous sizing can loop (see HUDWindowController)
        let window = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 420, height: 140),
                             styleMask: [.nonactivatingPanel, .fullSizeContentView],
                             backing: .buffered, defer: false)
        window.level = .floating
        window.backgroundColor = .clear
        window.isOpaque = false
        window.hasShadow = true
        window.isMovableByWindowBackground = true
        window.contentViewController = hosting
        window.setContentSize(hosting.sizeThatFits(in: NSSize(width: 600, height: 600)))
        super.init(window: window)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
}

@MainActor
final class OCRProgressModel: ObservableObject {
    @Published var info: OCRProgress = .preparing
}

private struct OCRProgressView: View {
    @ObservedObject var model: OCRProgressModel
    let onCancel: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            HStack(spacing: 14) {
                Image(systemName: model.info.symbol)
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(.tint)
                    .symbolEffect(.pulse, options: .repeating)
                    .contentTransition(.symbolEffect(.replace))
                    .frame(width: 44, height: 44)
                    .background(.tint.opacity(0.15), in: RoundedRectangle(cornerRadius: 12, style: .continuous))

                VStack(alignment: .leading, spacing: 2) {
                    Text(model.info.label)
                        .font(.headline)
                        .contentTransition(.opacity)
                    Text("KhmerClipper OCR")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 12)

                if let fraction = model.info.fraction {
                    badge("\(Int(fraction * 100))%")
                } else if let detail = model.info.detail {
                    badge(detail, symbol: "character.cursor.ibeam")
                }

                Button(action: onCancel) {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .bold))
                        .frame(width: 24, height: 24)
                        .background(.quaternary, in: Circle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .keyboardShortcut(.cancelAction)
                .help("Cancel")
            }

            // nil fraction -> indeterminate bar; otherwise a filled bar.
            Group {
                if let fraction = model.info.fraction {
                    ProgressView(value: fraction)
                } else {
                    ProgressView().progressViewStyle(.linear)
                }
            }
            .controlSize(.small)
            .tint(.accentColor)
        }
        .frame(width: 340)
        .padding(18)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).strokeBorder(.white.opacity(0.12)))
        .animation(.snappy, value: model.info.label)
        .padding(10)
    }

    private func badge(_ text: String, symbol: String? = nil) -> some View {
        HStack(spacing: 4) {
            if let symbol { Image(systemName: symbol) }
            Text(text).contentTransition(.numericText())
        }
        .font(.callout.monospacedDigit().weight(.medium))
        .foregroundStyle(.secondary)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.quaternary.opacity(0.6), in: Capsule())
        .animation(.snappy, value: text)
    }
}
