import SwiftUI
import AppKit

/// Small floating panel shown after a capture. It says what happened: text copied,
/// text ready to copy, nothing found, or why the capture failed.
@MainActor
final class HUDWindowController: NSWindowController {
    enum Kind {
        case copied(preview: String)
        case ready(text: String)   // auto-copy is off: clicking the HUD copies
        case noText
        case error(message: String)
    }

    private static var current: HUDWindowController?

    @discardableResult
    static func show(_ kind: Kind) -> HUDWindowController {
        current?.close()
        let controller = HUDWindowController(kind: kind)
        current = controller
        controller.showAndAutoDismiss(after: kind.duration)
        return controller
    }

    private init(kind: Kind) {
        let window = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 320, height: 72),
                             styleMask: [.nonactivatingPanel, .fullSizeContentView],
                             backing: .buffered, defer: false)
        window.level = .floating
        window.backgroundColor = .clear
        window.isOpaque = false
        window.hasShadow = true
        super.init(window: window)
        let hosting = NSHostingController(rootView: StatusHUDView(kind: kind) { [weak self] in
            if case .ready(let text) = kind { ClipboardManager.copy(text) }
            self?.close()
        })
        hosting.sizingOptions = [.preferredContentSize]   // window tracks SwiftUI's fitting size
        window.contentViewController = hosting
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    private func showAndAutoDismiss(after seconds: Double) {
        center()
        window?.orderFrontRegardless()
        // preferredContentSize resizes on the next runloop tick; re-center once it has.
        DispatchQueue.main.async { [weak self] in self?.center() }
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { [weak self] in
            self?.close()
        }
    }

    private func center() {
        guard let window, let screen = NSScreen.main else { return }
        let frame = window.frame
        window.setFrameOrigin(NSPoint(x: screen.frame.midX - frame.width / 2, y: screen.frame.midY - frame.height / 2))
    }
}

private extension HUDWindowController.Kind {
    var duration: Double {
        switch self {
        case .copied: 1.5
        case .noText: 2
        case .ready, .error: 5   // long enough to read the reason or click to copy
        }
    }

    var symbol: String {
        switch self {
        case .copied, .ready: "checkmark"
        case .noText: "text.magnifyingglass"
        case .error: "xmark"
        }
    }

    var color: Color {
        switch self {
        case .copied, .ready: .green
        case .noText: .gray
        case .error: .red
        }
    }

    var title: String {
        switch self {
        case .copied: "Copied"
        case .ready: "Done"
        case .noText: "No text found"
        case .error: "Capture failed"
        }
    }

    var subtitle: String {
        switch self {
        case .copied(let preview): preview.split(whereSeparator: \.isNewline).first.map(String.init) ?? ""
        case .ready: "Click to copy"
        case .noText: "Try selecting a larger area"
        case .error(let message): message
        }
    }
}

private struct StatusHUDView: View {
    let kind: HUDWindowController.Kind
    let onClick: () -> Void
    @State private var shown = false

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: kind.symbol)
                .font(.system(size: 16, weight: .bold))
                .foregroundStyle(.white)
                .symbolEffect(.bounce, value: shown)
                .frame(width: 34, height: 34)
                .background(Circle().fill(kind.color.gradient))

            VStack(alignment: .leading, spacing: 2) {
                Text(kind.title).font(.headline)
                Text(kind.subtitle)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: 280, alignment: .leading)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).strokeBorder(.white.opacity(0.12)))
        .contentShape(Rectangle())
        .onTapGesture(perform: onClick)
        .scaleEffect(shown ? 1 : 0.85)
        .opacity(shown ? 1 : 0)
        .padding(10)
        .onAppear {
            withAnimation(.spring(response: 0.35, dampingFraction: 0.7)) { shown = true }
        }
    }
}

/// Keeps a strong reference to the translate window so it isn't deallocated
/// the moment the presenting closure returns.
@MainActor
enum TranslateWindowPresenter {
    private static var controller: NSWindowController?

    static func show(text: String) {
        let hosting = NSHostingController(rootView: TranslateView(sourceText: text))
        let window = NSWindow(contentViewController: hosting)
        window.title = "Translate"
        window.styleMask = [.titled, .closable]
        window.center()
        let windowController = NSWindowController(window: window)
        controller = windowController
        windowController.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
