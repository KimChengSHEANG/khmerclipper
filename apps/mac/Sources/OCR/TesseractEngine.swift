import AppKit

enum TesseractError: Error {
    case bundleMissing
    case pngEncodingFailed
    case processFailed(String)
}

/// Wraps a vendored, self-contained Tesseract binary (see Vendor/tesseract) to
/// recognize Khmer script, which Vision cannot recognize at all. Runs with
/// `khm+eng` so Khmer text mixed with English (common in real captures, e.g.
/// "ដក language ផ្សេងទៀត") recognizes correctly instead of just the Khmer runs.
/// The binary and its dylibs were rewritten with `dylibbundler` to
/// `@executable_path`-relative install names so it runs without a Homebrew
/// install on the end user's Mac.
enum TesseractEngine {
    /// Points at the folder containing `bin/tesseract` + `tessdata`. Normally the
    /// app's bundled `tesseract` resource; tests set this to the repo's
    /// `Vendor/tesseract` so OCR can run in a hostless test bundle (no Bundle.main).
    static var resourceRootOverride: URL?

    private static var resourceRoot: URL? {
        resourceRootOverride
            ?? Bundle.main.resourceURL?.appendingPathComponent("tesseract", isDirectory: true)
    }

    static func recognizeText(in image: CGImage) async throws -> String {
        try await run(image)
    }

    /// One text line plus Tesseract's lowest word confidence (0-100), read
    /// from its TSV output. Lets a caller trust Tesseract only when every word
    /// on the line is confident.
    static func recognizeWords(in image: CGImage) async throws -> [(text: String, confidence: Double)] {
        parseTSV(try await run(image, extraArguments: ["-c", "tessedit_create_tsv=1"]))
    }

    /// TSV columns: level page_num block_num par_num line_num word_num left top
    /// width height conf text. Word rows have a non-empty text column.
    static func parseTSV(_ tsv: String) -> [(text: String, confidence: Double)] {
        tsv.split(whereSeparator: \.isNewline).dropFirst().compactMap { row in
            let f = row.split(separator: "\t", omittingEmptySubsequences: false)
            guard f.count > 11, let conf = Double(f[10]) else { return nil }
            let word = f[11...].joined(separator: "\t").trimmingCharacters(in: .whitespaces)
            return word.isEmpty ? nil : (word, conf)
        }
    }

    private static func run(_ image: CGImage, extraArguments: [String] = []) async throws -> String {
        guard let root = resourceRoot else { throw TesseractError.bundleMissing }
        let binary = root.appendingPathComponent("bin/tesseract")
        let tessdata = root.appendingPathComponent("tessdata")

        let imageURL = try writeTemporaryPNG(image)
        defer { try? FileManager.default.removeItem(at: imageURL) }

        return try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            process.executableURL = binary
            process.arguments = [
                imageURL.path, "stdout",
                "--tessdata-dir", tessdata.path,
                "-l", "khm+eng",
                "--psm", "6",
            ] + extraArguments
            let outPipe = Pipe()
            let errPipe = Pipe()
            process.standardOutput = outPipe
            process.standardError = errPipe

            do {
                try process.run()
            } catch {
                continuation.resume(throwing: error)
                return
            }
            process.terminationHandler = { proc in
                let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
                let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                if proc.terminationStatus == 0 {
                    let text = String(data: outData, encoding: .utf8) ?? ""
                    continuation.resume(returning: text.trimmingCharacters(in: .whitespacesAndNewlines))
                } else {
                    let message = String(data: errData, encoding: .utf8) ?? "tesseract exited with \(proc.terminationStatus)"
                    continuation.resume(throwing: TesseractError.processFailed(message))
                }
            }
        }
    }

    private static func writeTemporaryPNG(_ image: CGImage) throws -> URL {
        let rep = NSBitmapImageRep(cgImage: image)
        guard let data = rep.representation(using: .png, properties: [:]) else {
            throw TesseractError.pngEncodingFailed
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
        try data.write(to: url)
        return url
    }
}
