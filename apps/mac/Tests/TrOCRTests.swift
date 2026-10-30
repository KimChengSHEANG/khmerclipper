import XCTest
import AppKit
import CoreML
// TrOCREngine.swift + TrOCRTokenizer.swift are compiled directly into this test bundle (see project.yml).

final class TrOCRTokenizerTests: XCTestCase {
    /// Ids/expected strings recorded from the reference HF tokenizers
    /// (`RobertaTokenizer.decode` / `XLMRobertaTokenizer.decode`, both with
    /// `skip_special_tokens=True`) for the first few real vocab entries of
    /// trocr-base ('.', 'Ġthe', ',', 'Ġto' at ids 4,5,6,7) and trocr-small
    /// ('▁the', '.', ',', '▁to' at ids 4,5,6,7). Exercises the two pieces of
    /// decode logic this file actually implements: GPT2 byte-level BPE
    /// reconstruction (leading space kept) and SentencePiece '▁'-to-space
    /// substitution (leading space trimmed).
    func testBPEDecodeReconstructsBytesAndKeepsLeadingSpace() throws {
        let json = """
        {"kind":"bpe","bos_id":0,"eos_id":2,"pad_id":1,"unk_id":3,"decoder_start_id":2,
         "id_to_token":["<s>","<pad>","</s>","<unk>",".","Ġthe",",","Ġto"]}
        """
        let tokenizer = try TrOCRTokenizer(data: Data(json.utf8))
        XCTAssertEqual(tokenizer.decode(ids: [0, 5, 7, 2]), " the to")
    }

    /// The khmer fine-tune's vocab mixes byte-BPE pieces with literal added
    /// tokens (raw Khmer UTF-8). literal_ids marks the latter; decode must
    /// byte-map the former but pass the latter through verbatim. Here ids 4,5
    /// are byte-BPE 'a'/'b' and id 6 is a literal 'ក' (Khmer). Expect "abក".
    func testBPEDecodePassesLiteralAddedTokensThroughVerbatim() throws {
        let json = """
        {"kind":"bpe","bos_id":0,"eos_id":2,"pad_id":1,"unk_id":3,"decoder_start_id":2,
         "literal_ids":[6],
         "id_to_token":["<s>","<pad>","</s>","<unk>","a","b","ក"]}
        """
        let tokenizer = try TrOCRTokenizer(data: Data(json.utf8))
        XCTAssertEqual(tokenizer.decode(ids: [0, 4, 5, 6, 2]), "abក")
    }

    func testSentencePieceDecodeReplacesMarkerAndTrimsLeadingSpace() throws {
        let json = """
        {"kind":"sentencepiece","bos_id":0,"eos_id":2,"pad_id":1,"unk_id":3,"decoder_start_id":2,
         "id_to_token":["<s>","<pad>","</s>","<unk>","▁the",".",",","▁to"]}
        """
        let tokenizer = try TrOCRTokenizer(data: Data(json.utf8))
        XCTAssertEqual(tokenizer.decode(ids: [0, 4, 7, 2]), "the to")
    }

    /// The official microsoft/trocr-*-printed checkpoints happen to start
    /// generation from eos_token_id (a documented quirk) -- but that's a
    /// per-checkpoint fact, not a universal rule: a fine-tune can (and one
    /// does) start from bos instead. decoderStartId must reflect whatever
    /// tokenizer.json actually says, not assume it equals eosId.
    func testDecoderStartIdReflectsExportedValueNotAssumedEqualToEos() throws {
        let quirky = try TrOCRTokenizer(data: Data("""
        {"kind":"bpe","bos_id":0,"eos_id":2,"pad_id":1,"unk_id":3,"decoder_start_id":2,"id_to_token":["<s>","<pad>","</s>","<unk>"]}
        """.utf8))
        XCTAssertEqual(quirky.decoderStartId, 2)

        let overridden = try TrOCRTokenizer(data: Data("""
        {"kind":"bpe","bos_id":0,"eos_id":2,"pad_id":1,"unk_id":3,"decoder_start_id":0,"id_to_token":["<s>","<pad>","</s>","<unk>"]}
        """.utf8))
        XCTAssertEqual(overridden.decoderStartId, 0)
    }
}

/// End-to-end smoke test for the CoreML pipeline (preprocessing -> encoder ->
/// greedy decode -> tokenizer). Does not assert exact text (that depends on
/// the checkpoint); pixel parity with PIL is pinned by TrOCRPreprocessTests and
/// accuracy is measured by training/eval/scripts/bench.py.
final class TrOCREndToEndTests: XCTestCase {
    static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    override class func setUp() {
        TrOCREngine.resourceRootOverride = repoRoot.appendingPathComponent("Vendor/trocr")
    }

    func testEachModelSizeProducesNonEmptyText() async throws {
        let png = Self.repoRoot.appendingPathComponent("Tests/Fixtures/ocr/1.png")
        let image = try XCTUnwrap(loadCGImage(png))

        for size in TrOCRModelSize.allCases {
            let recognized = try await TrOCREngine.recognizeText(in: image, model: size)
            XCTAssertFalse(recognized.trimmingCharacters(in: .whitespaces).isEmpty, "TrOCR \(size.rawValue) returned empty text")
        }
    }

    private func loadCGImage(_ url: URL) -> CGImage? {
        NSImage(contentsOf: url)?.cgImage(forProposedRect: nil, context: nil, hints: nil)
    }
}

final class TrOCRLineSplitTests: XCTestCase {
    private static let monorepoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // apps/mac
        .deletingLastPathComponent()   // apps
        .deletingLastPathComponent()   // repo root

    /// These frozen eval fixtures are each ONE text line whose above-base Khmer
    /// marks form a separate ink run 37-46% of the base line's height; they
    /// must not be read as extra lines. ui-008 is dark mode, ui-005 light.
    func testSingleLineFixturesWithDetachedMarksStayOneStrip() throws {
        for name in ["ui/ui-008.png", "ui/ui-005.png", "ui/ui-009.png", "printed/printed-024.png"] {
            let url = Self.monorepoRoot.appendingPathComponent("training/eval/fixtures/\(name)")
            let image = try XCTUnwrap(NSImage(contentsOf: url)?.cgImage(forProposedRect: nil, context: nil, hints: nil), name)
            XCTAssertEqual(TrOCREngine.lineStrips(image).count, 1, name)
        }
    }

    /// TrOCR decodes one line per pass, so lineStrips must find each text line
    /// (including Khmer with marks above/below) or the output loses its
    /// linebreaks.
    func testLineStripsSplitsMultiLineClipIntoOneStripPerLine() throws {
        let lines = ["ច្បាប់ភូមិបាល", "ផ្តល់សិទ្ធិអំណាចឲ្យរដ្ឋ", "Hello world"]
        let size = NSSize(width: 600, height: 240)
        let rep = try XCTUnwrap(NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: Int(size.width), pixelsHigh: Int(size.height),
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSColor.white.setFill()
        NSRect(origin: .zero, size: size).fill()
        let attrs: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: 36), .foregroundColor: NSColor.black]
        for (i, line) in lines.enumerated() {
            (line as NSString).draw(at: NSPoint(x: 20, y: size.height - CGFloat(i + 1) * 75), withAttributes: attrs)
        }
        NSGraphicsContext.restoreGraphicsState()
        let image = try XCTUnwrap(rep.cgImage)

        XCTAssertEqual(TrOCREngine.lineStrips(image).count, 3)
    }

    /// At normal Khmer line heights the subscripts of one line touch the
    /// above-marks of the next, so no blank row separates them. Four such
    /// lines (mirrors training/eval/tests/test_pipeline.py touching_lines)
    /// must still come out as four strips covering the whole image.
    func testTwoLineCutSplitsALongAndAShortLineButNotOneLine() {
        // A full line, a 2-row bridge where marks touch, then a short wrapped line.
        let two = [Double](repeating: 100, count: 10) + [5, 5] + [Double](repeating: 40, count: 10)
        XCTAssertEqual(TrOCREngine.twoLineCut(two), 10)
        // One line: sparse above-marks, a dense core across the middle, sparse subscripts.
        let one = [Double](repeating: 10, count: 6) + [Double](repeating: 100, count: 10) + [Double](repeating: 10, count: 6)
        XCTAssertNil(TrOCREngine.twoLineCut(one))
    }

    func testLineStripsSplitsTouchingLinesAtThePitch() throws {
        let width = 260, pitch = 22, n = 4, height = n * pitch + 4
        let ctx = try XCTUnwrap(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        ctx.setFillColor(.white)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.setFillColor(.black)
        // PIL-style inclusive (x0, y0, x1, y1) boxes, y down from the top.
        func box(_ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int) {
            ctx.fill(CGRect(x: x0, y: height - 1 - y1, width: x1 - x0 + 1, height: y1 - y0 + 1))
        }
        for i in 0..<n {
            let top = 2 + i * pitch
            box(10, top + 7, width - 10, top + 15)          // dense core band
            box(30, top, 60, top + 6)                       // above marks
            box(90, top + 15, 120, top + pitch + 1)         // subscripts
            box(150, top + 15, 151, top + pitch + 8)        // bridge into the next line
        }
        let image = try XCTUnwrap(ctx.makeImage())

        let strips = TrOCREngine.lineStrips(image)
        XCTAssertEqual(strips.count, 4)
        XCTAssertEqual(strips.map(\.height).reduce(0, +), height)
    }

    /// Mirrors apps/desktop tests/test_preprocess.py gradient_bars: on a gradient the
    /// corner colour marks most of the clip as ink, so lines are found against a
    /// local background; a band of 2 px grain above a line is dropped as texture.
    private func gradientBars(_ rows: [(Int, Int)], grain: Bool = false) throws -> CGImage {
        let width = 400, height = 200
        var px = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let t = Double(x) / Double(width - 1), p = (y * width + x) * 4
                px[p] = UInt8(40 * (1 - t) + 240 * t); px[p + 1] = UInt8(40 * (1 - t) + 200 * t); px[p + 2] = UInt8(200 * (1 - t) + 40 * t)
            }
        }
        func black(_ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int) {
            for y in y0...y1 { for x in x0...x1 { let p = (y * width + x) * 4; px[p] = 0; px[p + 1] = 0; px[p + 2] = 0 } }
        }
        for (top, bottom) in rows { black(20, top, width - 21, bottom) }
        if grain {
            var g = SystemRandomNumberGenerator()
            for _ in 0..<200 { let x = Int.random(in: 20..<380, using: &g), y = Int.random(in: 30..<76, using: &g); black(x, y, x + 1, y + 1) }
        }
        let ctx = try XCTUnwrap(CGContext(data: &px, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                          space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        return try XCTUnwrap(ctx.makeImage())
    }

    func testLineStripsFindsLinesOnAGradient() throws {
        let strips = TrOCREngine.lineStrips(try gradientBars([(30, 60), (120, 150)]))
        XCTAssertEqual(strips.map(\.height), [90, 110])
    }

    /// Mirrors apps/desktop test_band_varies_on_a_gradient_not_on_a_flat_clip_with_edge_glyphs.
    func testBandVariesOnAGradientNotOnAFlatClipWithEdgeGlyphs() {
        let w = 100, h = 64
        var flat = [UInt8](repeating: 245, count: w * h * 4)
        for y in 0..<30 { for x in 0..<12 { for k in 0..<3 { flat[(y * w + x) * 4 + k] = 20 } } }
        XCTAssertFalse(TrOCREngine.bandVaries(flat, width: w, height: h))
        var grad = [UInt8](repeating: 255, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let t = Double(x) / Double(w - 1), p = (y * w + x) * 4
                grad[p] = UInt8(235 * (1 - t) + 150 * t); grad[p + 1] = UInt8(235 * (1 - t) + 160 * t); grad[p + 2] = UInt8(245 * (1 - t) + 215 * t)
            }
        }
        XCTAssertTrue(TrOCREngine.bandVaries(grad, width: w, height: h))
    }

    func testCompiledModelCacheFollowsThePackageNotItsDate() {
        let a = Data("package a".utf8), b = Data("package b".utf8)
        XCTAssertTrue(TrOCREngine.cacheIsCurrent(compiledExists: true, stamp: a, manifest: a))
        XCTAssertFalse(TrOCREngine.cacheIsCurrent(compiledExists: true, stamp: a, manifest: b))
        XCTAssertFalse(TrOCREngine.cacheIsCurrent(compiledExists: true, stamp: nil, manifest: a))
        XCTAssertFalse(TrOCREngine.cacheIsCurrent(compiledExists: false, stamp: a, manifest: a))
    }

    /// Mirrors apps/desktop test_drop_stray_symbols_trims_icon_noise_at_line_ends.
    func testDropStraySymbolsTrimsIconNoiseAtLineEnds() {
        XCTAssertEqual(TrOCREngine.dropStraySymbols("* ស្ដេច រាជរថ & >"), "ស្ដេច រាជរថ")
        XCTAssertEqual(TrOCREngine.dropStraySymbols("ក = ខ"), "ក = ខ")
        XCTAssertEqual(TrOCREngine.dropStraySymbols("ខ្មែរ។ (១)"), "ខ្មែរ។ (១)")
        XCTAssertEqual(TrOCREngine.dropStraySymbols(">"), ">")
        XCTAssertEqual(TrOCREngine.dropStraySymbols("──── ស្ដេច ──── រាជរថ"), "ស្ដេច  រាជរថ")
        XCTAssertEqual(TrOCREngine.dropStraySymbols("ស្ដេច ── រាជរថ"), "ស្ដេច ── រាជរថ")
        XCTAssertEqual(TrOCREngine.dropStraySymbols("ស្ដេច\u{FFFD}\u{FFFD} រាជរថ"), "ស្ដេច រាជរថ")
    }

    /// Mirrors apps/desktop test_line_strips_drops_a_border_only_run_on_a_clean_clip.
    func testLineStripsDropsABorderOnlyRunOnACleanClip() throws {
        let width = 400, height = 120
        let ctx = try XCTUnwrap(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        ctx.setFillColor(.white)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.setFillColor(.black)
        func box(_ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int) {
            ctx.fill(CGRect(x: x0, y: height - 1 - y1, width: x1 - x0 + 1, height: y1 - y0 + 1))
        }
        for x in stride(from: 20, to: 380, by: 12) { box(x, 20, x + 8, 44) }
        for i in 0..<9 { box(40 + 30 * i, 96 + i, 51 + 30 * i, 97 + i) }  // short 2 px dashes
        let strips = TrOCREngine.lineStrips(try XCTUnwrap(ctx.makeImage()))
        XCTAssertEqual(strips.map(\.height), [70])
    }

    /// Mirrors apps/desktop test_line_strips_a_far_widget_does_not_collapse_the_lines (real-015).
    func testLineStripsAFarWidgetDoesNotCollapseTheLines() throws {
        let width = 800, height = 160
        let ctx = try XCTUnwrap(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        ctx.setFillColor(.white)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.setFillColor(.black)
        func box(_ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int) {
            ctx.fill(CGRect(x: x0, y: height - 1 - y1, width: x1 - x0 + 1, height: y1 - y0 + 1))
        }
        box(700, 0, 760, 9)
        for top in [51, 93, 135] {
            for x in stride(from: 10, to: 740, by: 14) { box(x, top, x + 9, min(159, top + 28)) }
        }
        XCTAssertGreaterThanOrEqual(TrOCREngine.lineStrips(try XCTUnwrap(ctx.makeImage())).count, 3)
    }

    /// Mirrors apps/desktop test_split_columns_reads_a_multi_line_strip_column_by_column (real-008).
    func testSplitColumnsReadsAMultiLineStripColumnByColumn() throws {
        let width = 900, height = 200
        let ctx = try XCTUnwrap(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        ctx.setFillColor(.white)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.setFillColor(.black)
        for (x0, tops) in [(20, [10, 60, 110, 160]), (520, [30, 90, 150])] {
            for top in tops {
                for x in stride(from: x0, to: x0 + 340, by: 14) {
                    ctx.fill(CGRect(x: x, y: height - 1 - (top + 22), width: 10, height: 23))
                }
            }
        }
        let parts = TrOCREngine.splitColumns(try XCTUnwrap(ctx.makeImage()))
        XCTAssertEqual(parts.count, 7)
        XCTAssertTrue(parts.allSatisfy { $0.width < width })
        XCTAssertEqual(Set(parts.prefix(4).map(\.width)).count, 1)  // the left column's 4 lines first
    }

    /// Mirrors apps/desktop test_split_columns_resplits_a_tall_strip_without_a_gutter.
    func testSplitColumnsResplitsATallStripWithoutAGutter() throws {
        let width = 900, height = 260
        let ctx = try XCTUnwrap(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        ctx.setFillColor(.white)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.setFillColor(.black)
        for top in [10, 60, 110, 160, 210] {
            for x in stride(from: 20, to: 860, by: 14) { ctx.fill(CGRect(x: x, y: height - 1 - (top + 22), width: 10, height: 23)) }
        }
        XCTAssertEqual(TrOCREngine.splitColumns(try XCTUnwrap(ctx.makeImage())).count, 5)
    }

    /// Mirrors apps/desktop test_photo_rows_leaves_clean_and_single_line_clips_alone.
    func testPhotoRowsLeavesCleanAndSingleLineClipsAlone() throws {
        let width = 660, height = 110
        let ctx = try XCTUnwrap(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        ctx.setFillColor(.white)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.setFillColor(.black)
        for top in [12, 45, 78] {
            for x in stride(from: 40, to: 620, by: 13) { ctx.fill(CGRect(x: x, y: height - 1 - (top + 20), width: 9, height: 21)) }
        }
        let clean = try XCTUnwrap(ctx.makeImage())
        XCTAssertNil(TrOCREngine.photoRows(clean))
        XCTAssertNil(TrOCREngine.photoRows(try XCTUnwrap(clean.cropping(to: CGRect(x: 0, y: 8, width: width, height: 28)))))
    }

    func testLineStripsDropsATextureOnlyRunOnABusyClip() throws {
        let strips = TrOCREngine.lineStrips(try gradientBars([(120, 150)], grain: true))
        XCTAssertEqual(strips.count, 1)
        XCTAssertLessThan(strips[0].height, 200)
    }
}

/// Mirrors training/eval/tests/test_pipeline.py test_split_wide_*.
final class TrOCRWideSplitTests: XCTestCase {
    /// White strip with black PIL-style inclusive boxes (x0, y0, x1, y1), y down.
    private func strip(_ width: Int, _ height: Int, _ boxes: [(Int, Int, Int, Int)],
                       dark: Bool = false) throws -> CGImage {
        let ctx = try XCTUnwrap(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        ctx.setFillColor(dark ? CGColor(gray: 0.1, alpha: 1) : .white)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.setFillColor(dark ? CGColor(gray: 0.9, alpha: 1) : .black)
        for (x0, y0, x1, y1) in boxes {
            ctx.fill(CGRect(x: x0, y: height - 1 - y1, width: x1 - x0 + 1, height: y1 - y0 + 1))
        }
        return try XCTUnwrap(ctx.makeImage())
    }

    func testWideLineSplitsAtTheGapWithASpace() throws {
        let pieces = TrOCREngine.splitWide(try strip(400, 24, [(0, 2, 189, 21), (210, 2, 399, 21)]), maxAspect: 10)
        XCTAssertEqual(pieces.map(\.separator), ["", " "])
    }

    func testNarrowGapJoinsWithoutASpace() throws {
        let pieces = TrOCREngine.splitWide(try strip(400, 24, [(0, 2, 198, 21), (201, 2, 399, 21)]), maxAspect: 10)
        XCTAssertEqual(pieces.map(\.separator), ["", ""])
    }

    /// Mirrors apps/desktop test_split_wide_cuts_a_menu_row_at_every_wide_gap (real-021).
    func testWideLineCutsAMenuRowAtEveryWideGap() throws {
        var boxes: [(Int, Int, Int, Int)] = []
        for k in 0..<6 {
            for x in stride(from: 10 + 210 * k, to: 110 + 210 * k, by: 12) { boxes.append((x, 8, x + 8, 32)) }
        }
        XCTAssertEqual(TrOCREngine.splitWide(try strip(1300, 40, boxes), maxAspect: TrOCREngine.wideSplitAspect).count, 6)
        boxes.append((1290, 5, 1293, 35))  // scrollbar
        XCTAssertEqual(TrOCREngine.splitWide(try strip(1300, 40, boxes), maxAspect: TrOCREngine.wideSplitAspect).count, 6)
    }

    func testNoGapKeepsTheStripWhole() throws {
        let pieces = TrOCREngine.splitWide(try strip(400, 24, [(0, 2, 399, 21)]), maxAspect: 10)
        XCTAssertEqual(pieces.count, 1)
        XCTAssertEqual(pieces[0].image.width, 400)
    }

    func testDarkModeSplits() throws {
        let image = try strip(400, 24, [(0, 2, 189, 21), (210, 2, 399, 21)], dark: true)
        XCTAssertEqual(TrOCREngine.splitWide(image, maxAspect: 10).count, 2)
    }

    func testMeasuresTheLineNotTheBlankTail() throws {
        let image = try strip(400, 24, [(0, 6, 69, 18), (80, 6, 150, 18)])
        XCTAssertEqual(TrOCREngine.splitWide(image, maxAspect: 10).count, 1)
    }

    func testNeverEmitsAPieceOfNeighbourMarks() throws {
        let image = try strip(600, 24, [(0, 6, 140, 18), (160, 6, 300, 18), (420, 0, 440, 2), (520, 21, 540, 23)])
        let pieces = TrOCREngine.splitWide(image, maxAspect: 8)
        XCTAssertEqual(pieces.count, 2)
        XCTAssertTrue(pieces.allSatisfy { $0.image.width <= 320 })
    }
}

/// Mirrors apps/desktop tests/test_preprocess.py test_content_crop_*.
final class TrOCRContentCropTests: XCTestCase {
    private func canvas(dark: Bool, box: (Int, Int, Int, Int)?) throws -> CGImage {
        let (width, height) = (400, 200)
        let ctx = try XCTUnwrap(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        ctx.setFillColor(dark ? .black : .white)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        if let (x0, y0, x1, y1) = box {
            ctx.setFillColor(dark ? .white : .black)
            ctx.fill(CGRect(x: x0, y: height - 1 - y1, width: x1 - x0 + 1, height: y1 - y0 + 1))
        }
        return try XCTUnwrap(ctx.makeImage())
    }

    func testPadsFourPercentAroundInk() throws {
        for dark in [false, true] {
            let out = TrOCREngine.contentCrop(try canvas(dark: dark, box: (100, 80, 299, 119)))
            XCTAssertEqual([out.width, out.height], [216, 42], "dark=\(dark)")  // x 92..308, y 79..121
        }
    }

    func testKeepsBlankAndSliverImages() throws {
        let blank = try canvas(dark: false, box: nil)
        XCTAssertEqual(TrOCREngine.contentCrop(blank).width, 400)
        let sliver = try canvas(dark: false, box: (10, 10, 11, 11))
        XCTAssertEqual(TrOCREngine.contentCrop(sliver).width, 400)
    }
}

final class TrOCRPreprocessTests: XCTestCase {
    private static let macRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()

    /// Training resizes with PIL bilinear (OCRDataset._preprocess_image);
    /// on-device pixels must match it or greedy decode drifts.
    func testResizedRGBMatchesPILBilinear() throws {
        let png = Self.macRoot.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("training/eval/fixtures/printed/printed-001.png")
        let image = try XCTUnwrap(NSImage(contentsOf: png)?.cgImage(forProposedRect: nil, context: nil, hints: nil))
        let golden = try [UInt8](Data(contentsOf: Self.macRoot.appendingPathComponent("Tests/Fixtures/ocr/printed-001.pil384.rgb")))

        let swift = try XCTUnwrap(TrOCREngine.resizedRGB(image, size: 384))

        XCTAssertEqual(swift.count, golden.count)
        let maxDiff = zip(swift, golden).map { abs(Int($0) - Int($1)) }.max() ?? 0
        XCTAssertLessThanOrEqual(maxDiff, 1)
    }
}

final class TrOCRDecodeGuardTests: XCTestCase {
    func testBlockedTokenCapsRuns() {
        XCTAssertNil(TrOCREngine.blockedToken([0, 5, 5, 5]))           // 3 in a row: allowed
        XCTAssertEqual(TrOCREngine.blockedToken([0, 5, 5, 5, 5]), 5)   // 4 in a row: block a 5th
        XCTAssertNil(TrOCREngine.blockedToken([0, 5, 6, 5, 6]))        // alternating: not a run
        XCTAssertNil(TrOCREngine.blockedToken([0]))
    }

    func testArgmaxSkipsBlockedToken() throws {
        let logits = try MLMultiArray(shape: [1, 1, 4], dataType: .float32)
        for (i, v) in [0.1, 0.9, 0.5, 0.2].enumerated() { logits[i] = NSNumber(value: v) }
        XCTAssertEqual(TrOCREngine.argmaxLastStep(logits), 1)
        XCTAssertEqual(TrOCREngine.argmaxLastStep(logits, excluding: 1), 2)
    }

    func testContainsKhmerRoutesLatinOnlyLines() {
        XCTAssertFalse(TrOCREngine.containsKhmer("Warranty 11 months"))
        XCTAssertFalse(TrOCREngine.containsKhmer(""))
        XCTAssertTrue(TrOCREngine.containsKhmer("កុំព្យូទ័រ & គ្រឿងបន្លាស់"))
        XCTAssertTrue(TrOCREngine.containsKhmer("ដក language"))
    }

    func testTesseractTSVGivesWordsAndLowestConfidence() {
        let tsv = """
        level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
        1\t1\t0\t0\t0\t0\t0\t0\t410\t72\t-1\t
        5\t1\t1\t1\t1\t1\t30\t10\t120\t40\t95.0\tWarranty
        5\t1\t1\t1\t1\t2\t160\t10\t30\t40\t72.5\t11
        """
        let r = TesseractEngine.parseTSV(tsv)
        XCTAssertEqual(r.text, "Warranty 11")
        XCTAssertEqual(r.minConfidence, 72.5)
        XCTAssertEqual(TesseractEngine.parseTSV("").minConfidence, 0)
    }

    func testDecodeCapMatchesTrainingMaxLabelLength() {
        XCTAssertEqual(TrOCREngine.maxLength, 128)   // training MAX_LABEL_LEN
    }
}

extension TrOCREndToEndTests {
    /// printed-080 is a 75-code-point line (~1 token per Khmer code point); the
    /// old 64-token cap cut it off. Count unicode scalars, not Characters:
    /// a Khmer cluster (consonant + coeng + subscript + vowel) is ONE Character.
    func testLongLineIsNotTruncatedAt64Tokens() async throws {
        let png = Self.repoRoot.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("training/eval/fixtures/printed/printed-080.png")
        let image = try XCTUnwrap(NSImage(contentsOf: png)?.cgImage(forProposedRect: nil, context: nil, hints: nil))
        let text = try await TrOCREngine.recognizeText(in: image, model: .khmer)
        XCTAssertGreaterThan(text.unicodeScalars.count, 64)
    }

    /// Each line gets a fresh, zeroed KV state: recognizing A, then B, then A
    /// again must give the same A (no stale cache leaking between calls).
    func testRepeatedRecognitionIsDeterministic() async throws {
        let fixtures = Self.repoRoot.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("training/eval/fixtures")
        func load(_ name: String) throws -> CGImage {
            try XCTUnwrap(NSImage(contentsOf: fixtures.appendingPathComponent(name))?
                .cgImage(forProposedRect: nil, context: nil, hints: nil))
        }
        let a = try load("printed/printed-001.png"), b = try load("ui/ui-008.png")
        let first = try await TrOCREngine.recognizeText(in: a, model: .khmer)
        _ = try await TrOCREngine.recognizeText(in: b, model: .khmer)
        let again = try await TrOCREngine.recognizeText(in: a, model: .khmer)
        XCTAssertEqual(first, again)
        XCTAssertFalse(first.isEmpty)
    }
}

/// Noise-robust preprocessing (spec 2026-10-23), mirrored from training/eval/tests/test_noise.py
/// and apps/desktop/tests/test_noise.py with the same synthetic masks.
final class TrOCRNoiseTests: XCTestCase {
    /// A width x height mask with the given (x0, y0, x1, y1) half-open boxes set.
    private func mask(_ w: Int, _ h: Int, _ boxes: [(Int, Int, Int, Int)]) -> [Bool] {
        var m = [Bool](repeating: false, count: w * h)
        for (x0, y0, x1, y1) in boxes { for y in y0..<y1 { for x in x0..<x1 { m[y * w + x] = true } } }
        return m
    }

    private func allSet(_ m: [Bool], _ w: Int, _ box: (Int, Int, Int, Int)) -> Bool {
        (box.1..<box.3).allSatisfy { y in (box.0..<box.2).allSatisfy { m[y * w + $0] } }
    }

    func testBackgroundIgnoresAnIconInACorner() {
        let w = 200, h = 40
        var px = [UInt8](repeating: 250, count: w * h * 4)
        for y in 0..<12 { for x in 0..<12 { let p = (y * w + x) * 4; px[p] = 200; px[p + 1] = 30; px[p + 2] = 30 } }
        let bg = TrOCREngine.backgroundColor(px, width: w, height: h)
        XCTAssertTrue(bg == (250, 250, 250))
    }

    func testTextMaskDropsLinesAndBlocksButKeepsGlyphs() {
        let w = 300, h = 60
        var boxes = stride(from: 10, to: 200, by: 9).map { ($0, 20, $0 + 5, 30) }
        boxes += [(20, 16, 22, 18), (0, 45, 300, 46), (250, 5, 290, 55)]
        let (kept, removed) = TrOCREngine.textMask(mask(w, h, boxes), width: w, height: h)
        XCTAssertTrue(allSet(kept, w, (10, 20, 15, 30)) && allSet(kept, w, (20, 16, 22, 18)))
        XCTAssertTrue(allSet(removed, w, (0, 45, 300, 46)) && allSet(removed, w, (250, 5, 290, 55)))
    }

    func testTextMaskKeepsTiny1xGlyphs() {
        let m = mask(80, 12, stride(from: 2, to: 78, by: 5).map { ($0, 3, $0 + 3, 8) })
        XCTAssertEqual(TrOCREngine.textMask(m, width: 80, height: 12).kept, m)
    }

    func testTextMaskKeepsShortThinStrokesLikeKhanAndEquals() {
        let w = 200, h = 50
        var boxes = stride(from: 10, to: 120, by: 12).map { ($0, 15, $0 + 7, 35) }
        boxes += [(130, 14, 132, 36), (145, 22, 160, 24), (145, 27, 160, 29)]
        let kept = TrOCREngine.textMask(mask(w, h, boxes), width: w, height: h).kept
        XCTAssertTrue(allSet(kept, w, (130, 14, 132, 36)) && allSet(kept, w, (145, 22, 160, 24)))
    }

    func testTextMaskKeepsASmallMarkSetApartFromItsWord() {
        let w = 120, h = 40
        let kept = TrOCREngine.textMask(mask(w, h, [(10, 15, 18, 30), (22, 15, 30, 30), (34, 15, 42, 30), (100, 8, 103, 12)]),
                                        width: w, height: h).kept
        XCTAssertTrue(allSet(kept, w, (100, 8, 103, 12)))
    }

    func testTextMaskKeepsADenseBlobTouchingTheClipEdge() {
        let w = 300, h = 60
        var boxes = stride(from: 40, to: 280, by: 12).map { ($0, 20, $0 + 7, 32) }
        boxes.append((0, 0, 46, 50))
        XCTAssertTrue(allSet(TrOCREngine.textMask(mask(w, h, boxes), width: w, height: h).kept, w, (40, 20, 47, 32)))
    }

    func testTextMaskLeavesAMaskThatIsMostlyInkAlone() {
        let w = 200, h = 40
        var m = [Bool](repeating: true, count: w * h)
        for y in 0..<h { for x in stride(from: 0, to: w, by: 7) { m[y * w + x] = false } }
        let (kept, removed) = TrOCREngine.textMask(m, width: w, height: h)
        XCTAssertEqual(kept, m)
        XCTAssertFalse(removed.contains(true))
    }

    func testTextMaskKeepsALoneDigitAwayFromOtherText() {
        let w = 300, h = 110
        var boxes: [(Int, Int, Int, Int)] = []
        for y0 in [5, 30] { boxes += stride(from: 10, to: 280, by: 12).map { ($0, y0, $0 + 7, y0 + 14) } }
        boxes.append((140, 90, 148, 104))
        XCTAssertTrue(allSet(TrOCREngine.textMask(mask(w, h, boxes), width: w, height: h).kept, w, (140, 90, 148, 104)))
    }

    func testEdgeFragmentNeedsASlicedEdgeRow() {
        func pairs(_ r: [(lo: Int, hi: Int)]) -> [[Int]] { r.map { [$0.lo, $0.hi] } }
        var keep = [Double](repeating: 0, count: 80)
        for y in 0..<20 { keep[y] = 100 }; for y in 30..<50 { keep[y] = 100 }
        for y in 60..<71 { keep[y] = 90 }; for y in 71..<80 { keep[y] = 40 }; keep[79] = 8
        XCTAssertEqual(pairs(TrOCREngine.dropEdgeFragments([(0, 19), (30, 49), (60, 79)], height: 80, profile: keep)),
                       [[0, 19], [30, 49], [60, 79]])
        var cut = [Double](repeating: 0, count: 80)
        for y in 0..<5 { cut[y] = 60 }; for y in 10..<31 { cut[y] = 100 }; for y in 40..<61 { cut[y] = 100 }
        XCTAssertEqual(pairs(TrOCREngine.dropEdgeFragments([(0, 4), (10, 30), (40, 60)], height: 80, profile: cut)),
                       [[10, 30], [40, 60]])
    }

    func testBandBackgroundNeedsToExplainMoreOfTheClip() {
        // Review I1: noisy dark photo, flat light strokes crossing the edge band; corners on background.
        let w = 200, h = 60
        var px = [UInt8](repeating: 255, count: w * h * 4)
        var seed: UInt64 = 7
        for i in 0..<(w * h) {
            for c in 0..<3 {
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                px[i * 4 + c] = UInt8(16 + Int(seed >> 60) * 3)  // 16...61: spread over several 16-level bins
            }
        }
        for y in 0..<h { for x in 10..<190 where (x / 10) % 5 < 2 { for c in 0..<3 { px[(y * w + x) * 4 + c] = 250 } } }
        let band = TrOCREngine.backgroundColor(px, width: w, height: h)
        XCTAssertEqual(band.0, 250)  // the band mode is the text colour
        let bg = TrOCREngine.clipBackground(px, width: w, height: h)
        XCTAssertLessThan(bg.0, 200)  // the corners' dark background is kept
    }

    func testPercentileMatchesNumpy() {
        // numpy 2.x: 30.00000000000001 for p90 of these heights (review M1)
        let hs = [3, 4, 6, 7, 9, 12, 15, 20, 21, 21, 22, 23, 33, 39]
        XCTAssertEqual(TrOCREngine.percentile(hs, 90), 30.00000000000001)
    }

    func testDropEdgeFragments() {
        func pairs(_ r: [(lo: Int, hi: Int)]) -> [[Int]] { r.map { [$0.lo, $0.hi] } }
        XCTAssertEqual(pairs(TrOCREngine.dropEdgeFragments([(0, 4), (10, 30), (40, 60), (70, 79)], height: 80, profile: nil)), [[10, 30], [40, 60]])
        XCTAssertEqual(pairs(TrOCREngine.dropEdgeFragments([(0, 22)], height: 23, profile: nil)), [[0, 22]])
        XCTAssertEqual(pairs(TrOCREngine.dropEdgeFragments([(0, 20), (30, 50)], height: 51, profile: nil)), [[0, 20], [30, 50]])
        // Every run on an edge: judged against the tallest (noisy-edgecut-029), the 5-row sliver goes.
        XCTAssertEqual(pairs(TrOCREngine.dropEdgeFragments([(0, 4), (70, 79)], height: 80, profile: nil)), [[70, 79]])
    }
}
