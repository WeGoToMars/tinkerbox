import AppIntents
import AVFoundation
import Foundation

func check(_ condition: Bool, _ message: String) {
    if !condition {
        FileHandle.standardError.write(Data("FAIL: \(message)\n".utf8))
        exit(1)
    }
}

@main struct SpeechShortcutsSmoke {
    static func main() async throws {
        let runtime = Runtime.shared
        let model = ProcessInfo.processInfo.environment["TINKERBOX_SPEECH_MODEL"] ?? "omnivoice"
        let shortcutModel = try await SpeechModelQuery().entities(for: [model]).first!
        await runtime.select(model: model, for: .speech)
        _ = try await runtime.start(model: model)
        let capabilities = Catalog.speechCapabilities[model]!
        let options = try await SpeechModeQuery().suggestedEntities().filter { $0.modelID == model }
        check(options.map(\.title).contains("Automatic voice") && options.map(\.title).contains("Voice clone"), "Missing synthesis modes")
        let restored = try await SpeechModeQuery().entities(for: [options.first { $0.modeID == "auto" }!.id])
        check(restored.count == 1 && restored[0].title == "Automatic voice", "Saved mode did not resolve")

        let generate = GenerateSpeechIntent()
        generate.model = shortcutModel
        generate.texts = ["Hello! Welcome to Tinkerbox.", "A familiar voice can tell a new story."]
        generate.mode = restored[0]
        generate.seed = 42
        let first = try await generate.perform().value!
        check(first.count == 2 && first.map(\.filename) == ["speech-42-1.wav", "speech-42-2.wav"], "Output ordering/seed filenames")
        let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        for (index, file) in first.enumerated() {
            let path = folder.appending(path: "\(index).wav")
            try file.data.write(to: path)
            let audio = try AVAudioFile(forReading: path)
            check(audio.length > 0 && audio.fileFormat.sampleRate >= 24000, "Invalid WAV output")
        }
        let replay = try await generate.perform().value!
        check(replay.map(\.data) == first.map(\.data), "Seeded batch replay differs")
        let single = GenerateSpeechIntent()
        single.model = shortcutModel
        single.texts = [generate.texts[0]]
        single.seed = 42
        let standalone = try await single.perform().value!
        check(standalone.count == 1 && !standalone[0].data.isEmpty, "Missing standalone audio")

        // Changing the app's selection never changes this Shortcut's saved model or implicit mode.
        let otherModel = model == "omnivoice" ? "voxcpm2" : "omnivoice"
        await runtime.select(model: otherModel, for: .speech)
        let afterSwitch = try await single.perform().value!
        check(afterSwitch[0].data == standalone[0].data, "App model selection changed the Shortcut output")
        check(Runtime.selectedModel(for: .speech) == otherModel, "Shortcut changed the app's saved model")

        // Blank list members fail as a whole, and worker-side validation leaves the next request usable.
        let blank = GenerateSpeechIntent()
        blank.model = shortcutModel
        blank.texts = ["Hello.", " "]
        do { _ = try await blank.perform(); preconditionFailure("Blank speech was accepted") }
        catch { check(error.localizedDescription.contains("must not be empty"), "Blank input error: \(error)") }
        let invalid = GenerateSpeechIntent()
        invalid.model = shortcutModel
        invalid.texts = single.texts
        invalid.speed = 0
        do { _ = try await invalid.perform(); preconditionFailure("Invalid speed was accepted") }
        catch { check(error.localizedDescription.contains("positive"), "Speed validation: \(error)") }
        let unsupported = GenerateSpeechIntent()
        unsupported.model = shortcutModel
        unsupported.texts = single.texts
        unsupported.mode = SpeechModeEntity(model: "unavailable-model", mode: capabilities.modes[0])
        do { _ = try await unsupported.perform(); preconditionFailure("Unknown mode was accepted") }
        catch { check(error.localizedDescription.contains("does not support"), "Mode validation: \(error)") }

        // Exercise every advertised reference-required conditioning mode through both actual actions.
        for mode in capabilities.modes where mode.reference == "required" {
            let prepare = CreateVoicePromptIntent()
            prepare.model = shortcutModel
            prepare.reference = first[0]
            prepare.mode = options.first { $0.modeID == mode.id }!
            prepare.preprocess = false
            prepare.transcript = mode.referenceTranscript == "none" ? nil : generate.texts[0]
            let saved = try await prepare.perform().value!
            check(saved.filename == "voice.pt" && !saved.data.isEmpty, "Missing voice prompt")
            let clone = GenerateSpeechIntent()
            clone.model = shortcutModel
            clone.texts = ["Every sentence should reach its final word."]
            clone.mode = prepare.mode
            clone.voice = saved
            clone.seed = 23
            let cloned = try await clone.perform().value!
            check(!cloned[0].data.isEmpty, "Missing cloned speech")
            await runtime.unload()
            let reloaded = try await clone.perform().value!
            check(reloaded[0].data == cloned[0].data, "Saved voice replay after unload differs")
            // A temporary reference copied from IntentFile survives until the worker finishes preparing it.
            clone.voice = nil
            clone.reference = first[0]
            clone.transcript = prepare.transcript
            let direct = try await clone.perform().value!
            check(!direct[0].data.isEmpty, "Missing direct-reference speech")
            clone.voice = saved
            do { _ = try await clone.perform(); preconditionFailure("Ambiguous reference was accepted") }
            catch { check(error.localizedDescription.contains("not both"), "Reference conflict validation: \(error)") }
        }
        let designed = GenerateSpeechIntent()
        designed.model = shortcutModel
        designed.texts = single.texts
        designed.mode = options.first { $0.modeID == "design" }!
        designed.instruction = "female, low pitch"
        designed.seed = 42
        designed.steps = 4
        designed.guidance = 2
        let designReply = try await designed.perform().value!
        check(!designReply[0].data.isEmpty, "Missing designed speech")

        // Shortcuts and the UI share the same request queue and deterministic seed behavior.
        check(Runtime.selectedModel(for: .speech) == otherModel, "Voice preparation changed the app's model")
        await runtime.select(model: model, for: .speech)
        async let shortcut = single.perform()
        async let ui = runtime.speak(SpeechRequest(items: [SpeechItem(text: single.texts[0])], seed: 42))
        let (shortcutReply, uiReply) = try await (shortcut, ui)
        check(shortcutReply.value![0].data == standalone[0].data && uiReply.audio![0].wav == standalone[0].data,
              "Concurrent UI/Shortcut request isolation or replay differs")
        await runtime.unload()
        print("PASS: \(model) speech App Intents, ordered WAV files, seed replay, voice prompts, cloning/continuation after unload, design, errors and concurrent UI requests")
    }
}
