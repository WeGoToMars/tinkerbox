import Foundation
import AVFoundation

// Runs the real Swift actor, bundled worker, ANE export and CPU codec.
@main struct SpeechSmoke {
    static func main() async throws {
        let model = Catalog.bundled.first { $0.id == "omnivoice" }
        precondition(model?.nonVerbalTags?.contains("[laughter]") == true)
        let runtime = Runtime.shared
        let ready = try await runtime.start(model: "omnivoice")
        precondition(ready.ready == true && !(ready.speechParameters ?? []).isEmpty)
        precondition(ready.speechCapabilities?.controls.contains { $0.key == "duration" } == true)
        let short = try await runtime.speak(SpeechRequest(items: [SpeechItem(text: "fuck yeah")], seed: 7))
        precondition(short.audio?.count == 1)
        if let path = ProcessInfo.processInfo.environment["TINKERBOX_SPEECH_SMOKE_WAV"] {
            try short.audio![0].wav.write(to: URL(fileURLWithPath: path))
        }
        let parameters: [String: SpeechValue] = [
            "num_step": .number(4), "position_temperature": .number(0),
            "class_temperature": .number(0), "postprocess_output": .boolean(false),
            "pad_duration": .number(0), "fade_duration": .number(0)
        ]
        let request = SpeechRequest(items: [SpeechItem(text: "Hello.", mode: "auto", controls: ["duration": .number(1)]),
                                             SpeechItem(text: "Welcome.", instruct: "female, low pitch", mode: "design",
                                                        controls: ["duration": .number(1.4)])],
                                    parameters: parameters, seed: 42)
        let first = try await runtime.speak(request)
        precondition(first.cloneMs == 0 && first.audioMs! > 0)
        precondition(first.audio?.count == 2)
        precondition(first.audio![0].sampleRate == 24000 && first.audio![0].duration == 1)
        precondition(first.audio![1].duration == 1.4)
        var randomParameters = parameters
        randomParameters["position_temperature"] = .number(5)
        var randomRequest = SpeechRequest(items: [SpeechItem(text: "Hello.", duration: 1)],
                                          parameters: randomParameters)
        let random = try await runtime.speak(randomRequest)
        precondition(random.seed != nil)
        randomRequest.seed = random.seed
        let replay = try await runtime.speak(randomRequest)
        precondition(replay.audio![0].wav == random.audio![0].wav)
        let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let wav = folder.appending(path: "reference.wav")
        let reference = folder.appending(path: "reference.m4a")
        let spoken = try await runtime.speak(SpeechRequest(items: [SpeechItem(
            text: "A familiar voice can tell a new story.", language: "en")], seed: 42))
        try exportAudio(spoken.audio![0].wav, to: wav)
        let exported = try Data(contentsOf: wav)
        precondition(exported == spoken.audio![0].wav)
        try exportAudio(spoken.audio![0].wav, to: reference)
        do {
            let encoded = try AVAudioFile(forReading: reference)
            precondition(encoded.fileFormat.streamDescription.pointee.mFormatID == kAudioFormatMPEG4AAC)
            let player = try AVAudioPlayer(contentsOf: reference)
            precondition(player.prepareToPlay() && abs(player.duration - spoken.audio![0].duration) < 0.1)
        }
        let transcript = try await ReferenceTranscription.transcribe(reference, locale: Locale(identifier: "en-US"))
        precondition(transcript.lowercased().contains("familiar voice") && transcript.lowercased().contains("new story"), transcript)
        // Blank transcripts use Apple's transcriber before reaching the Python worker.
        await runtime.unload()
        let prompt = try await runtime.createVoice(SpeechVoiceRequest(
            refAudio: reference.path, refText: nil, preprocessPrompt: false))
        let voice = folder.appending(path: "voice.pt")
        try prompt.write(to: voice)
        await runtime.unload()
        let cloned = try await runtime.speak(SpeechRequest(
            items: [SpeechItem(text: "Welcome.", voiceClonePrompt: voice.path, duration: 1)],
            parameters: parameters, seed: 42))
        precondition(cloned.audio?.first?.duration == 1)
        let directClone = try await runtime.speak(SpeechRequest(
            items: [SpeechItem(text: "Welcome.", refAudio: reference.path, duration: 1)],
            parameters: parameters, seed: 42))
        precondition(directClone.audio?.first?.duration == 1)
        precondition(directClone.cloneMs! > 0 && directClone.audioMs! > 0)
        precondition(abs(directClone.elapsedMs! - directClone.cloneMs! - directClone.audioMs!) < 0.001)
        do {
            _ = try await runtime.speak(SpeechRequest(items: [SpeechItem(text: "Hello.", speed: 0)]))
            preconditionFailure("invalid speed was accepted")
        } catch { }
        do {
            _ = try await runtime.speak(SpeechRequest(items: [SpeechItem(
                text: String(repeating: "Hello ", count: 50_000), duration: 1)],
                                                      parameters: parameters, seed: 42))
            preconditionFailure("oversized speech context was accepted")
        } catch {
            precondition(error.localizedDescription.contains("bundle limit"))
        }
        async let one = runtime.speak(request)
        async let two = runtime.speak(request)
        let (a, b) = try await (one, two)
        precondition(a.audio![0].wav == first.audio![0].wav && b.audio![0].wav == first.audio![0].wav)
        // Selecting the local bundle again replaces the worker and preserves public behavior.
        if let asset = ProcessInfo.processInfo.environment["TINKERBOX_ASSET"] {
            await runtime.select(model: "omnivoice", for: .speech, asset: asset)
            let restarted = try await runtime.speak(request)
            precondition(restarted.audio![0].wav == first.audio![0].wav)
        }
        print("PASS: speech worker, native transcription, M4A cloning after unload, automatic/saved cloning, ordered batch, design, seed replay, restart, context limit, errors and concurrent request isolation")
    }
}
