import Foundation

// Runs the real Swift actor, bundled worker, ANE export and CPU codec.
@main struct SpeechSmoke {
    static func main() async throws {
        let runtime = Runtime.shared
        let ready = try await runtime.start(model: "omnivoice")
        precondition(ready.ready == true && !(ready.speechParameters ?? []).isEmpty)
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
        let request = SpeechRequest(items: [SpeechItem(text: "Hello.", duration: 1),
                                             SpeechItem(text: "Welcome.", instruct: "female, low pitch", duration: 1.4)],
                                    parameters: parameters, seed: 42)
        let first = try await runtime.speak(request)
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
        let reference = folder.appending(path: "reference.wav")
        try first.audio![0].wav.write(to: reference)
        let prompt = try await runtime.createVoice(SpeechVoiceRequest(
            refAudio: reference.path, refText: "Hello.", preprocessPrompt: false))
        let voice = folder.appending(path: "voice.pt")
        try prompt.write(to: voice)
        let cloned = try await runtime.speak(SpeechRequest(
            items: [SpeechItem(text: "Welcome.", voiceClonePrompt: voice.path, duration: 1)],
            parameters: parameters, seed: 42))
        precondition(cloned.audio?.first?.duration == 1)
        do {
            _ = try await runtime.speak(SpeechRequest(items: [SpeechItem(text: "Hello.", speed: 0)]))
            preconditionFailure("invalid speed was accepted")
        } catch { }
        var oversizedParameters = parameters
        oversizedParameters["audio_chunk_threshold"] = .number(120)
        do {
            _ = try await runtime.speak(SpeechRequest(items: [SpeechItem(text: "Hello.", duration: 60)],
                                                      parameters: oversizedParameters, seed: 42))
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
        print("PASS: speech worker, ordered batch, design, saved clone, seed replay, restart, context limit, errors and concurrent request isolation")
    }
}
