import Foundation

@main struct VoxCPM2SpeechSmoke {
    static func main() async throws {
        UserDefaults.standard.set("voxcpm2", forKey: "model.speech")
        let runtime = Runtime.shared
        let ready = try await runtime.start(model: "voxcpm2")
        precondition(ready.speechCapabilities?.modes.first { $0.id == "clone" }?.referenceTranscript == "none")
        precondition(ready.speechCapabilities?.modes.first { $0.id == "continuation" }?.referenceTranscript == "required")
        let request = SpeechRequest(items: [SpeechItem(text: "Hello! Welcome to Tinkerbox.", mode: "auto")], seed: 42)
        let first = try await runtime.speak(request)
        precondition(first.audio?.first?.sampleRate == 48000 && first.audio!.first!.duration > 0)
        await runtime.unload()
        let replay = try await runtime.speak(request)
        precondition(first.audio!.first!.wav == replay.audio!.first!.wav)
        let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let reference = folder.appending(path: "reference.wav")
        try first.audio!.first!.wav.write(to: reference)
        for mode in ["clone", "continuation", "clone_continuation"] {
            let prompt = try await runtime.createVoice(SpeechVoiceRequest(
                refAudio: reference.path, refText: mode == "clone" ? nil : "Hello! Welcome to Tinkerbox.",
                preprocessPrompt: false, mode: mode))
            let voice = folder.appending(path: "\(mode).pt")
            try prompt.write(to: voice)
            let reply = try await runtime.speak(SpeechRequest(items: [SpeechItem(
                text: "Every sentence should reach its final word.", voiceClonePrompt: voice.path, mode: mode)], seed: 23))
            precondition(reply.audio?.first?.sampleRate == 48000 && reply.audio!.first!.duration > 0)
        }
        print("PASS: VoxCPM2 capabilities, 48 kHz speech, replay after unload, transcript-free saved cloning and both continuation modes through the real worker")
        await runtime.unload()
    }
}
