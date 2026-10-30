import AVFoundation
import SwiftUI

@MainActor @Observable final class VoiceRecording {
    private(set) var armed = false
    private(set) var preparing = false
    private(set) var active = false
    private(set) var peaks: [Float] = []
    private(set) var duration = 0.0
    var error: String?
    private var recorder: AVAudioRecorder?
    private var timer: Timer?
    private var permission: Task<Void, Never>?
    private var completion: ((URL) -> Void)?
    private var take = 0
    private let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)

    func arm() {
        guard !armed, !preparing else { return }
        preparing = true
        error = nil
        permission = Task { [self] in
            let allowed = await AVAudioApplication.requestRecordPermission()
            guard !Task.isCancelled else { return }
            preparing = false
            armed = allowed
            if !allowed {
                error = "Allow microphone access in System Settings → Privacy & Security → Microphone to record a sample."
            }
        }
    }

    func disarm(discard: Bool = false) {
        permission?.cancel()
        permission = nil
        preparing = false
        armed = false
        stop(discard: discard)
    }

    func start(completion: @escaping (URL) -> Void) {
        guard armed, !active else { return }
        error = nil
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            take += 1
            let url = folder.appending(path: "Voice sample \(take).wav")
            let recorder = try AVAudioRecorder(url: url, settings: [
                AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 24_000,
                AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false
            ])
            recorder.isMeteringEnabled = true
            guard recorder.record(forDuration: 20) else {
                throw CocoaError(.fileWriteUnknown)
            }
            self.recorder = recorder
            self.completion = completion
            peaks = []
            duration = 0
            active = true
            timer = Timer.scheduledTimer(withTimeInterval: 1 / 30, repeats: true) { [weak self] _ in
                MainActor.assumeIsolated { self?.meter() }
            }
        } catch {
            self.error = error.localizedDescription
        }
    }

    func stop(discard: Bool = false) {
        active = false
        timer?.invalidate()
        timer = nil
        let completion = self.completion
        self.completion = nil
        guard let recorder else { return }
        duration = max(duration, recorder.currentTime)
        recorder.stop()
        self.recorder = nil
        if discard || duration < 0.25 {
            try? FileManager.default.removeItem(at: recorder.url)
            if !discard { error = "Hold Space a little longer to record a sample." }
        } else {
            completion?(recorder.url)
        }
    }

    private func meter() {
        guard let recorder else { return }
        guard recorder.isRecording else { stop(); return }
        recorder.updateMeters()
        duration = recorder.currentTime
        peaks.append(pow(10, recorder.peakPower(forChannel: 0) / 20))
    }

    isolated deinit {
        timer?.invalidate()
        recorder?.stop()
        try? FileManager.default.removeItem(at: folder)
    }
}
