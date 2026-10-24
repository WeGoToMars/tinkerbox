import SwiftUI
import AVFoundation

struct AudioWaveform: View {
    let peaks: [Float]
    var progress: Double?

    var body: some View {
        Canvas { context, size in
            let maximum = max(0.01, peaks.max() ?? 0)
            let count = min(peaks.count, max(1, Int(size.width / 3)))
            for index in 0..<count {
                let start = index * peaks.count / count
                let end = (index + 1) * peaks.count / count
                let peak = (peaks[start..<end].max() ?? 0) / maximum
                let x = (Double(index) + 0.5) * size.width / Double(count)
                let height = max(2, Double(peak) * (size.height - 12))
                let bar = CGRect(x: x - 1, y: (size.height - height) / 2, width: 2, height: height)
                context.fill(Path(roundedRect: bar, cornerRadius: 1),
                             with: .color(progress == nil || x <= size.width * (progress ?? 0)
                                          ? .accentColor : .secondary.opacity(0.45)))
            }
            if let progress {
                let x = min(size.width - 1, max(1, size.width * progress))
                context.fill(Path(CGRect(x: x - 1, y: 0, width: 2, height: size.height)), with: .color(.accentColor))
            }
        }
    }
}

struct AudioPlayback: View {
    let data: Data
    var spacebarEnabled = true
    var fileExtension = "wav"
    var suspended = false
    var onInteraction: () -> Void = {}
    @State private var player: AVAudioPlayer?
    @State private var peaks: [Float] = []
    @State private var playing = false
    @State private var position = 0.0
    @State private var error: String?

    private var duration: Double { player?.duration ?? 0 }

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 30, paused: !playing)) { timeline in
            let time = playing ? player?.currentTime ?? position : position
            VStack(spacing: 12) {
                GeometryReader { geometry in
                    AudioWaveform(peaks: peaks, progress: duration > 0 ? time / duration : 0)
                    .contentShape(Rectangle())
                    .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                        seek(value.location.x / max(1, geometry.size.width) * duration)
                    })
                }
                .frame(height: 88)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Playback position")
                .accessibilityValue("\(stamp(time)) of \(stamp(duration))")
                .accessibilityAdjustableAction { direction in
                    seek((player?.currentTime ?? position) + (direction == .increment ? 5 : -5))
                }
                HStack {
                    Text(stamp(time))
                    Spacer()
                    Text("−" + stamp(max(0, duration - time)))
                }
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                HStack(spacing: 32) {
                    Button("Back 15 seconds", systemImage: "gobackward.15") { seek(time - 15) }
                    Button(playing ? "Pause" : "Play", systemImage: playing ? "pause.fill" : "play.fill") {
                        onInteraction()
                        guard let player else { return }
                        if playing {
                            position = player.currentTime
                            player.pause()
                            playing = false
                        } else {
                            if position >= duration { seek(0) }
                            playing = player.play()
                            if !playing { error = "Could not start audio playback." }
                        }
                    }
                    .font(.system(size: 30)).frame(width: 44, height: 44)
                    .keyboardShortcut(spacebarEnabled ? KeyboardShortcut(.space, modifiers: []) : nil)
                    Button("Forward 15 seconds", systemImage: "goforward.15") { seek(time + 15) }
                }
                .font(.title2).labelStyle(.iconOnly).buttonStyle(.plain)
                .disabled(player == nil)
                if let error { Text(error).font(.caption).foregroundStyle(.red) }
            }
            .onChange(of: timeline.date) {
                if playing && player?.isPlaying == false {
                    position = duration
                    playing = false
                }
            }
        }
        .task(id: data) {
            player?.stop()
            player = nil
            playing = false
            position = 0
            peaks = []
            error = nil
            do {
                let samples = try await Task.detached { try Self.envelope(data, fileExtension: fileExtension) }.value
                try Task.checkCancellation()
                player = try AVAudioPlayer(data: data)
                player?.prepareToPlay()
                peaks = samples
            } catch is CancellationError { }
            catch { self.error = error.localizedDescription }
        }
        .onDisappear {
            position = player?.currentTime ?? 0
            player?.pause()
            playing = false
        }
        .onChange(of: suspended) {
            if suspended {
                position = player?.currentTime ?? position
                player?.pause()
                playing = false
            }
        }
    }

    private func seek(_ time: Double) {
        onInteraction()
        position = min(duration, max(0, time))
        player?.currentTime = position
        if position >= duration { player?.pause(); playing = false }
    }

    private func stamp(_ time: Double) -> String {
        let seconds = Int(max(0, time))
        return String(format: "%d:%02d", seconds / 60, seconds % 60)
    }

    nonisolated private static func envelope(_ data: Data, fileExtension: String) throws -> [Float] {
        let url = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString + "." + fileExtension)
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)
        let file = try AVAudioFile(forReading: url)
        guard file.length > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else { return [] }
        try file.read(into: buffer)
        guard let channels = buffer.floatChannelData else { return [] }
        let frames = Int(buffer.frameLength), count = min(1200, frames)
        var peaks = [Float](repeating: 0, count: count)
        for index in 0..<count {
            for channel in 0..<Int(buffer.format.channelCount) {
                for frame in (index * frames / count)..<((index + 1) * frames / count) {
                    peaks[index] = max(peaks[index], abs(channels[channel][frame]))
                }
            }
        }
        return peaks
    }
}
