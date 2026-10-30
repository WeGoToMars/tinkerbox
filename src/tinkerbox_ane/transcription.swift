import AVFoundation
import Speech

private struct TranscriptionError: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}

@MainActor enum ReferenceTranscription {
    static var locales: [Locale] {
        get async {
            await SpeechTranscriber.supportedLocales.sorted {
                Locale.current.localizedString(forIdentifier: $0.identifier) ?? $0.identifier <
                Locale.current.localizedString(forIdentifier: $1.identifier) ?? $1.identifier
            }
        }
    }

    static func transcribe(_ url: URL, locale: Locale = .current,
                           status: @MainActor (String) -> Void = { _ in }) async throws -> String {
        guard SpeechTranscriber.isAvailable else {
            throw TranscriptionError("Speech transcription is unavailable on this Mac. Enter the reference transcript manually.")
        }
        guard let supported = await SpeechTranscriber.supportedLocale(equivalentTo: locale) else {
            throw TranscriptionError("Choose a supported reference language or enter the transcript manually.")
        }
        let transcriber = SpeechTranscriber(locale: supported, preset: .transcription)
        try Task.checkCancellation()
        if let installation = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            status("Downloading speech recognition for \(Locale.current.localizedString(forIdentifier: supported.identifier) ?? supported.identifier)…")
            try await installation.downloadAndInstall()
        }
        try Task.checkCancellation()
        status("Transcribing reference audio…")
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let results = Task {
            var text = ""
            for try await result in transcriber.results { text += String(result.text.characters) }
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return try await withTaskCancellationHandler {
            do {
                try await analyzer.start(inputAudioFile: AVAudioFile(forReading: url), finishAfterFile: true)
                let text = try await results.value
                try Task.checkCancellation()
                guard !text.isEmpty else {
                    throw TranscriptionError("No speech was recognized. Try another recording or enter the transcript manually.")
                }
                return text
            } catch {
                results.cancel()
                await analyzer.cancelAndFinishNow()
                throw error
            }
        } onCancel: {
            Task { await analyzer.cancelAndFinishNow() }
        }
    }
}
