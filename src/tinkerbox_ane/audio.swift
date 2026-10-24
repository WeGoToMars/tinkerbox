import AVFoundation

func convertAudio(_ source: URL, to destination: URL) throws {
  let input = try AVAudioFile(forReading: source)
  let settings: [String: Any] = ["m4a", "mp4"].contains(destination.pathExtension.lowercased()) ? [
    AVFormatIDKey: kAudioFormatMPEG4AAC,
    AVSampleRateKey: input.processingFormat.sampleRate,
    AVNumberOfChannelsKey: input.processingFormat.channelCount,
    AVEncoderBitRateKey: 64_000 * Int(input.processingFormat.channelCount)
  ] : input.processingFormat.settings
  let output = try AVAudioFile(forWriting: destination, settings: settings)
  guard let buffer = AVAudioPCMBuffer(pcmFormat: input.processingFormat, frameCapacity: 16_384) else {
    throw NSError(domain: "TinkerboxAudio", code: 1,
                  userInfo: [NSLocalizedDescriptionKey: "Cannot allocate an audio conversion buffer."])
  }
  while input.framePosition < input.length {
    try input.read(into: buffer)
    if buffer.frameLength == 0 { break }
    try output.write(from: buffer)
  }
}

func exportAudio(_ wav: Data, to destination: URL) throws {
  if destination.pathExtension.lowercased() == "wav" {
    try wav.write(to: destination, options: .atomic)
    return
  }
  let temporary = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString + ".wav")
  defer { try? FileManager.default.removeItem(at: temporary) }
  try wav.write(to: temporary)
  try convertAudio(temporary, to: destination)
}
