// Generic causal tensor stages: grow KV storage in the native host and send only
// the new hidden rows across the pipe. Model weights and function names come from
// the backend's manifest, just as they do for the other native runners.
import CoreAI
import Foundation

struct CachedTensorFunction: Decodable {
  let name: String
  let capacity: Int
  let width: Int
}

struct CachedTensorConfiguration: Decodable {
  let hidden: Int
  let channels: Int
  let output: String
  let functions: [CachedTensorFunction]
  let last_only: Bool?
}

final class CachedTensorRunner {
  let configuration: CachedTensorConfiguration
  let executable: TensorFunctions
  var cache: NDArray?
  var capacity = 0
  var position = 0
  var functionSeconds: [String: Double] = [:]

  init(url: URL, configuration: CachedTensorConfiguration) async throws {
    self.configuration = configuration
    executable = try await TensorFunctions(url: url)
  }

  func reset() {
    cache = nil
    capacity = 0
    position = 0
  }

  func ensureCapacity(_ next: Int, function: InferenceFunction) throws {
    if next == capacity { return }
    var grown = NDArray(descriptor: try descriptor(function.descriptor.inputDescriptor(of: "cache"), "cache"))
    write4(&grown) { dst, shape, ds in
      forEachContiguousBlock(shape, ds) { _, offset, count in
        memset(dst.advanced(by: offset), 0, count * 2)
      }
      if let old = cache {
        read4(old) { src, _, ss in
          for channel in 0..<configuration.channels {
            if ss[3] == 1 && ds[3] == 1 {
              dst.advanced(by: channel * ds[1]).update(from: src.advanced(by: channel * ss[1]), count: position)
            } else {
              for index in 0..<position { dst[channel * ds[1] + index * ds[3]] = src[channel * ss[1] + index * ss[3]] }
            }
          }
        }
      }
    }
    cache = grown
    capacity = next
  }

  func run(_ name: String, count: Int, values: [String: [Float16]]) async throws -> ([Int], [Float16]) {
    guard let spec = configuration.functions.first(where: { $0.name == name }) else {
      throw HostError("unknown cached tensor function \(name)")
    }
    guard count > 0, count <= spec.width, position + count <= spec.capacity else {
      throw HostError("cached tensor request exceeds its function capacity")
    }
    // Independent sequences are interleaved on the token axis. Padding counts
    // are logical positions per sequence; no model-specific prompt logic lives here.
    let padding = values["_padding"] ?? [0]
    let rows = padding.count
    guard rows > 0, count % rows == 0, spec.width % rows == 0, position % rows == 0,
          padding.allSatisfy({ $0.isFinite && $0 >= 0 && $0.rounded(.towardZero) == $0 }) else {
      throw HostError("invalid cached tensor row padding")
    }
    let function = try await executable.get(name)
    try ensureCapacity(spec.capacity, function: function)
    var inputs: [String: NDArray] = [:]
    for (key, value) in values where key != "_padding" { inputs[key] = try inputArray(function, key, value) }
    var mask = NDArray(descriptor: try descriptor(function.descriptor.inputDescriptor(of: "mask"), "mask"))
    write4(&mask) { dst, shape, strides in
      forEachContiguousBlock(shape, strides) { _, offset, count in
        dst.advanced(by: offset).update(repeating: maskMin, count: count)
      }
      for row in 0..<spec.width {
        let offset = row * strides[2]
        let sequence = row % rows
        let start = Int(padding[sequence]) * rows + sequence
        if start < position {
          for index in stride(from: start, to: position, by: rows) { dst[offset + index * strides[3]] = 0 }
        }
        for index in stride(from: sequence, to: min(row + 1, count), by: rows) {
          if position + index >= start { dst[offset + (capacity + index) * strides[3]] = 0 }
        }
      }
    }
    inputs["mask"] = mask
    inputs["cache"] = cache
    let before = engineSeconds
    var outputs = try await timed(name, function, inputs)
    functionSeconds[name, default: 0] += engineSeconds - before
    // Drop the input's reference before updating the cache's new rows.
    inputs["cache"] = nil
    guard let result = outputs.remove(configuration.output)?.ndArray else {
      throw HostError("missing cached tensor output")
    }
    let hiddenSize = configuration.hidden
    let returned = configuration.last_only == true ? rows : count
    let first = count - returned
    var hidden = [Float16](repeating: 0, count: hiddenSize * returned)
    read4(result) { src, _, ss in
      for channel in 0..<hiddenSize {
        for index in 0..<returned {
          hidden[channel * returned + index] = src[channel * ss[1] + (first + index) * ss[3]]
        }
      }
      write4(&cache!) { dst, _, ds in
        for channel in 0..<configuration.channels {
          let output = src.advanced(by: (hiddenSize + channel) * ss[1])
          let input = dst.advanced(by: channel * ds[1] + position * ds[3])
          if count == 1 {
            input.pointee = output.pointee
          } else if ss[3] == 1 && ds[3] == 1 {
            input.update(from: output, count: count)
          } else {
            for index in 0..<count { input[index * ds[3]] = output[index * ss[3]] }
          }
        }
      }
    }
    position += count
    return ([1, hiddenSize, 1, returned], hidden)
  }
}

func cachedTensorHost(_ asset: URL, configuration: String) async {
  let runner: CachedTensorRunner
  do {
    let spec = try JSONDecoder().decode(CachedTensorConfiguration.self, from: Data(configuration.utf8))
    runner = try await CachedTensorRunner(url: asset, configuration: spec)
    sendJSON(["ready": true])
  } catch {
    sendJSON(["error": "\(error)"])
    return
  }
  while let operation = readExact(1)?.first {
    if operation == UInt8(ascii: "Q") { return }
    if operation == UInt8(ascii: "M") {
      sendJSON(["seconds": engineSeconds, "calls": engineCalls,
                "function_seconds": runner.functionSeconds])
      (engineSeconds, engineCalls) = (0, [:])
      runner.functionSeconds = [:]
      continue
    }
    if operation == UInt8(ascii: "R") {
      runner.reset()
      writeAll([0])
      continue
    }
    guard operation == UInt8(ascii: "C"), let count = readExact(4), let name = readString(),
          let values = readInputs() else { return }
    do {
      let (shape, hidden) = try await runner.run(name, count: u32(count, 0), values: values)
      sendTensor(shape, hidden)
    } catch { sendError("\(error)") }
  }
}
