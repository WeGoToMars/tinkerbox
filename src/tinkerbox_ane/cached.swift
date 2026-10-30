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
}

final class CachedTensorRunner {
  let configuration: CachedTensorConfiguration
  let executable: TensorFunctions
  var cache: NDArray?
  var capacity = 0
  var position = 0

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
    let function = try await executable.get(name)
    try ensureCapacity(spec.capacity, function: function)
    var inputs: [String: NDArray] = [:]
    for (key, value) in values { inputs[key] = try inputArray(function, key, value) }
    var mask = NDArray(descriptor: try descriptor(function.descriptor.inputDescriptor(of: "mask"), "mask"))
    write4(&mask) { dst, shape, strides in
      forEachContiguousBlock(shape, strides) { _, offset, count in
        dst.advanced(by: offset).update(repeating: maskMin, count: count)
      }
      for row in 0..<spec.width {
        let offset = row * strides[2]
        for index in 0..<position { dst[offset + index * strides[3]] = 0 }
        for index in 0..<min(row + 1, count) { dst[offset + (capacity + index) * strides[3]] = 0 }
      }
    }
    inputs["mask"] = mask
    inputs["cache"] = cache
    var outputs = try await function.run(inputs: inputs)
    // Drop the input's reference before updating the cache's new rows.
    inputs["cache"] = nil
    guard let result = outputs.remove(configuration.output)?.ndArray else {
      throw HostError("missing cached tensor output")
    }
    var hidden = [Float16](repeating: 0, count: configuration.hidden * count)
    read4(result) { src, _, ss in
      for channel in 0..<configuration.hidden {
        for index in 0..<count { hidden[channel * count + index] = src[channel * ss[1] + index * ss[3]] }
      }
      write4(&cache!) { dst, _, ds in
        for channel in 0..<configuration.channels {
          for index in 0..<count {
            dst[channel * ds[1] + (position + index) * ds[3]] = src[(configuration.hidden + channel) * ss[1] + index * ss[3]]
          }
        }
      }
    }
    position += count
    return ([1, configuration.hidden, 1, count], hidden)
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
