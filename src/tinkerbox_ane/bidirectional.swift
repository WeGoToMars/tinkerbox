// Full bidirectional transformer layers with query chunks and reusable KV storage:
// every position of the sequence attends every other.
import CoreAI
import Darwin
import Foundation

struct BidirectionalAttention: Decodable {
  let name: String
  let capacity: Int
}
struct BidirectionalConfiguration: Decodable {
  let hidden: Int, heads: Int, kvHeads: Int, headDim: Int, width: Int, logits: Int
  let projections: [String], finishes: [String]
  let attention: [BidirectionalAttention]
  let head: String
}

final class BidirectionalTensorRunner {
  let config: BidirectionalConfiguration
  let executable: TensorFunctions

  init(url: URL, config: BidirectionalConfiguration) async throws {
    self.config = config
    executable = try await TensorFunctions(url: url)
  }

  func array(_ function: InferenceFunction, _ name: String) throws -> NDArray {
    NDArray(descriptor: try descriptor(function.descriptor.inputDescriptor(of: name), name))
  }
  func zero(_ array: inout NDArray) {
    write4(&array) { dst, shape, strides in
      forEachContiguousBlock(shape, strides) { _, offset, count in memset(dst.advanced(by: offset), 0, count * 2) }
    }
  }
  func hiddenChunk(_ hidden: NDArray, at offset: Int, count: Int, into chunk: inout NDArray) {
    zero(&chunk)
    read4(hidden) { src, _, ss in
      write4(&chunk) { dst, _, ds in
        for channel in 0..<config.hidden {
          for index in 0..<count { dst[channel * ds[1] + index * ds[3]] = src[channel * ss[1] + (offset + index) * ss[3]] }
        }
      }
    }
  }

  func storeProjection(_ source: NDArray, channelOffset: Int, heads: Int, at offset: Int, count: Int, into target: inout NDArray) {
    read4(source) { src, _, ss in
      write4(&target) { dst, _, ds in
        for head in 0..<heads { for index in 0..<count { for dim in 0..<config.headDim {
          dst[head * ds[1] + (offset + index) * ds[2] + dim * ds[3]] = src[(channelOffset + head * config.headDim + dim) * ss[1] + index * ss[3]]
        } } }
      }
    }
  }

  func run(values: [String: [Float16]]) async throws -> ([Int], [Float16]) {
    let count = (values["hidden"]?.count ?? 0) / config.hidden
    guard let tier = config.attention.first(where: { $0.capacity >= count }) else {
      throw HostError("bidirectional context limit exceeded")
    }
    let attend = try await executable.get(tier.name)
    var hidden = NDArray(shape: [1, config.hidden, 1, count], scalarType: .float16)
    var next = NDArray(shape: [1, config.hidden, 1, count], scalarType: .float16)
    write4(&hidden) { dst, _, ds in
      for channel in 0..<config.hidden {
        for index in 0..<count { dst[channel * ds[1] + index * ds[3]] = values["hidden"]![channel * count + index] }
      }
    }
    var queries = NDArray(shape: [1, config.heads, tier.capacity, config.headDim], scalarType: .float16)
    var keys = try array(attend, "key"), vectors = try array(attend, "value")
    zero(&queries); zero(&keys); zero(&vectors)
    var mask = try array(attend, "mask")
    write4(&mask) { dst, shape, strides in
      forEachContiguousBlock(shape, strides) { _, offset, size in dst.advanced(by: offset).update(repeating: Float16(-30000), count: size) }
      for index in 0..<count { dst[index * strides[3]] = 0 }
    }
    for layer in config.projections.indices {
      let project = try await executable.get(config.projections[layer])
      for offset in stride(from: 0, to: count, by: config.width) {
        let size = min(config.width, count - offset)
        var input = try array(project, "hidden")
        hiddenChunk(hidden, at: offset, count: size, into: &input)
        var inputs = ["hidden": input]
        for name in ["cos", "sin"] {
          var tensor = try array(project, name)
          zero(&tensor)
          write4(&tensor) { dst, _, ds in
            for dim in 0..<config.headDim {
              for index in 0..<size { dst[dim * ds[2] + index * ds[3]] = values[name]![dim * count + offset + index] }
            }
          }
          inputs[name] = tensor
        }
        var outputs = try await project.run(inputs: inputs)
        guard let qkv = outputs.remove("features")?.ndArray else { throw HostError("missing projected features") }
        let qChannels = config.heads * config.headDim, kvChannels = config.kvHeads * config.headDim
        storeProjection(qkv, channelOffset: 0, heads: config.heads, at: offset, count: size, into: &queries)
        storeProjection(qkv, channelOffset: qChannels, heads: config.kvHeads, at: offset, count: size, into: &keys)
        storeProjection(qkv, channelOffset: qChannels + kvChannels, heads: config.kvHeads, at: offset, count: size, into: &vectors)
      }
      let finish = try await executable.get(config.finishes[layer])
      for offset in stride(from: 0, to: count, by: config.width) {
        let size = min(config.width, count - offset)
        var query = try array(attend, "query")
        zero(&query)
        read4(queries) { src, _, ss in
          write4(&query) { dst, _, ds in
            for head in 0..<config.heads { for index in 0..<size { for dim in 0..<config.headDim {
              dst[head * ds[1] + index * ds[2] + dim * ds[3]] = src[head * ss[1] + (offset + index) * ss[2] + dim * ss[3]]
            } } }
          }
        }
        var attention = try await attend.run(inputs: ["query": query, "key": keys, "value": vectors, "mask": mask])
        guard let attended = attention.remove("features")?.ndArray else { throw HostError("missing attended features") }
        var input = try array(finish, "hidden")
        hiddenChunk(hidden, at: offset, count: size, into: &input)
        var outputs = try await finish.run(inputs: ["hidden": input, "attended": attended])
        guard let output = outputs.remove("features")?.ndArray else { throw HostError("missing finished features") }
        read4(output) { src, _, ss in
          write4(&next) { dst, _, ds in
            for channel in 0..<config.hidden { for index in 0..<size {
              dst[channel * ds[1] + (offset + index) * ds[3]] = src[channel * ss[1] + index * ss[3]]
            } }
          }
        }
      }
      swap(&hidden, &next)
    }
    let head = try await executable.get(config.head)
    var logits = [Float16](repeating: 0, count: config.logits * count)
    for offset in stride(from: 0, to: count, by: config.width) {
      let size = min(config.width, count - offset)
      var input = try array(head, "hidden")
      hiddenChunk(hidden, at: offset, count: size, into: &input)
      var outputs = try await head.run(inputs: ["hidden": input])
      guard let result = outputs.remove("logits")?.ndArray else { throw HostError("missing output logits") }
      read4(result) { src, _, ss in
        for channel in 0..<config.logits { for index in 0..<size {
          logits[channel * count + offset + index] = src[channel * ss[1] + index * ss[3]]
        } }
      }
    }
    return ([1, config.logits, 1, count], logits)
  }
}

func bidirectionalTensorHost(_ asset: URL, configuration: String) async {
  let runner: BidirectionalTensorRunner
  do {
    let config = try JSONDecoder().decode(BidirectionalConfiguration.self, from: Data(configuration.utf8))
    runner = try await BidirectionalTensorRunner(url: asset, config: config)
    sendJSON(["ready": true])
  } catch { sendJSON(["error": "\(error)"]); return }
  while let operation = readExact(1)?.first {
    if operation == UInt8(ascii: "Q") { return }
    guard operation == UInt8(ascii: "B"), let values = readInputs() else { return }
    do {
      let (shape, logits) = try await runner.run(values: values)
      sendTensor(shape, logits)
    } catch { sendError("\(error)") }
  }
}
