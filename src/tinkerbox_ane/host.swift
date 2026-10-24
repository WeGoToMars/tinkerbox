// Swift releases output buffers that the Python binding retains.
// Pipe: R = RoPE tables, S = chain, T = packed tree, V = vision, Q = quit; integers are little-endian.
// The bundle manifest selects the runner: staged bundles list their stages, others run chunk-major.

import CoreAI
import Darwin
import Foundation

let maskMin = Float16(-30000)  // model.MASK_MIN
let input = "hidden_in"
let output = "hidden_out"

struct HostError: Error, CustomStringConvertible {
  let description: String
  init(_ description: String) { self.description = description }
}

func readExact(_ count: Int) -> [UInt8]? {
  var buffer = [UInt8](repeating: 0, count: count)
  var done = 0
  while done < count {
    let n = buffer.withUnsafeMutableBytes { read(0, $0.baseAddress! + done, count - done) }
    if n <= 0 { return nil }
    done += n
  }
  return buffer
}

func readHalves(_ count: Int) -> [Float16]? {
  var values = [Float16](repeating: 0, count: count)
  let complete = values.withUnsafeMutableBytes { raw in
    var done = 0
    while done < raw.count {
      let n = read(0, raw.baseAddress! + done, raw.count - done)
      if n <= 0 { return false }
      done += n
    }
    return true
  }
  return complete ? values : nil
}

// Keep framework stdout logs out of the reply pipe.
nonisolated(unsafe) var replyFD: Int32 = 1

func protectReplies() {
  replyFD = dup(1)
  dup2(2, 1)
}

func writeAll(_ bytes: [UInt8]) {
  var done = 0
  while done < bytes.count {
    let n = bytes.withUnsafeBytes { write(replyFD, $0.baseAddress! + done, bytes.count - done) }
    if n <= 0 { exit(1) }  // Python is gone
    done += n
  }
}

func u32(_ bytes: [UInt8], _ at: Int) -> Int {
  Int(bytes[at]) | Int(bytes[at + 1]) << 8 | Int(bytes[at + 2]) << 16 | Int(bytes[at + 3]) << 24
}

func le32(_ value: Int) -> [UInt8] { (0..<4).map { UInt8(truncatingIfNeeded: value >> (8 * $0)) } }

func halves(_ bytes: [UInt8]) -> [Float16] {
  bytes.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
}

// Neural Engine time and calls per function since the last metrics request.
nonisolated(unsafe) var engineSeconds = 0.0
nonisolated(unsafe) var engineCalls: [String: Int] = [:]

func timed(_ name: String, _ function: InferenceFunction, _ inputs: [String: NDArray]) async throws
  -> InferenceFunction.Outputs
{
  let started = ContinuousClock.now
  let result = try await function.run(inputs: inputs)
  let elapsed = ContinuousClock.now - started
  engineSeconds += Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
  engineCalls[name, default: 0] += 1
  return result
}

func sendJSON(_ object: [String: Any]) {
  let data = try! JSONSerialization.data(withJSONObject: object)
  writeAll(le32(data.count) + [UInt8](data))
}

func sendError(_ message: String) {
  let text = [UInt8](message.utf8)
  writeAll([1] + le32(text.count) + text)
}

// Core AI pads some axes; use strides for every fp16 array access.
func write4(_ array: inout NDArray, _ body: (UnsafeMutablePointer<Float16>, [Int], [Int]) -> Void) {
  let view = array.mutableView(as: Float16.self)
  view.withUnsafeMutablePointer { pointer, shape, strides in
    body(pointer, (0..<shape.count).map { shape[$0] }, (0..<strides.count).map { strides[$0] })
  }
}

func read4<R>(_ array: NDArray, _ body: (UnsafePointer<Float16>, [Int], [Int]) -> R) -> R {
  array.rawView().withUnsafeBytes { raw, shape, strides in
    body(
      raw.assumingMemoryBound(to: Float16.self),
      (0..<shape.count).map { shape[$0] }, (0..<strides.count).map { strides[$0] })
  }
}

func descriptor(_ value: InferenceValue.Descriptor?, _ name: String) throws -> NDArrayDescriptor {
  guard case .ndArray(let nd)? = value else { throw HostError("\(name) is not an ndarray") }
  return nd
}

func loadAsset(_ url: URL) async throws -> AIModel {
  // Core AI rejects an asset directory that is itself a symbolic link.
  let canonicalURL = url.resolvingSymlinksInPath()
  var options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
  options.expectFrequentReshapes = false
  // Recover invalid persistent function caches before processing input.
  for attempt in 0..<2 {
    do {
      let model = try await AIModel.specialize(
        contentsOf: canonicalURL, options: options, cachePolicy: .persistent)
      for name in model.functionNames {
        guard try model.loadFunction(named: name) != nil else {
          throw HostError("missing function \(name)")
        }
      }
      return model
    } catch {
      let message = String(reflecting: error)
      if attempt == 1 || !message.contains("nilError") { throw error }
      FileHandle.standardError.write(
        Data(
          "REBUILD_INVALID_SPECIALIZATION asset=\(url.lastPathComponent) error=\(message)\n".utf8))
      try AIModelCache.default.deleteEntry(for: canonicalURL, options: options)
    }
  }
  fatalError("unreachable")
}
@_silgen_name("installStaging") func installStaging()

func column(_ array: NDArray, _ column: Int) -> [Float16] {
  read4(array) { p, shape, s in (0..<shape[1]).map { p[$0 * s[1] + column * s[3]] } }
}

func fillInput(_ array: inout NDArray, _ x: [Float16]) {
  write4(&array) { p, shape, s in
    for c in 0..<shape[1] {
      for t in 0..<shape[3] { p[c * s[1] + t * s[3]] = x[c * shape[3] + t] }
    }
  }
}

// Runs a model's text stack; each manifest layout has one runner.
protocol Runner: AnyObject {
  var context: Int { get }
  var ready: [String: Any] { get }
  func setRope(cos: [Float16], sin: [Float16], positions: Int) throws
  func hidden(tokens: Int, answers: [Int]?) async throws -> [Float16]
}

struct Layout: Decodable {
  let version: Int
  let stages: [Int]?
  let vision_asset: String?
}

// The bundle's vision asset; a model package picks one of its functions, such as a patch capacity,
// and names its inputs. Inputs and the features output are row-major in their declared shapes.
final class Vision {
  let url: URL
  var model: AIModel?
  var functions: [String: InferenceFunction] = [:]
  var inputs: [String: [String: NDArray]] = [:]

  init(url: URL) { self.url = url }

  func run(_ name: String, _ values: [String: [Float16]]) async throws -> (shape: [Int], features: [Float16]) {
    if model == nil {
      guard FileManager.default.fileExists(atPath: url.path) else {
        throw HostError("missing vision asset at \(url.path)")
      }
      model = try await loadAsset(url)
    }
    if functions[name] == nil {
      guard let function = try model!.loadFunction(named: name) else {
        throw HostError("missing vision function \(name)")
      }
      functions[name] = function
    }
    let function = functions[name]!
    var arrays = inputs[name] ?? [:]
    for (input, x) in values {
      if arrays[input] == nil {
        arrays[input] = NDArray(
          descriptor: try descriptor(function.descriptor.inputDescriptor(of: input), input))
      }
      var matches = false
      write4(&arrays[input]!) { p, shape, s in
        matches = shape.reduce(1, *) == x.count
        if matches { forEachOffset(shape, s) { i, offset in p[offset] = x[i] } }
      }
      guard matches else { throw HostError("vision input \(input) has the wrong element count") }
    }
    inputs[name] = arrays
    guard Set(values.keys) == Set(arrays.keys) else {
      throw HostError("vision inputs differ from the previous call")
    }
    var out = try await timed(name, function, arrays)
    guard let features = out.remove("features")?.ndArray else {
      throw HostError("missing vision features output")
    }
    return read4(features) { src, shape, s in
      var result = [Float16](repeating: 0, count: shape.reduce(1, *))
      forEachOffset(shape, s) { i, offset in result[i] = src[offset] }
      return (shape, result)
    }
  }
}

// Visit every element of a strided array in row-major order.
func forEachOffset(_ shape: [Int], _ strides: [Int], _ body: (Int, Int) -> Void) {
  let count = shape.reduce(1, *)
  var index = [Int](repeating: 0, count: shape.count)
  var offset = 0
  for linear in 0..<count {
    body(linear, offset)
    var axis = shape.count - 1
    while axis >= 0 {
      index[axis] += 1
      offset += strides[axis]
      if index[axis] < shape[axis] { break }
      offset -= strides[axis] * shape[axis]
      index[axis] = 0
      axis -= 1
    }
  }
}

// Copy the largest contiguous suffix at once while honoring padded outer strides.
func forEachContiguousBlock(_ shape: [Int], _ strides: [Int], _ body: (Int, Int, Int) -> Void) {
  var axis = shape.count
  var block = 1
  while axis > 0 {
    let next = axis - 1
    if shape[next] != 1 && strides[next] != block { break }
    block *= shape[next]
    axis = next
  }
  forEachOffset(Array(shape.prefix(axis)), Array(strides.prefix(axis))) { linear, offset in
    body(linear * block, offset, block)
  }
}

@main struct Host {
  static func main() async {
    let args = CommandLine.arguments
    if args.count == 2 && args[1] == "--architecture" {
      print(AIModel.deviceArchitectureName)
      exit(0)
    }
    if args.count == 4 && args[1] == "--convert-audio" {
      protectReplies()
      do {
        try convertAudio(URL(fileURLWithPath: args[2]), to: URL(fileURLWithPath: args[3]))
        sendJSON(["ready": true])
        exit(0)
      } catch {
        sendJSON(["error": error.localizedDescription])
        exit(1)
      }
    }
    if (args.count == 3 || args.count == 4) && args[1] == "--transcribe" {
      protectReplies()
      do {
        let locale = args.count == 4 ? Locale(identifier: args[3]) : Locale.current
        let text = try await ReferenceTranscription.transcribe(URL(fileURLWithPath: args[2]), locale: locale)
        sendJSON(["text": text])
        exit(0)
      } catch {
        sendJSON(["error": error.localizedDescription])
        exit(1)
      }
    }
    if (args.count == 3 || args.count == 4) && args[1] == "--tensor" {
      protectReplies()
      installStaging()
      await tensorHost(URL(fileURLWithPath: args[2]), output: args.count == 4 ? args[3] : "logits")
      exit(0)
    }
    guard args.count == 2 else {
      FileHandle.standardError.write("usage: tinkerbox-ane-host <bundle>\n".data(using: .utf8)!)
      exit(2)
    }
    protectReplies()
    installStaging()
    let root = URL(fileURLWithPath: args[1])
    let model: Runner
    var vision: Vision?
    do {
      let layout = try JSONDecoder().decode(
        Layout.self, from: Data(contentsOf: root.appendingPathComponent("model.json")))
      guard layout.version == 2 else { throw HostError("unsupported bundle manifest") }
      model = layout.stages == nil ? try ChunkedRunner(root: root) : try StagedRunner(root: root)
      // Only the manifest's asset passed the bundle's source identity checks.
      vision = layout.vision_asset.map { Vision(url: root.appendingPathComponent($0)) }
      sendJSON(model.ready.merging(["version": layout.version]) { a, _ in a })
    } catch {
      sendJSON(["error": "\(error)"])
      exit(1)
    }
    while let op = readExact(1)?.first {
      switch op {
      case UInt8(ascii: "R"):
        guard let header = readExact(8) else { exit(1) }
        let (dim, positions) = (u32(header, 0), u32(header, 4))
        guard let c = readExact(dim * positions * 2), let s = readExact(dim * positions * 2) else {
          exit(1)
        }
        do {
          try model.setRope(cos: halves(c), sin: halves(s), positions: positions)
          writeAll([0])
        } catch { sendError("\(error)") }
      case UInt8(ascii: "S"):
        guard let header = readExact(4) else { exit(1) }
        let tokens = u32(header, 0)
        do {
          let h = try await model.hidden(tokens: tokens, answers: nil)
          var reply: [UInt8] = [0]
          h.withUnsafeBytes { reply += $0 }
          writeAll(reply)
        } catch { sendError("\(error)") }
      case UInt8(ascii: "T"):
        guard let header = readExact(8) else { exit(1) }
        let tokens = u32(header, 0), count = u32(header, 4)
        guard count > 0, count <= tokens, tokens <= model.context,
              let slots = readExact(count * 4) else { exit(1) }
        let answers = stride(from: 0, to: slots.count, by: 4).map { u32(slots, $0) }
        do {
          let h = try await model.hidden(tokens: tokens, answers: answers)
          var reply: [UInt8] = [0]
          h.withUnsafeBytes { reply += $0 }
          writeAll(reply)
        } catch { sendError("\(error)") }
      case UInt8(ascii: "V"):
        // Read every named input before validating, so an error leaves the pipe in sync.
        guard let nameLength = readExact(4), let function = readExact(u32(nameLength, 0)),
              let header = readExact(4) else { exit(1) }
        var values: [String: [Float16]] = [:]
        for _ in 0..<u32(header, 0) {
          guard let length = readExact(4), let name = readExact(u32(length, 0)),
                let count = readExact(4), let x = readHalves(u32(count, 0)) else { exit(1) }
          values[String(decoding: name, as: UTF8.self)] = x
        }
        do {
          guard let vision else { throw HostError("this bundle has no vision tower") }
          let (shape, features) = try await vision.run(String(decoding: function, as: UTF8.self), values)
          var reply: [UInt8] = [0] + le32(shape.count) + shape.flatMap(le32)
          features.withUnsafeBytes { reply += $0 }
          writeAll(reply)
        } catch { sendError("\(error)") }
      case UInt8(ascii: "M"):
        sendJSON(["seconds": engineSeconds, "calls": engineCalls])
        (engineSeconds, engineCalls) = (0, [:])
      case UInt8(ascii: "Q"):
        exit(0)
      default:
        sendError("unknown op \(op)")
        exit(1)
      }
    }
    exit(0)
  }
}

// Stateless named-tensor entry points, shared by diffusion and other non-cached tasks.
// Swift owns each output so the Python bindings cannot retain ANE output buffers.
func tensorHost(_ asset: URL, output: String) async {
  let model: AIModel
  do {
    model = try await loadAsset(asset)
    sendJSON(["ready": true])
  } catch {
    sendJSON(["error": "\(error)"])
    return
  }
  var functions: [String: InferenceFunction] = [:]
  while let op = readExact(1)?.first {
    if op == UInt8(ascii: "Q") { return }
    guard op == UInt8(ascii: "F"), let n = readExact(4),
          let nameBytes = readExact(u32(n, 0)), let count = readExact(4) else { return }
    let name = String(decoding: nameBytes, as: UTF8.self)
    var values: [String: [Float16]] = [:]
    for _ in 0..<u32(count, 0) {
      guard let length = readExact(4), let key = readExact(u32(length, 0)),
            let size = readExact(4), let value = readHalves(u32(size, 0)) else { return }
      values[String(decoding: key, as: UTF8.self)] = value
    }
    do {
      if functions[name] == nil {
        guard let function = try model.loadFunction(named: name) else {
          throw HostError("missing function \(name)")
        }
        functions[name] = function
      }
      let function = functions[name]!
      var inputs: [String: NDArray] = [:]
      for (key, value) in values {
        var array = NDArray(descriptor: try descriptor(function.descriptor.inputDescriptor(of: key), key))
        var matches = false
        write4(&array) { p, shape, strides in
          matches = shape.reduce(1, *) == value.count
          if matches {
            value.withUnsafeBufferPointer { source in
              forEachContiguousBlock(shape, strides) { linear, offset, count in
                p.advanced(by: offset).update(from: source.baseAddress!.advanced(by: linear), count: count)
              }
            }
          }
        }
        guard matches else { throw HostError("wrong element count for \(key)") }
        inputs[key] = array
      }
      var outputs = try await function.run(inputs: inputs)
      guard let result = outputs.remove(output)?.ndArray else {
        throw HostError("missing \(output) output")
      }
      read4(result) { p, shape, strides in
        var flat = [Float16](repeating: 0, count: shape.reduce(1, *))
        flat.withUnsafeMutableBufferPointer { target in
          forEachContiguousBlock(shape, strides) { linear, offset, count in
            target.baseAddress!.advanced(by: linear).update(from: p.advanced(by: offset), count: count)
          }
        }
        var reply: [UInt8] = [0] + le32(shape.count) + shape.flatMap(le32)
        flat.withUnsafeBytes { reply += $0 }
        writeAll(reply)
      }
    } catch { sendError("\(error)") }
  }
}
