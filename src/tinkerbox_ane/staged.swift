// Layer-major runner for stacks whose full attention sees the whole prompt (Qwen3.5 with
// noncausal full attention). Each stage scans its recurrent layers over every 128-token chunk,
// collecting the complete prompt's keys, then runs its full-attention layer on every chunk.

import CoreAI
import Foundation

// Copy a chunk's [1, heads, width, dim] keys or values into the prompt cache at a token offset.
func place(_ source: NDArray, into cache: inout NDArray, at offset: Int) {
  read4(source) { src, shape, ss in
    write4(&cache) { dst, _, ds in
      for h in 0..<shape[1] { for t in 0..<shape[2] {
        let from = src + h * ss[1] + t * ss[2], to = dst + h * ds[1] + (offset + t) * ds[2]
        if ss[3] == 1 && ds[3] == 1 {
          to.update(from: from, count: shape[3])
        } else {
          for d in 0..<shape[3] { to[d * ds[3]] = from[d * ss[3]] }
        }
      } }
    }
  }
}

// One prompt cache per capacity, reused by every stage and request. Masked keys keep stale finite
// values, which contribute nothing to attention.
final class PromptCache {
  var keys: NDArray, values: NDArray, mask: NDArray
  init(keys: NDArray, values: NDArray, mask: NDArray) { (self.keys, self.values, self.mask) = (keys, values, mask) }
}

struct StageAsset: Decodable { let asset: String }

struct StageManifest: Decodable {
  let width: Int, context: Int, hidden: Int, rotary_dim: Int, conv_taps: Int
  let assets: [StageAsset], tiers: [Int], stages: [Int]
  let prompts: [Int]?
}

final class StagedRunner: Runner {
  let manifest: StageManifest
  let root: URL
  let width: Int, context: Int, hiddenSize: Int
  var activeModels: [Int: AIModel] = [:]
  var functions: [String: InferenceFunction] = [:]
  var cos: [Float16] = [], sin: [Float16] = [], positions = 0
  var caches: [Int: PromptCache] = [:]
  // Per-request inputs that never change: each stage's empty recurrent state and each chunk's RoPE.
  var emptyStates: [Int: NDArray] = [:]
  var ropeChunks: [(NDArray, NDArray)] = []
  var ropePrompts: [Int: (NDArray, NDArray)] = [:]

  init(root: URL) throws {
    self.root = root
    manifest = try JSONDecoder().decode(
      StageManifest.self, from: Data(contentsOf: root.appendingPathComponent("model.json")))
    guard manifest.assets.count == manifest.stages.count else { throw HostError("one asset per stage") }
    width = manifest.width
    context = manifest.context
    hiddenSize = manifest.hidden
  }

  var ready: [String: Any] { ["width": width, "context": context, "hidden": hiddenSize,
                               "conv_taps": manifest.conv_taps] }

  func function(_ name: String, stage: Int) async throws -> InferenceFunction {
    if let function = functions[name] { return function }
    if activeModels[stage] == nil {
      activeModels[stage] = try await loadAsset(root.appendingPathComponent(manifest.assets[stage].asset))
    }
    guard let function = try activeModels[stage]!.loadFunction(named: name) else {
      throw HostError("missing function \(name)")
    }
    functions[name] = function
    return function
  }

  func inputArray(_ function: InferenceFunction, _ name: String, zeroed: Bool = true) throws -> NDArray {
    var array = NDArray(descriptor: try descriptor(function.descriptor.inputDescriptor(of: name), name))
    guard zeroed else { return array }
    write4(&array) { p, shape, s in
      for b in 0..<shape[0] { for c in 0..<shape[1] {
        for h in 0..<shape[2] { for w in 0..<shape[3] {
          p[b * s[0] + c * s[1] + h * s[2] + w * s[3]] = 0
        } }
      } }
    }
    return array
  }

  func attention(_ attend: InferenceFunction, capacity: Int) throws -> PromptCache {
    if let cache = caches[capacity] { return cache }
    let cache = PromptCache(keys: try inputArray(attend, "key"), values: try inputArray(attend, "value"),
                            mask: try inputArray(attend, "mask"))
    caches[capacity] = cache
    return cache
  }

  func setRope(cos: [Float16], sin: [Float16], positions: Int) throws {
    guard positions >= context, cos.count == sin.count,
          cos.count == positions * (manifest.rotary_dim / 2) else {
      throw HostError("invalid RoPE")
    }
    (self.cos, self.sin, self.positions) = (cos, sin, positions)
    ropeChunks = []
    ropePrompts = [:]
  }

  // Copy RoPE columns [offset, offset + width) into a [1, 1, rotary/2, width] input pair.
  func ropeInputs(_ function: InferenceFunction, offset: Int) throws -> (NDArray, NDArray) {
    var c = try inputArray(function, "cos", zeroed: false), s = try inputArray(function, "sin", zeroed: false)
    for (table, array) in [(cos, 0), (sin, 1)] {
      let fill = { (p: UnsafeMutablePointer<Float16>, shape: [Int], strides: [Int]) in
        for i in 0..<shape[2] { for t in 0..<shape[3] {
          p[i * strides[2] + t * strides[3]] = table[i * self.positions + offset + t]
        } }
      }
      if array == 0 { write4(&c, fill) } else { write4(&s, fill) }
    }
    return (c, s)
  }

  // A short prompt: each stage runs once over the whole padded prompt.
  func prompt(tokens: Int, width promptWidth: Int, chunks: [[Float16]]) async throws -> [Float16] {
    let first = try await function("prompt0c\(promptWidth)", stage: 0)
    var x = try inputArray(first, input, zeroed: false)
    write4(&x) { p, shape, s in
      for c in 0..<shape[1] { for t in 0..<shape[3] {
        let chunk = t / self.width
        p[c * s[1] + t * s[3]] = chunk < chunks.count ? chunks[chunk][c * self.width + t % self.width] : 0
      } }
    }
    if ropePrompts[promptWidth] == nil { ropePrompts[promptWidth] = try ropeInputs(first, offset: 0) }
    let (c, s) = ropePrompts[promptWidth]!
    var mask = try inputArray(first, "mask", zeroed: false)
    write4(&mask) { p, shape, strides in
      for t in 0..<shape[3] { p[t * strides[3]] = t < tokens ? 0 : maskMin }
    }
    for stage in manifest.stages.indices {
      let name = "prompt\(stage)c\(promptWidth)"
      let run = try await function(name, stage: stage)
      var out = try await timed(name, run, [input: x, "cos": c, "sin": s, "mask": mask])
      guard let hidden = out.remove(output)?.ndArray else { throw HostError("missing prompt output") }
      x = hidden
    }
    return column(x, tokens - 1)
  }

  func hidden(tokens: Int, answers: [Int]?) async throws -> [Float16] {
    // Full attention over the whole prompt leaves no question-independent trunk to share.
    guard answers == nil else { throw HostError("this model answers one question per pass") }
    guard positions > 0, tokens >= 1, tokens <= context,
          let capacity = manifest.tiers.first(where: { $0 >= tokens }) else {
      throw HostError("invalid request")
    }
    let count = (tokens + width - 1) / width
    var embeddings: [[Float16]] = []
    for _ in 0..<count {
      writeAll([0])
      guard let values = readHalves(width * hiddenSize) else {
        throw HostError("incomplete embedding chunk")
      }
      embeddings.append(values)
    }
    if let promptWidth = manifest.prompts?.first(where: { $0 >= tokens }) {
      return try await prompt(tokens: tokens, width: promptWidth, chunks: embeddings)
    }
    let first = try await function("scan0", stage: 0)
    var chunks: [[String: NDArray]] = []
    for values in embeddings {
      // fillInput writes every element.
      var x = try inputArray(first, input, zeroed: false)
      fillInput(&x, values)
      chunks.append([output: x])
    }
    let attendShape = try await function("attend0c\(capacity)", stage: 0)
    let cache = try attention(attendShape, capacity: capacity)
    write4(&cache.mask) { p, shape, strides in
      for t in 0..<shape[3] { p[t * strides[3]] = t < tokens ? 0 : maskMin }
    }
    // RoPE inputs depend only on the chunk position; every stage and request reuses them.
    for chunk in ropeChunks.count..<max(count, ropeChunks.count) {
      ropeChunks.append(try ropeInputs(first, offset: chunk * width))
    }
    let rope = ropeChunks
    for stage in manifest.stages.indices {
      let scan = try await function("scan\(stage)", stage: stage)
      let attend = try await function("attend\(stage)c\(capacity)", stage: stage)
      if emptyStates[stage] == nil { emptyStates[stage] = try inputArray(scan, "packed_state") }
      var state = emptyStates[stage]!
      // Finish the causal scans and collect every key before running any full attention.
      for chunk in 0..<count {
        var out = try await timed("scan\(stage)", scan, [input: chunks[chunk][output]!, "cos": rope[chunk].0,
                                                          "sin": rope[chunk].1, "packed_state": state])
        guard let nextState = out.remove("packed_state_out")?.ndArray else { throw HostError("missing scan state") }
        state = nextState
        var saved: [String: NDArray] = [:]
        for name in [output, "query", "gate"] {
          guard let array = out.remove(name)?.ndArray else { throw HostError("missing scan output \(name)") }
          saved[name] = array
        }
        guard let k = out.remove("key")?.ndArray, let v = out.remove("value")?.ndArray else {
          throw HostError("missing scan keys")
        }
        place(k, into: &cache.keys, at: chunk * width)
        place(v, into: &cache.values, at: chunk * width)
        chunks[chunk] = saved
      }
      // The readout needs only the last token after the final layer.
      for chunk in (stage == manifest.stages.count - 1 ? count - 1 : 0)..<count {
        var out = try await timed("attend\(stage)c\(capacity)", attend, [input: chunks[chunk][output]!,
          "query": chunks[chunk]["query"]!, "gate": chunks[chunk]["gate"]!,
          "key": cache.keys, "value": cache.values, "mask": cache.mask])
        guard let hidden = out.remove(output)?.ndArray else { throw HostError("missing attention output") }
        chunks[chunk] = [output: hidden]
      }
    }
    return column(chunks[count - 1][output]!, (tokens - 1) % width)
  }
}
