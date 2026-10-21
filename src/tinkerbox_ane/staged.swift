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
}

final class StagedRunner: Runner {
  let manifest: StageManifest
  let root: URL
  let width: Int, context: Int, hiddenSize: Int
  var activeModels: [Int: AIModel] = [:]
  var cos: [Float16] = [], sin: [Float16] = [], positions = 0
  var caches: [Int: PromptCache] = [:]

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
    if activeModels[stage] == nil {
      activeModels[stage] = try await loadAsset(root.appendingPathComponent(manifest.assets[stage].asset))
    }
    guard let function = try activeModels[stage]!.loadFunction(named: name) else {
      throw HostError("missing function \(name)")
    }
    return function
  }

  func inputArray(_ function: InferenceFunction, _ name: String) throws -> NDArray {
    var array = NDArray(descriptor: try descriptor(function.descriptor.inputDescriptor(of: name), name))
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
  }

  func hidden(tokens: Int, answers: [Int]?) async throws -> [Float16] {
    // Full attention over the whole prompt leaves no question-independent trunk to share.
    guard answers == nil else { throw HostError("this model answers one question per pass") }
    guard positions > 0, tokens >= 1, tokens <= context,
          let capacity = manifest.tiers.first(where: { $0 >= tokens }) else {
      throw HostError("invalid request")
    }
    let count = (tokens + width - 1) / width
    let first = try await function("scan0", stage: 0)
    var chunks: [[String: NDArray]] = []
    for _ in 0..<count {
      writeAll([0])
      guard let values = readHalves(width * hiddenSize) else {
        throw HostError("incomplete embedding chunk")
      }
      var x = try inputArray(first, input)
      fillInput(&x, values)
      chunks.append([output: x])
    }
    let attendShape = try await function("attend0c\(capacity)", stage: 0)
    let cache = try attention(attendShape, capacity: capacity)
    write4(&cache.mask) { p, shape, strides in
      for t in 0..<shape[3] { p[t * strides[3]] = t < tokens ? 0 : maskMin }
    }
    // RoPE inputs depend only on the chunk position; every stage reuses them.
    var rope: [(NDArray, NDArray)] = []
    for chunk in 0..<count {
      var c = try inputArray(first, "cos"), s = try inputArray(first, "sin")
      for (table, array) in [(cos, 0), (sin, 1)] {
        let fill = { (p: UnsafeMutablePointer<Float16>, shape: [Int], strides: [Int]) in
          for i in 0..<shape[2] { for t in 0..<shape[3] {
            p[i * strides[2] + t * strides[3]] = table[i * self.positions + chunk * self.width + t]
          } }
        }
        if array == 0 { write4(&c, fill) } else { write4(&s, fill) }
      }
      rope.append((c, s))
    }
    for stage in manifest.stages.indices {
      let scan = try await function("scan\(stage)", stage: stage)
      let attend = try await function("attend\(stage)c\(capacity)", stage: stage)
      var state = try inputArray(scan, "packed_state")
      // Finish the causal scans and collect every key before running any full attention.
      for chunk in 0..<count {
        var out = try await scan.run(inputs: [input: chunks[chunk][output]!, "cos": rope[chunk].0,
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
        var out = try await attend.run(inputs: [input: chunks[chunk][output]!,
          "query": chunks[chunk]["query"]!, "gate": chunks[chunk]["gate"]!,
          "key": cache.keys, "value": cache.values, "mask": cache.mask])
        guard let hidden = out.remove(output)?.ndArray else { throw HostError("missing attention output") }
        chunks[chunk] = [output: hidden]
      }
    }
    return column(chunks[count - 1][output]!, (tokens - 1) % width)
  }
}
