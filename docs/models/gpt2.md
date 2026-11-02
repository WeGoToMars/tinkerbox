# GPT-2

OpenAI's original GPT-2 base models, for plain text completion with no chat template. All sizes run on M1–M6 with a 1,024-token context. Requires Tinkerbox 2.3.0.

| Model ID | Parameters |
| --- | ---: |
| `gpt2` | 124M |
| `gpt2-medium` | 355M |
| `gpt2-large` | 774M |
| `gpt2-xl` | 1.56B |

The IDs above are FP16. Append `-int8` or `-lut8` for compressed weights, such as `gpt2-xl-lut8`. INT8 is the smallest. LUT8 stays closer to FP16 output distributions on WikiText-2. The [bundle repository](https://huggingface.co/wegotomars/GPT2-CoreAI-ANE) has sizes and benchmarks.

The transformer runs on the Neural Engine. Tokenization, embeddings, the FP32 vocabulary readout and sampling run on the CPU. In the app, use the **Text Completion** page.

## Usage

```python
import tinkerbox_ane

with tinkerbox_ane.load("gpt2") as model:
    reply = model.complete(prompt="Once upon a time,", max_tokens=32, temperature=0)
    print(reply["choices"][0]["text"])

    replies = model.complete(
        prompt=["The scientist discovered", "The musician played"],
        max_tokens=32, temperature=0.8, top_k=40, seed=42,
    )
```

`complete` follows the [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions/create) for `prompt` (a string or list), `max_tokens` (default 16), `temperature` (default 1; 0 is greedy), `top_p`, `seed`, `n=1` and `stream`. Responses have ordered `choices` with `finish_reason` `stop` or `length`, plus aggregate `usage`. `top_k` and `batch_size` are extensions. Each choice also carries its `token_ids` and `metrics` (input and output tokens, batch size). An empty prompt starts from the BOS token. The prompt plus `max_tokens` must fit in the context.

With `stream=True`, `complete` returns an iterator of chunks whose choices carry text deltas, indexed by prompt. A final chunk per prompt sets `finish_reason`. `stream_options={"include_usage": True}` adds usage to the last chunk. Close an unfinished stream before the next request on the same model.

```python
with tinkerbox_ane.load("gpt2") as model:
    for chunk in model.complete(prompt="Once upon a time,", max_tokens=64, stream=True):
        for choice in chunk["choices"]:
            print(choice["text"], end="", flush=True)
```

`count_tokens(prompt)` returns the prompt length. `token_spans(prompt)` returns each visible token's IDs with UTF-16 offsets, for highlighting.

## Batching

A list of prompts runs as batches of 1, 4 or 8. Prompts can have different lengths and stay isolated. The runtime picks the size from prompt lengths and the output budget; `batch_size=4` caps it at four. A seeded request runs each prompt on its own, so a prompt replays the same text whatever else is in the call. Calls on one model run one at a time.

## Build

See the [developer guide](../model-publishing.md#export-and-compile). Without `--batch-sizes`, exports support batches of 1 and 4. Original weights use the [GPT-2 Modified MIT License](../../LICENSES/GPT-2-MIT.txt).
