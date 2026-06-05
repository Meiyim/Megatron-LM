# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import time

from megatron.core import mpu
from megatron.core.inference.communication_utils import broadcast_float_list
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_server.tokenization import tokenize_prompts
from megatron.core.utils import accepts_parameter

logger = logging.getLogger(__name__)

_qps_state = {"count": 0, "window_start": time.time()}


def run_mcore_engine(
    engine,
    prompts=None,
    temperature=1.0,
    top_k=0,
    top_p=0.0,
    logprobs=True,
    tokens_to_generate=0,
    top_n_logprobs=0,
    random_seed=-1,
    stop_words=None,
):
    """Server-compatible version of the MCore Engine, used in
    tools/run_text_generation_server.py."""

    values = [tokens_to_generate, logprobs, top_k, top_p, temperature, top_n_logprobs, random_seed]
    values_float_tensor = broadcast_float_list(len(values), float_list=values, data_parallel=False)
    tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    top_k = int(values_float_tensor[2].item())
    top_p = values_float_tensor[3].item()
    temperature = values_float_tensor[4].item()
    top_n_logprobs = int(values_float_tensor[5].item())
    random_seed = int(values_float_tensor[6].item())

    if random_seed > 0:
        engine.controller.sampling_rng.manual_seed(random_seed)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_segments=True,
        return_log_probs=return_output_log_probs,
        num_tokens_to_generate=tokens_to_generate,
        top_n_logprobs=top_n_logprobs,
        skip_prompt_log_probs=False,
        stop_words=stop_words,
    )

    tokenizer = engine.controller.tokenizer
    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        tokenizer=tokenizer,
        prompts=prompts,
        tokens_to_generate=tokens_to_generate,
        add_BOS=False,
        data_parallel=False,
    )

    tokenized_prompts = []
    for p, l in zip(context_tokens_tensor, context_length_tensor):
        tokenized_prompts.append(p[:l].cpu().numpy().tolist())

    max_seq_len = getattr(
        engine.inference_wrapped_model.inference_context, 'max_sequence_length', None
    )
    if max_seq_len is not None:
        for i, toks in enumerate(tokenized_prompts):
            if len(toks) + tokens_to_generate > max_seq_len:
                raise ValueError(
                    f"Request {i}: prompt_tokens ({len(toks)}) + tokens_to_generate "
                    f"({tokens_to_generate}) = {len(toks) + tokens_to_generate} exceeds "
                    f"max_sequence_length ({max_seq_len})"
                )

    # Log request info and QPS (rank-0 only — it's the Flask server and has authoritative count)
    if mpu.is_pipeline_first_stage() and prompts is not None:
        _qps_state["count"] += 1
        now = time.time()
        elapsed = now - _qps_state["window_start"]
        qps = _qps_state["count"] / elapsed if elapsed > 0 else 0
        for i, toks in enumerate(tokenized_prompts):
            logger.info(
                f"[req {i}] recv prompt_tokens={len(toks)} tokens_to_generate={tokens_to_generate}"
                f" | qps={qps:.2f} total_reqs={_qps_state['count']} window={elapsed:.1f}s"
            )
        if elapsed >= 60:
            _qps_state["count"] = 0
            _qps_state["window_start"] = now

    # Detokenize prompts into strings to pass through the engine
    detokenized_prompts = [
        (
            tokenizer.detokenize(p, skip_special_tokens=True)
            if accepts_parameter(tokenizer.detokenize, "skip_special_tokens")
            else tokenizer.detokenize(p)
        )
        for p in tokenized_prompts
    ]

    requests = []
    for i in range(len(tokenized_prompts)):
        req = InferenceRequest(
            prompt=detokenized_prompts[i],
            prompt_tokens=tokenized_prompts[i],
            sampling_params=sampling_params,
            request_id=engine.get_new_request_id(),
        )
        requests.append(req)

    t0 = time.time()
    result = engine.generate(inference_requests=requests)
    gen_time = time.time() - t0

    # Normalize token fields: prompt_tokens to list, generated_tokens to tensor
    for r in result:
        if hasattr(r, "prompt_tokens") and hasattr(r.prompt_tokens, "tolist"):
            r.prompt_tokens = r.prompt_tokens.tolist()
        if hasattr(r, "generated_tokens") and isinstance(r.generated_tokens, list):
            import torch as _torch
            r.generated_tokens = _torch.tensor(r.generated_tokens, dtype=_torch.long)

    # Only post-process on the server rank (first stage with prompts)
    if mpu.is_pipeline_first_stage() and prompts is not None:
        for i, x in enumerate(result):
            n_gen = len(x.generated_tokens) if x.generated_tokens is not None else 0
            logger.info(
                f"[req {i}] done prompt_tokens={len(tokenized_prompts[i])} requested={tokens_to_generate} generated={n_gen} latency={gen_time:.2f}s"
            )
        response_dict = {
            # Send original prompts, not x.prompt, to circumvent tokenization artifacts
            "text": [p + x.generated_text for p, x in zip(prompts, result)],
            "tokens": [x.prompt_tokens + x.generated_tokens.tolist() for x in result],
        }
        if sampling_params.return_log_probs:
            response_logprobs = [
                (x.prompt_log_probs or []) + (x.generated_log_probs or []) for x in result
            ]
            response_dict["logprobs"] = response_logprobs
        if sampling_params.return_segments:
            response_dict["segments"] = [x.segments for x in result]
        if sampling_params.top_n_logprobs > 0:
            # TODO(ksanthanam): Support enabling `skip_prompt_log_probs`
            assert (
                sampling_params.return_prompt_top_n_logprobs
            ), "skip_prompt_log_probs must be False"
            response_dict["top_n_logprobs"] = [
                x.prompt_top_n_logprobs + x.generated_top_n_logprobs for x in result
            ]

        return response_dict
    return None
