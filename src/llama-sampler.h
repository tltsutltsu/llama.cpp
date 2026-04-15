#pragma once

#include "llama.h"

#include <vector>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

struct llama_sampler * llama_sampler_init_dry_testing(
        int32_t context_size,
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);

// Transform a raw min-p schedule point list the same way llama_sampler_init_min_p_schedule does:
// filter non-finite, optionally normalize (multiply positions by n_predict - 1), stable sort by
// position, and epsilon dedupe with "last wins". Externally linkable so tests can compare this
// against the common-side twin. Not LLAMA_API — internal symbol.
std::vector<std::pair<float, float>> llama_min_p_schedule_prepare_points(
        const std::vector<std::pair<float, float>> & raw,
        bool    normalized,
        int32_t n_predict);
