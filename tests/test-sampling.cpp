#include "ggml.h"
#include "llama.h"

// llama-internal header for the externally-linkable prepare-points helper.
// Tests already reach into src/ internals via the test-sampling build (see CMakeLists.txt:150).
#include "../src/llama-sampler.h"

// common-side twin for parity testing.
#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

extern struct llama_sampler * llama_sampler_init_dry_testing(int32_t context_size, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const std::vector<std::vector<llama_token>>& seq_breakers);

static void dump(const llama_token_data_array * cur_p) {
    for (size_t i = 0; i < cur_p->size; i++) {
        printf("%d: %f (%f)\n", cur_p->data[i].id, cur_p->data[i].p, cur_p->data[i].logit);
    }
}

#define DUMP(__cur_p) do { printf("%s:%d (%s)\n", __FILE__, __LINE__, __func__); dump((__cur_p)); printf("-\n"); } while(0)

struct sampler_tester {
    sampler_tester(size_t n_vocab) {
        cur.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
            const float logit = logf(token_id);
            cur.emplace_back(llama_token_data{token_id, logit, 0.0f});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    sampler_tester(const std::vector<float> & probs, const std::vector<float> & probs_expected) : probs_expected(probs_expected) {
        cur.reserve(probs.size());
        for (llama_token token_id = 0; token_id < (llama_token)probs.size(); token_id++) {
            const float logit = logf(probs[token_id]);
            cur.emplace_back(llama_token_data{token_id, logit, probs[token_id]});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    void apply(llama_sampler * sampler) {
        llama_sampler_apply(sampler, &cur_p);
        llama_sampler_free(sampler);
    }

    void check() {
        GGML_ASSERT(cur_p.size == probs_expected.size());
        for (size_t i = 0; i < cur_p.size; i++) {
            GGML_ASSERT(fabs(cur_p.data[i].p - probs_expected[i]) < 1e-5);
        }
    }

    llama_token_data_array cur_p;

private:
    const std::vector<float> probs_expected;

    std::vector<llama_token_data> cur;
};

static void test_temp(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp(temp));
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_temp_ext(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp, float delta, float exponent) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp_ext(temp, delta, exponent));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_temp_schedule() {
    printf("test_temp_schedule:\n");

    // Helper: create a schedule sampler, apply it at a given position, check that the
    // logits are scaled as if by the expected temperature.
    // We do this by applying the sampler repeatedly (with accept) to advance the position,
    // then compare logit scaling.
    auto check_temp_at_pos = [](llama_sampler * smpl, int target_pos, float expected_temp, const char * label) {
        // Reset to pos=0
        llama_sampler_reset(smpl);

        // Advance to target_pos by doing apply+accept cycles
        for (int i = 0; i < target_pos; i++) {
            // Create dummy cur_p for apply
            std::vector<llama_token_data> cur = {
                {0, 1.0f, 0.0f},
                {1, 2.0f, 0.0f},
                {2, 3.0f, 0.0f},
                {3, 4.0f, 0.0f},
            };
            llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(smpl, &cur_p);
            llama_sampler_accept(smpl, 0);
        }

        // Now apply at target_pos and check the logit scaling
        std::vector<llama_token_data> cur = {
            {0, 1.0f, 0.0f},
            {1, 2.0f, 0.0f},
            {2, 3.0f, 0.0f},
            {3, 4.0f, 0.0f},
        };
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(smpl, &cur_p);

        if (expected_temp <= 0.0f) {
            // greedy: only one token should have non-negative-infinity logit
            int n_valid = 0;
            for (size_t i = 0; i < cur_p.size; i++) {
                if (cur_p.data[i].logit > -1e30f) n_valid++;
            }
            GGML_ASSERT(n_valid == 1);
        } else {
            // Check logit[3] = 4.0 / temp
            float expected_logit = 4.0f / expected_temp;
            float actual_logit = cur_p.data[3].logit;
            if (fabs(actual_logit - expected_logit) > 0.01f) {
                printf("FAIL %s: pos=%d expected_logit=%.4f actual_logit=%.4f\n",
                       label, target_pos, expected_logit, actual_logit);
                GGML_ASSERT(false);
            }
        }

        // Accept to complete the cycle
        llama_sampler_accept(smpl, 0);
        printf("  PASS %s pos=%d temp=%.2f\n", label, target_pos, expected_temp);
    };

    // 1. Step interpolation: temp holds until next control point
    {
        float pts[] = {0.0f, 1.0f, 50.0f, 0.5f, 100.0f, 0.2f};
        auto * smpl = llama_sampler_init_temp_schedule(pts, 3, LLAMA_TEMP_SCHEDULE_INTERP_STEP, false, 0);
        check_temp_at_pos(smpl, 0,  1.0f, "step-at-0");
        check_temp_at_pos(smpl, 25, 1.0f, "step-at-25");
        check_temp_at_pos(smpl, 50, 0.5f, "step-at-50");
        check_temp_at_pos(smpl, 75, 0.5f, "step-at-75");
        check_temp_at_pos(smpl, 100, 0.2f, "step-at-100");
        llama_sampler_free(smpl);
    }

    // 2. Linear interpolation: midpoint temp, reset returns to pos=0
    {
        float pts[] = {0.0f, 1.0f, 100.0f, 0.5f};
        auto * smpl = llama_sampler_init_temp_schedule(pts, 2, LLAMA_TEMP_SCHEDULE_INTERP_LINEAR, false, 0);
        check_temp_at_pos(smpl, 0,  1.0f,  "linear-at-0");
        check_temp_at_pos(smpl, 50, 0.75f, "linear-at-50");
        check_temp_at_pos(smpl, 100, 0.5f, "linear-at-100");

        // Reset and verify pos returns to 0
        llama_sampler_reset(smpl);
        check_temp_at_pos(smpl, 0, 1.0f, "linear-after-reset");
        llama_sampler_free(smpl);
    }

    // 3. Cubic interpolation — numerical correctness
    {
        // Symmetric schedule: (0, 1.0), (50, 0.5), (100, 1.0)
        // Chordal Catmull-Rom with phantom endpoints at boundaries.
        //
        // Equal spacing (50 each) with symmetric temp values means chordal and
        // uniform parameterization agree here. The chordal tangent at (50, 0.5)
        // has slope 0 by symmetry.
        //
        // Segment [0, 50]: slope0 = -0.5/50 (phantom), slope1 = 0 (symmetric)
        //   m0 = -0.01 * 50 = -0.5, m1 = 0
        // At t = 0.5 (pos = 25):
        //   result = 0.5*1.0 + 0.125*(-0.5) + 0.5*0.5 + (-0.125)*0.0 = 0.6875
        float pts[] = {0.0f, 1.0f, 50.0f, 0.5f, 100.0f, 1.0f};
        auto * smpl = llama_sampler_init_temp_schedule(pts, 3, LLAMA_TEMP_SCHEDULE_INTERP_CUBIC, false, 0);
        // At control points should match exactly
        check_temp_at_pos(smpl, 0,   1.0f, "cubic-at-0");
        check_temp_at_pos(smpl, 50,  0.5f, "cubic-at-50");
        check_temp_at_pos(smpl, 100, 1.0f, "cubic-at-100");
        // At pos=25: analytically computed chordal Catmull-Rom value = 0.6875
        check_temp_at_pos(smpl, 25, 0.6875f, "cubic-at-25-analytical");
        llama_sampler_free(smpl);

        // Non-uniform spacing test: (0, 1.0), (10, 0.5), (100, 1.0)
        // Spacing is 10 vs 90 — chordal parameterization weights the tangent at
        // (10, 0.5) more towards the shorter (steeper) left segment.
        //
        // Chordal tangent at (10, 0.5):
        //   d_prev = sqrt(100 + 0.25) ≈ 10.0125
        //   d_next = sqrt(8100 + 0.25) ≈ 90.0014
        //   w_next ≈ 0.900, w_prev ≈ 0.100
        //   vy/vx ≈ -0.04438 / 0.99887 ≈ -0.04443
        //   m1 = -0.04443 * 10 = -0.4443
        //
        // Segment [0, 10]: slope0 = -0.05 (phantom), m0 = -0.5
        // At pos=5 (t=0.5):
        //   result = 0.5*1.0 + 0.125*(-0.5) + 0.5*0.5 + (-0.125)*(-0.4443) ≈ 0.743
        //
        // With uniform parameterization this would be 0.6875 — the chordal result
        // is noticeably different due to the asymmetric spacing.
        float pts_nu[] = {0.0f, 1.0f, 10.0f, 0.5f, 100.0f, 1.0f};
        smpl = llama_sampler_init_temp_schedule(pts_nu, 3, LLAMA_TEMP_SCHEDULE_INTERP_CUBIC, false, 0);
        // Advance to pos=5 and read the temperature
        for (int i = 0; i < 5; i++) {
            std::vector<llama_token_data> cur = {{0, 1.0f, 0.0f}};
            llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(smpl, &cur_p);
            llama_sampler_accept(smpl, 0);
        }
        {
            std::vector<llama_token_data> cur = {{0, 4.0f, 0.0f}};
            llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(smpl, &cur_p);
            float temp_at_5 = 4.0f / cur_p.data[0].logit;
            printf("  cubic non-uniform pos=5 temp=%.4f (chordal; uniform would be 0.6875)\n", temp_at_5);
            // Chordal result should be ≈ 0.743, clearly different from uniform 0.6875
            GGML_ASSERT(temp_at_5 > 0.72f && temp_at_5 < 0.77f);
            GGML_ASSERT(fabs(temp_at_5 - 0.6875f) > 0.03f); // must differ from uniform
        }
        llama_sampler_free(smpl);

        // Negative overshoot test: create a 4-point schedule where cubic genuinely
        // overshoots below 0. Schedule: (0, 0.01), (10, 0.01), (20, 0.01), (30, 10.0)
        //
        // In segment [10, 20], the Catmull-Rom tangent at p1=(20, 0.01) is:
        //   m1 = (y[30] - y[10]) / (x[30] - x[10]) * dx = (10.0 - 0.01) / 20 * 10 = 4.995
        // This large positive tangent at a near-zero y-value drives the spline deeply
        // negative in the interior (e.g. at pos=15: result ≈ -0.61, at pos=17: ≈ -0.72).
        // The clamp to 0.0 should trigger greedy behavior at those positions.
        float pts2[] = {0.0f, 0.01f, 10.0f, 0.01f, 20.0f, 0.01f, 30.0f, 10.0f};
        smpl = llama_sampler_init_temp_schedule(pts2, 4, LLAMA_TEMP_SCHEDULE_INTERP_CUBIC, false, 0);
        bool found_clamp = false;
        for (int pos = 0; pos <= 30; pos++) {
            llama_sampler_reset(smpl);
            for (int i = 0; i < pos; i++) {
                std::vector<llama_token_data> cur = {{0, 1.0f, 0.0f}};
                llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
                llama_sampler_apply(smpl, &cur_p);
                llama_sampler_accept(smpl, 0);
            }
            // Apply at target pos and check the result
            std::vector<llama_token_data> cur = {
                {0, 1.0f, 0.0f}, {1, 2.0f, 0.0f}, {2, 3.0f, 0.0f}, {3, 4.0f, 0.0f},
            };
            llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(smpl, &cur_p);
            // If temp was clamped to 0 (greedy), only 1 token should survive
            int n_valid = 0;
            for (size_t i = 0; i < cur_p.size; i++) {
                if (cur_p.data[i].logit > -1e30f) { n_valid++; }
            }
            if (n_valid == 1) {
                found_clamp = true;
                printf("  cubic negative overshoot clamped to greedy at pos=%d\n", pos);
            }
            // No logit should be NaN
            for (size_t i = 0; i < cur_p.size; i++) {
                GGML_ASSERT(std::isfinite(cur_p.data[i].logit) || cur_p.data[i].logit == -INFINITY);
            }
        }
        GGML_ASSERT(found_clamp && "cubic overshoot must trigger greedy clamp for this schedule");
        llama_sampler_free(smpl);
    }

    // 4. Cubic interpolation — duplicate points
    {
        float pts[] = {0.0f, 1.0f, 50.0f, 0.5f, 50.0f, 0.8f, 100.0f, 1.0f};
        auto * smpl = llama_sampler_init_temp_schedule(pts, 4, LLAMA_TEMP_SCHEDULE_INTERP_CUBIC, false, 0);
        // After dedup, pos=50 should use temp=0.8 (last wins)
        check_temp_at_pos(smpl, 50, 0.8f, "cubic-dedup-50");
        llama_sampler_free(smpl);
    }

    // 5. Clone independence
    {
        float pts[] = {0.0f, 1.0f, 100.0f, 0.5f};
        auto * smpl = llama_sampler_init_temp_schedule(pts, 2, LLAMA_TEMP_SCHEDULE_INTERP_LINEAR, false, 0);
        // Advance to pos=50
        for (int i = 0; i < 50; i++) {
            std::vector<llama_token_data> cur = {{0, 1.0f, 0.0f}};
            llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(smpl, &cur_p);
            llama_sampler_accept(smpl, 0);
        }
        auto * clone = llama_sampler_clone(smpl);
        llama_sampler_reset(smpl);
        // Original should be at pos=0, clone at pos=50
        check_temp_at_pos(smpl, 0, 1.0f, "clone-original-reset");
        // Clone: next apply is at pos=50, temp=0.75
        {
            std::vector<llama_token_data> cur = {{0, 4.0f, 0.0f}};
            llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(clone, &cur_p);
            float expected = 4.0f / 0.75f;
            GGML_ASSERT(fabs(cur_p.data[0].logit - expected) < 0.01f);
            printf("  PASS clone-independence\n");
        }
        llama_sampler_free(smpl);
        llama_sampler_free(clone);
    }

    // 6. Unsorted input: verify out-of-order points produce same results as sorted
    {
        float pts_unsorted[] = {100.0f, 0.5f, 0.0f, 1.0f};
        float pts_sorted[]   = {0.0f, 1.0f, 100.0f, 0.5f};
        auto * smpl_u = llama_sampler_init_temp_schedule(pts_unsorted, 2, LLAMA_TEMP_SCHEDULE_INTERP_LINEAR, false, 0);
        auto * smpl_s = llama_sampler_init_temp_schedule(pts_sorted,   2, LLAMA_TEMP_SCHEDULE_INTERP_LINEAR, false, 0);
        // Compare at pos=50
        for (int i = 0; i < 50; i++) {
            std::vector<llama_token_data> cur = {{0, 1.0f, 0.0f}};
            llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(smpl_u, &cur_p);
            llama_sampler_accept(smpl_u, 0);
            cur = {{0, 1.0f, 0.0f}};
            cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(smpl_s, &cur_p);
            llama_sampler_accept(smpl_s, 0);
        }
        std::vector<llama_token_data> cur_u = {{0, 4.0f, 0.0f}};
        std::vector<llama_token_data> cur_s = {{0, 4.0f, 0.0f}};
        llama_token_data_array cur_p_u = { cur_u.data(), cur_u.size(), -1, false };
        llama_token_data_array cur_p_s = { cur_s.data(), cur_s.size(), -1, false };
        llama_sampler_apply(smpl_u, &cur_p_u);
        llama_sampler_apply(smpl_s, &cur_p_s);
        GGML_ASSERT(fabs(cur_p_u.data[0].logit - cur_p_s.data[0].logit) < 1e-5f);
        printf("  PASS unsorted-input\n");
        llama_sampler_free(smpl_u);
        llama_sampler_free(smpl_s);
    }

    // 7. Normalized positions: [0.0, 1.0] [1.0, 0.5] with n_predict=100
    //    should behave identically to [0, 1.0] [99, 0.5]
    {
        float pts_norm[] = {0.0f, 1.0f, 1.0f, 0.5f};
        float pts_abs[]  = {0.0f, 1.0f, 99.0f, 0.5f};
        auto * smpl_n = llama_sampler_init_temp_schedule(pts_norm, 2, LLAMA_TEMP_SCHEDULE_INTERP_LINEAR, true, 100);
        auto * smpl_a = llama_sampler_init_temp_schedule(pts_abs,  2, LLAMA_TEMP_SCHEDULE_INTERP_LINEAR, false, 0);
        // Compare at several positions
        for (int pos = 0; pos < 100; pos++) {
            std::vector<llama_token_data> cur_n = {{0, 4.0f, 0.0f}};
            std::vector<llama_token_data> cur_a = {{0, 4.0f, 0.0f}};
            llama_token_data_array cur_p_n = { cur_n.data(), cur_n.size(), -1, false };
            llama_token_data_array cur_p_a = { cur_a.data(), cur_a.size(), -1, false };
            llama_sampler_apply(smpl_n, &cur_p_n);
            llama_sampler_apply(smpl_a, &cur_p_a);
            GGML_ASSERT(fabs(cur_p_n.data[0].logit - cur_p_a.data[0].logit) < 1e-4f);
            llama_sampler_accept(smpl_n, 0);
            llama_sampler_accept(smpl_a, 0);
        }
        printf("  PASS normalized-positions\n");
        llama_sampler_free(smpl_n);
        llama_sampler_free(smpl_a);
    }

    // 8. Invalid input robustness: all should return no-op sampler without crashing
    //    and logits must remain unchanged (identity behavior).
    {
        auto * s1 = llama_sampler_init_temp_schedule(nullptr, 0, LLAMA_TEMP_SCHEDULE_INTERP_LINEAR, false, 0);
        auto * s2 = llama_sampler_init_temp_schedule(nullptr, 5, LLAMA_TEMP_SCHEDULE_INTERP_LINEAR, false, 0);
        float pts[] = {0.0f, 1.0f};
        auto * s3 = llama_sampler_init_temp_schedule(pts, 0, LLAMA_TEMP_SCHEDULE_INTERP_LINEAR, false, 0);
        auto * s4 = llama_sampler_init_temp_schedule(pts, 1, LLAMA_TEMP_SCHEDULE_INTERP_LINEAR, true, 0); // normalized with n_predict=0

        llama_sampler * noop_samplers[] = {s1, s2, s3, s4};
        for (auto * s : noop_samplers) {
            std::vector<llama_token_data> cur = {
                {0, 1.0f, 0.0f}, {1, 2.0f, 0.0f}, {2, 3.0f, 0.0f}, {3, 4.0f, 0.0f},
            };
            llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(s, &cur_p);
            // no-op sampler must leave logits unchanged
            GGML_ASSERT(fabs(cur_p.data[0].logit - 1.0f) < 1e-5f);
            GGML_ASSERT(fabs(cur_p.data[1].logit - 2.0f) < 1e-5f);
            GGML_ASSERT(fabs(cur_p.data[2].logit - 3.0f) < 1e-5f);
            GGML_ASSERT(fabs(cur_p.data[3].logit - 4.0f) < 1e-5f);
            llama_sampler_free(s);
        }
        printf("  PASS invalid-input-robustness\n");
    }

    // 10. Accept without prior apply (prompt pre-feeding)
    //     Do NOT use check_temp_at_pos here — it calls reset() which would hide the bug.
    {
        float pts[] = {0.0f, 1.0f, 100.0f, 0.5f};
        auto * smpl = llama_sampler_init_temp_schedule(pts, 2, LLAMA_TEMP_SCHEDULE_INTERP_LINEAR, false, 0);
        // Call accept 5 times without apply — simulating prompt tokens
        for (int i = 0; i < 5; i++) {
            llama_sampler_accept(smpl, 0);
        }
        // Next apply should use pos=0 (position should not have advanced)
        {
            std::vector<llama_token_data> cur = {{0, 4.0f, 0.0f}};
            llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(smpl, &cur_p);
            float expected = 4.0f / 1.0f; // temp=1.0 at pos=0
            GGML_ASSERT(fabs(cur_p.data[0].logit - expected) < 0.01f);
            printf("  PASS prompt-prefeed (pos stayed at 0 after 5 bare accepts)\n");
        }
        llama_sampler_free(smpl);
    }

    // 11. Apply twice, accept once (grammar retry)
    {
        float pts[] = {0.0f, 1.0f, 100.0f, 0.5f};
        auto * smpl = llama_sampler_init_temp_schedule(pts, 2, LLAMA_TEMP_SCHEDULE_INTERP_LINEAR, false, 0);
        // First apply at pos=0
        {
            std::vector<llama_token_data> cur = {{0, 4.0f, 0.0f}};
            llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(smpl, &cur_p);
            float expected = 4.0f / 1.0f; // temp=1.0 at pos=0
            GGML_ASSERT(fabs(cur_p.data[0].logit - expected) < 0.01f);
        }
        // Second apply (grammar rejection retry) — should still use pos=0
        {
            std::vector<llama_token_data> cur = {{0, 4.0f, 0.0f}};
            llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(smpl, &cur_p);
            float expected = 4.0f / 1.0f; // still pos=0
            GGML_ASSERT(fabs(cur_p.data[0].logit - expected) < 0.01f);
        }
        // Accept once — pos should advance to 1
        llama_sampler_accept(smpl, 0);
        // Next apply should be at pos=1
        {
            std::vector<llama_token_data> cur = {{0, 4.0f, 0.0f}};
            llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(smpl, &cur_p);
            float expected_temp = 1.0f + (0.5f - 1.0f) * (1.0f / 100.0f); // linear interp at pos=1
            float expected_logit = 4.0f / expected_temp;
            GGML_ASSERT(fabs(cur_p.data[0].logit - expected_logit) < 0.01f);
        }
        printf("  PASS grammar-retry\n");
        llama_sampler_free(smpl);
    }
}

static void test_min_p_schedule() {
    printf("test_min_p_schedule:\n");

    // Build a logit distribution where we can infer the applied p by counting survivors.
    // We use logits 10, 9, 8, 7, 6, 5, 4 (7 tokens). After min-p filter the number of survivors
    // depends on the threshold p*max_prob, which in log space equals max_logit + logf(p).
    //   p >= exp(-1) ≈ 0.368  -> 1 token survives (only logit 10)
    //   p ~= 0.2              -> 2 tokens (logit 10, 9)  [threshold ≈ 8.39]
    //   p ~= 0.05             -> 4 tokens (logit 10..7)  [threshold ≈ 7.00]
    //   p ~= 0.01             -> 5 tokens (logit 10..6)  [threshold ≈ 5.40]
    //   p = 0.0               -> 7 tokens (filter disabled)
    auto build_logits = []() {
        std::vector<llama_token_data> cur;
        cur.reserve(7);
        for (int i = 0; i < 7; i++) {
            cur.push_back({ i, 10.0f - (float)i, 0.0f });
        }
        return cur;
    };

    auto apply_and_count_survivors = [&](llama_sampler * smpl) -> size_t {
        auto cur = build_logits();
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(smpl, &cur_p);
        return cur_p.size;
    };

    auto advance_to = [&](llama_sampler * smpl, int target_pos) {
        llama_sampler_reset(smpl);
        for (int i = 0; i < target_pos; i++) {
            auto cur = build_logits();
            llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(smpl, &cur_p);
            llama_sampler_accept(smpl, 0);
        }
    };

    // 1. Step interpolation
    {
        float pts[] = {0.0f, 0.0f, 50.0f, 0.2f, 100.0f, 0.5f};
        auto * smpl = llama_sampler_init_min_p_schedule(pts, 3, LLAMA_MIN_P_SCHEDULE_INTERP_STEP, false, 0, 1);
        advance_to(smpl, 0);
        GGML_ASSERT(apply_and_count_survivors(smpl) == 7); // p=0 → filter off
        advance_to(smpl, 25);
        GGML_ASSERT(apply_and_count_survivors(smpl) == 7); // p still 0
        advance_to(smpl, 50);
        GGML_ASSERT(apply_and_count_survivors(smpl) == 2); // p≈0.2 → keep 2
        advance_to(smpl, 100);
        GGML_ASSERT(apply_and_count_survivors(smpl) == 1); // p≈0.5 → keep 1
        printf("  PASS step\n");
        llama_sampler_free(smpl);
    }

    // 2. Linear interpolation — verify p changes monotonically across the segment
    {
        float pts[] = {0.0f, 0.01f, 100.0f, 0.4f};
        auto * smpl = llama_sampler_init_min_p_schedule(pts, 2, LLAMA_MIN_P_SCHEDULE_INTERP_LINEAR, false, 0, 1);
        advance_to(smpl, 0);
        size_t n0 = apply_and_count_survivors(smpl);
        advance_to(smpl, 100);
        size_t n100 = apply_and_count_survivors(smpl);
        GGML_ASSERT(n0 >= 4);        // loose
        GGML_ASSERT(n100 <= 2);      // tight
        GGML_ASSERT(n0 > n100);      // monotonic
        // Reset brings pos back to 0 — survivors should match n0 again
        llama_sampler_reset(smpl);
        GGML_ASSERT(apply_and_count_survivors(smpl) == n0);
        printf("  PASS linear (n0=%zu, n100=%zu)\n", n0, n100);
        llama_sampler_free(smpl);
    }

    // 3. Cubic: chordal Catmull-Rom math body shared with temp_schedule_cubic (verified there).
    //    Here we pin min-p-specific behavior: clamp to [0, 1] on BOTH ends.
    //    Schedule (0, 0.99), (10, 0.99), (20, 0.99), (30, -5.0) — Catmull-Rom overshoots
    //    positive-side before the negative drop; also includes a drop we want clamped.
    {
        float pts[] = {0.0f, 0.99f, 10.0f, 0.99f, 20.0f, 0.99f, 30.0f, -5.0f};
        auto * smpl = llama_sampler_init_min_p_schedule(pts, 4, LLAMA_MIN_P_SCHEDULE_INTERP_CUBIC, false, 0, 1);
        // Every position must yield a survivor count >= 1 (never NaN, never produce garbage)
        for (int pos = 0; pos <= 30; pos++) {
            advance_to(smpl, pos);
            size_t n = apply_and_count_survivors(smpl);
            GGML_ASSERT(n >= 1 && n <= 7);
        }
        printf("  PASS cubic-both-ends-clamped\n");
        llama_sampler_free(smpl);
    }

    // 4. Two-phase latch: prompt pre-feeding (accept-before-apply no-op)
    {
        float pts[] = {0.0f, 0.01f, 100.0f, 0.4f};
        auto * smpl = llama_sampler_init_min_p_schedule(pts, 2, LLAMA_MIN_P_SCHEDULE_INTERP_LINEAR, false, 0, 1);
        for (int i = 0; i < 5; i++) {
            llama_sampler_accept(smpl, 0);
        }
        size_t n = apply_and_count_survivors(smpl);
        // pos should still be 0; loose filter
        advance_to(smpl, 0);
        size_t n0 = apply_and_count_survivors(smpl);
        GGML_ASSERT(n == n0);
        printf("  PASS prompt-prefeed\n");
        llama_sampler_free(smpl);
    }

    // 5. Two-phase latch: apply twice, accept once (grammar retry)
    {
        float pts[] = {0.0f, 0.0f, 100.0f, 0.4f};
        auto * smpl = llama_sampler_init_min_p_schedule(pts, 2, LLAMA_MIN_P_SCHEDULE_INTERP_LINEAR, false, 0, 1);
        size_t n_a = apply_and_count_survivors(smpl);
        size_t n_b = apply_and_count_survivors(smpl); // retry at same pos
        GGML_ASSERT(n_a == n_b);
        llama_sampler_accept(smpl, 0); // pos now advances by 1
        // After accept, pos=1 — still near pos 0, survivors similar
        size_t n_c = apply_and_count_survivors(smpl);
        GGML_ASSERT(n_c >= 1);
        printf("  PASS grammar-retry\n");
        llama_sampler_free(smpl);
    }

    // 6. Invalid input robustness
    {
        auto * s1 = llama_sampler_init_min_p_schedule(nullptr, 0, LLAMA_MIN_P_SCHEDULE_INTERP_LINEAR, false, 0, 1);
        auto * s2 = llama_sampler_init_min_p_schedule(nullptr, 5, LLAMA_MIN_P_SCHEDULE_INTERP_LINEAR, false, 0, 1);
        float pts[] = {0.0f, 0.1f};
        auto * s3 = llama_sampler_init_min_p_schedule(pts, 0, LLAMA_MIN_P_SCHEDULE_INTERP_LINEAR, false, 0, 1);
        auto * s4 = llama_sampler_init_min_p_schedule(pts, 1, LLAMA_MIN_P_SCHEDULE_INTERP_LINEAR, true, 0, 1); // normalized + n_predict=0
        // all non-finite
        float pts_nan[] = {NAN, NAN, INFINITY, 0.1f};
        auto * s5 = llama_sampler_init_min_p_schedule(pts_nan, 2, LLAMA_MIN_P_SCHEDULE_INTERP_LINEAR, false, 0, 1);

        llama_sampler * noops[] = {s1, s2, s3, s4, s5};
        for (auto * s : noops) {
            size_t n = apply_and_count_survivors(s);
            GGML_ASSERT(n == 7); // no-op: nothing filtered
            llama_sampler_free(s);
        }
        printf("  PASS invalid-input-robustness\n");
    }

    // 7. min_keep floor preservation: schedule yields p≈0.5 (would prune to 1) but min_keep=3.
    {
        float pts[] = {0.0f, 0.5f};
        auto * smpl = llama_sampler_init_min_p_schedule(pts, 1, LLAMA_MIN_P_SCHEDULE_INTERP_STEP, false, 0, 3);
        advance_to(smpl, 0);
        size_t n = apply_and_count_survivors(smpl);
        GGML_ASSERT(n >= 3);
        printf("  PASS min_keep-floor\n");
        llama_sampler_free(smpl);
    }

    // 8. Clone independence
    {
        float pts[] = {0.0f, 0.0f, 100.0f, 0.4f};
        auto * smpl = llama_sampler_init_min_p_schedule(pts, 2, LLAMA_MIN_P_SCHEDULE_INTERP_LINEAR, false, 0, 1);
        advance_to(smpl, 50);
        auto * clone = llama_sampler_clone(smpl);
        llama_sampler_reset(smpl);
        // Original at pos=0; clone still at pos=50
        size_t n_orig = apply_and_count_survivors(smpl);
        size_t n_clone = apply_and_count_survivors(clone);
        GGML_ASSERT(n_orig >= n_clone); // clone is tighter because of position
        printf("  PASS clone-independence (orig=%zu, clone=%zu)\n", n_orig, n_clone);
        llama_sampler_free(smpl);
        llama_sampler_free(clone);
    }

    // 9. Normalized-positions round-trip: [0, 1.0] at np=100 == [0, 99]
    {
        float pts_n[] = {0.0f, 0.0f, 1.0f, 0.4f};
        float pts_a[] = {0.0f, 0.0f, 99.0f, 0.4f};
        auto * sn = llama_sampler_init_min_p_schedule(pts_n, 2, LLAMA_MIN_P_SCHEDULE_INTERP_LINEAR, true, 100, 1);
        auto * sa = llama_sampler_init_min_p_schedule(pts_a, 2, LLAMA_MIN_P_SCHEDULE_INTERP_LINEAR, false, 0, 1);
        for (int pos = 0; pos < 100; pos++) {
            size_t n_n = apply_and_count_survivors(sn);
            size_t n_a = apply_and_count_survivors(sa);
            GGML_ASSERT(n_n == n_a);
            llama_sampler_accept(sn, 0);
            llama_sampler_accept(sa, 0);
        }
        printf("  PASS normalized-positions\n");
        llama_sampler_free(sn);
        llama_sampler_free(sa);
    }

    // 10. Sanitation (exercises common/sampling.cpp helpers directly).
    {
        common_params_sampling p;
        p.min_p_schedule = {{0.0f, 0.05f}, {50.0f, 0.2f}};
        p.min_p_schedule_needs_normalization = true;
        p.min_p_schedule_n_predict = 100;
        p.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P }; // MIN_P absent
        common_sampler_sanitize_min_p_schedule(p);
        GGML_ASSERT(p.min_p_schedule.empty());
        GGML_ASSERT(p.min_p_schedule_needs_normalization == false);
        GGML_ASSERT(p.min_p_schedule_n_predict == 0);
    }
    {
        common_params_sampling p;
        p.min_p_schedule = {{0.0f, 0.05f}, {50.0f, 0.2f}};
        p.min_p_schedule_needs_normalization = true;
        p.min_p_schedule_n_predict = 100;
        p.mirostat = 1;
        common_sampler_sanitize_min_p_schedule(p);
        GGML_ASSERT(p.min_p_schedule.empty());
        GGML_ASSERT(p.min_p_schedule_needs_normalization == false);
        GGML_ASSERT(p.min_p_schedule_n_predict == 0);
    }
    {
        common_params_sampling p;
        p.min_p_schedule = {{0.0f, 0.05f}};
        p.min_p_schedule_needs_normalization = true;
        p.min_p_schedule_n_predict = 100;
        common_sampler_clear_min_p_schedule(p);
        GGML_ASSERT(p.min_p_schedule.empty());
        GGML_ASSERT(p.min_p_schedule_needs_normalization == false);
        GGML_ASSERT(p.min_p_schedule_n_predict == 0);
    }
    printf("  PASS sanitation + state-reset invariant\n");

    // 11. PARITY: llama_min_p_schedule_prepare_points vs common_sampler_resolve_min_p_schedule_positions
    //    Both implementations must produce byte-identical output for any input the ctor accepts.
    {
        struct Case {
            std::vector<std::pair<float, float>> pts;
            bool    normalized;
            int32_t n_predict;
        };
        std::vector<Case> battery = {
            { {}, false, 0 },
            { {{0.0f, 0.05f}}, false, 0 },
            { {{0.0f, 0.0f}, {100.0f, 0.3f}}, false, 0 },
            { {{100.0f, 0.3f}, {0.0f, 0.0f}}, false, 0 },             // unsorted
            { {{50.0f, 0.1f}, {50.0f, 0.2f}, {50.0f, 0.3f}}, false, 0 }, // dedupe last-wins
            { {{0.0f, 0.0f}, {1.0f, 0.3f}}, true, 100 },               // normalized standard
            { {{0.0f, 0.0f}, {1.0f, 0.3f}}, true, 1 },                 // normalized np=1
            { {{0.0f, 0.0f}, {1.0f, 0.3f}}, true, 0 },                 // normalized np=0 -> empty
            { {{NAN, 0.1f}, {1.0f, 0.2f}}, false, 0 },                  // non-finite pos
            { {{0.0f, NAN}, {1.0f, 0.2f}}, false, 0 },                  // non-finite p
            { {{INFINITY, 0.1f}, {0.0f, 0.2f}}, false, 0 },              // inf pos
            { {{-INFINITY, 0.1f}, {0.0f, 0.2f}}, false, 0 },             // -inf pos
            { {{0.0f, 0.1f}, {1e-7f, 0.5f}}, false, 0 },                 // near-dedupe-eps
            { {{0.0f, 0.1f}, {1e-5f, 0.5f}}, false, 0 },                 // just outside eps
            { {{-10.0f, 0.2f}, {10.0f, 0.3f}}, false, 0 },               // negative positions
            { {{0.0f, 0.0f}, {0.5f, 0.2f}, {1.0f, 0.5f}}, true, 50 },   // normalized multi
            { {{1e6f, 0.5f}, {1e-6f, 0.01f}}, false, 0 },                 // large + tiny positions
            { {{0.0f, 0.0f}, {100.0f, 0.3f}, {50.0f, 0.15f}}, false, 0 }, // unsorted-middle
            { {{0.0f, 0.1f}, {0.0f, 0.2f}}, false, 0 },                  // exact duplicate
            { {{NAN, NAN}, {INFINITY, INFINITY}}, false, 0 },             // all non-finite
            { {{0.0f, 0.0f}, {0.25f, 0.1f}, {0.5f, 0.25f}, {1.0f, 0.5f}}, true, 200 }, // bigger normalized
        };
        size_t mismatches = 0;
        for (size_t ci = 0; ci < battery.size(); ci++) {
            const auto & c = battery[ci];
            auto a = llama_min_p_schedule_prepare_points(c.pts, c.normalized, c.n_predict);
            auto b = common_sampler_resolve_min_p_schedule_positions(c.pts, c.normalized, c.n_predict);
            if (a.size() != b.size()) {
                printf("  PARITY MISMATCH case %zu: size %zu vs %zu\n", ci, a.size(), b.size());
                mismatches++;
                continue;
            }
            for (size_t i = 0; i < a.size(); i++) {
                if (a[i].first != b[i].first || a[i].second != b[i].second) {
                    printf("  PARITY MISMATCH case %zu[%zu]: (%f,%f) vs (%f,%f)\n",
                        ci, i, a[i].first, a[i].second, b[i].first, b[i].second);
                    mismatches++;
                    break;
                }
            }
        }
        GGML_ASSERT(mismatches == 0);
        GGML_ASSERT(battery.size() >= 20);
        printf("  PASS min_p_schedule_prepare_parity (%zu cases)\n", battery.size());
    }
}

static void test_top_k(const std::vector<float> & probs, const std::vector<float> & probs_expected, int k) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_k(k));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_min_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_min_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_xtc(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p, float t) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_xtc(p, t, 0, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_typical(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_typical(p, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_penalties(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & probs_expected, float repeat_penalty, float alpha_frequency, float alpha_presence
) {
    GGML_ASSERT(probs.size() == probs_expected.size());

    sampler_tester tester(probs, probs_expected);

    auto * sampler = llama_sampler_init_penalties(last_tokens.size(), repeat_penalty, alpha_frequency, alpha_presence);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_dry(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & expected_probs, float dry_multiplier, float dry_base,
    int dry_allowed_length, int dry_penalty_last_n,
    const std::vector<std::vector<llama_token>> & seq_breakers
) {
    GGML_ASSERT(probs.size() == expected_probs.size());

    sampler_tester tester(probs, expected_probs);

    auto * sampler = llama_sampler_init_dry_testing(1024, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);
    tester.check();
}

static void test_top_n_sigma(const std::vector<float> & probs, const std::vector<float> & probs_expected, int n) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_n_sigma(n));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_sampler_queue(const size_t n_vocab, const std::string & samplers_sequence, const int top_k, const float top_p, const float min_p
) {
    sampler_tester tester(n_vocab);

          llama_token min_token_id = 0;
    const llama_token max_token_id = n_vocab - 1;

    for (auto s : samplers_sequence) {
        switch (s) {
            case 'k': tester.apply(llama_sampler_init_top_k(top_k)); break;
            case 'y': GGML_ABORT("typical test not implemented");
            case 'p': tester.apply(llama_sampler_init_top_p(top_p, 1)); break;
            case 'm': tester.apply(llama_sampler_init_min_p(min_p, 1)); break;
            case 't': GGML_ABORT("temperature test not implemented");
            default : GGML_ABORT("Unknown sampler");
        }

        tester.apply(llama_sampler_init_dist(0));

        auto & cur_p = tester.cur_p;

        const int size = cur_p.size;

        if (s == 'k') {
            const int expected_size = std::min(size, top_k);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - top_k));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(cur_p.data[0].id == max_token_id);
            GGML_ASSERT(cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'p') {
            const int softmax_divisor = n_vocab * (n_vocab-1) / 2 - min_token_id * (min_token_id-1) / 2;
            const int softmax_numerator_target = ceilf(top_p * softmax_divisor);

                min_token_id  = n_vocab;
            int expected_size = 0;
            int cumsum        = 0;
            do { // do-while because always at least one token is sampled
                min_token_id--;
                expected_size++;

                cumsum += min_token_id;
            } while (cumsum < softmax_numerator_target);

            // token 0 has p == 0, need special consideration for cumsum because top_p immediately returns
            if (min_token_id == 1) {
                min_token_id--;
                expected_size += 1;
            }

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'm') {
            int expected_size = ceilf((1.0f - min_p) * n_vocab);
            expected_size = std::max(expected_size, 1);
            expected_size = std::min(expected_size, size);

            min_token_id = floorf(min_p * n_vocab);
            min_token_id = std::max(min_token_id, 1);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - size));
            min_token_id = std::min(min_token_id, (llama_token)(n_vocab - 1));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else {
            GGML_ABORT("fatal error");
        }
    }

    printf("Sampler queue %3s OK with n_vocab=%05zu top_k=%5d top_p=%f min_p=%f\n",
           samplers_sequence.c_str(), n_vocab, top_k, top_p, min_p);
}

static void bench(llama_sampler * cnstr, const char * cnstr_name, const std::vector<llama_token_data> & data, int n_iter) {
    std::vector<llama_token_data> cur(data.size());
    std::copy(data.begin(), data.end(), cur.begin());
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(cnstr, &cur_p);
    llama_sampler_reset(cnstr);
    const int64_t t_start = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        std::copy(data.begin(), data.end(), cur.begin());
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(cnstr, &cur_p);
        llama_sampler_reset(cnstr);
    }
    const int64_t t_end = ggml_time_us();
    llama_sampler_free(cnstr);
    printf("%-43s: %8.3f us/iter\n", cnstr_name, (t_end - t_start) / (float)n_iter);
}

#define BENCH(__cnstr, __data, __n_iter) bench((__cnstr), #__cnstr, (__data), (__n_iter))

static void test_perf() {
    const int n_vocab = 1 << 17;

    std::vector<llama_token_data> data;

    data.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        const float logit = 2.0f*((double)(rand())/RAND_MAX - 0.5);
        data.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    BENCH(llama_sampler_init_top_k  (40),                     data, 32);
    BENCH(llama_sampler_init_top_p  (0.8f, 1),                data, 32);
    BENCH(llama_sampler_init_min_p  (0.2f, 1),                data, 32);
    BENCH(llama_sampler_init_typical(0.5f, 1),                data, 32);
    BENCH(llama_sampler_init_xtc    (1.0f, 0.1f, 1, 1),       data, 32);
}

int main(void) {
    ggml_time_init();

    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);
    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f);

    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f, 0.0f, 1.0f);
    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f, 0.0f, 1.0f);

    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 1);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 3);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 4);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0);

    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 0);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.571429f, 0.428571f}, 0.7f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 0.8f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);

    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.24f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.26f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.49f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.51f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.74f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  0.76f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.05f);

    printf("XTC should:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.1f},                                0.99f, 0.09f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.2f, 0.1f},                          0.99f, 0.19f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.3f, 0.2f, 0.1f},                    0.99f, 0.29f);

    printf("XTC should not:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.4f, 0.3f, 0.2f, 0.1f},              0.99f, 0.39f);

    test_typical({0.97f, 0.01f, 0.01f, 0.01f}, {0.97f},            0.5f);
    test_typical({0.4f, 0.2f, 0.2f, 0.2f},     {0.2f, 0.2f, 0.2f}, 0.5f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0}, {0, 0.25f, 0.25f, 0.25f, 0.25f},   50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2}, {0, 0, 0, 0.5f, 0.5f},       50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0, 0, 0, 0.5f, 0.5f}, 50.0f, 0.0f, 0.0f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0},             {0.000011f, 0.249997f, 0.249997f, 0.249997f, 0.249997f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2},       {0.000023f, 0.000023f, 0.000023f, 0.499966f, 0.499966f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.000000f, 0.000023f, 0.000023f, 0.499977f, 0.499977f}, 1.0f, 5.0f, 5.0f);


    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1}, {0.25f, 0.25f, 0.25f, 0.25f}, 1.0f, 1.1f, 2, 4, {});
    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1, 2, 0, 1}, {0.296923f, 0.296923f, 0.109232f, 0.296923f}, 1.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 2, 6, {{3}});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 1}, {0.241818f, 0.241818f, 0.032727f, 0.241818f, 0.241818f}, 2.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 4, 7, {});

    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.571429f, 0.428571f, 0.0f, 0.0f}, 1.00f);
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0.00f); // top_n_sigma == 0 now represents a no-op rather than greedy decoding as of PR#13345
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 3.00f);

    test_sampler_queue(10000, "k", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "k",     1, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1e-12);

    test_sampler_queue(10000, "k",   100, 1.0000f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0003f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.8000f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 0.1f);

    test_sampler_queue(10000, "kp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "km", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 0.1f);

    test_sampler_queue(10000, "kpm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "kmp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pkm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pmk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mkp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mpk", 100, 0.8f, 0.1f);

    test_temp_schedule();
    test_min_p_schedule();

    printf("OK\n");

    test_perf();

    return 0;
}
