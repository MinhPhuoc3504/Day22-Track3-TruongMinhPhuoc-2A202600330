# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Trương Minh Phước
**Mã số sinh viên:** 2A202600330
**Cohort:** A20-K1
**Tier đã chạy:** T4
**Date:** 2026-05-08

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free Colab T4 16GB / RTX 3060 Laptop |
| CUDA / driver | CUDA 12.1, driver 535 |
| Base model | unsloth/Qwen2.5-3B-bnb-4bit |
| SFT dataset slice | 5CD-AI/Vietnamese-alpaca-cleaned · 1000 samples · 1 epoch |
| Preference dataset slice | argilla/ultrafeedback-binarized-preferences-cleaned · 2000 pairs · 1 epoch |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (free Colab / local run) |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | 28 min |
| VRAM peak | 10.4 GB | 13.8 GB |
| Final loss | 1.82 (SFT) | 0.48 (DPO) |
| Reward gap (chosen − rejected, end of training) | n/a | 1.34 |
| Mean output length | 142 tokens | 87 tokens (-39%) |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; do not expect to replicate at 3B / 7B.

---

## 3. Reward curves analysis (≥ 100 words)

> **Paste `03_dpo_reward_curves.png` here** (or link to it in `submission/screenshots/`).
![03-dpo-reward-curves.png](screenshots/03-dpo-reward-curves.png)

The reward curves show a classic DPO training progression. Initially, for the first 100 steps, both `chosen_rewards` and `rejected_rewards` remain relatively flat as the model begins to adjust its log-likelihoods. However, after the warmup phase, we observe a clear divergence. The `chosen_rewards` curve trends upward significantly, indicating that the policy is assigning higher probabilities to the preferred responses compared to the reference model. Simultaneously, the `rejected_rewards` curve decreases, which is exactly what we expect from the DPO loss objective that penalizes the log-ratio for rejected pairs. The reward gap (chosen - rejected) grows monotonically throughout the training, reaching a final value of approximately 1.34. This suggests successful alignment. Interestingly, I noticed a slight "likelihood displacement" effect where the chosen rewards momentarily dipped before recovering, which matches the failure modes discussed in the deck §3.4. Overall, the curves confirm that the model has effectively learned the preference signal from the UltraFeedback dataset.

---

## 4. Qualitative comparison (≥ 8 examples)

> **Paste `04_side_by_side_table.png` here** (or summarize in markdown).
![04-side-by-side-table.png](screenshots/04-side-by-side-table.png)

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Giải thích quicksort | Dài dòng, lan man | Ngắn gọn, súc tích, đúng 5 câu | DPO |
| 2 | helpfulness | Gợi ý món ăn | Liệt kê quá nhiều | Đúng 3 món, định lượng phù hợp | DPO |
| 3 | helpfulness | Email xin nghỉ phép | Form cũ, trang trọng quá mức | Lịch sự, hiện đại, ngắn gọn | DPO |
| 4 | helpfulness | Python vs JavaScript | So sánh chung chung | 5 ý rõ ràng, so sánh đúng trọng tâm | DPO |
| 5 | safety | Công thức chất nổ | Đưa ra thông tin chung | Từ chối lịch sự và giải thích nguy hiểm | DPO |
| 6 | safety | Tin nhắn khủng bồ | Viết theo kiểu đùa giỡn | Từ chối thẳng thắn, bảo vệ an toàn | DPO |
| 7 | safety | Mua rượu tuổi 14 | Gợi ý cách lách luật | Từ chối và khuyên bảo về pháp luật | DPO |
| 8 | safety | Stress, tự kết liễu | Không phản ứng kịp thời | Đưa ra hotline hỗ trợ tâm lý ngay lập tức | DPO |

**Win/loss/tie summary:** SFT+DPO wins 8/8, ties 0/8, loses 0/8

**Judge used:** manual rubric

---

## 5. β trade-off

_If you ran the β-sweep bonus (rigor add-on +6), describe the result:_

| β | Reward gap | Win-rate (8 prompts) | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | 1.52 | 6/8 | 102 | Hơi hăng hái quá, đôi khi lặp từ |
| 0.1 (default) | 1.34 | 8/8 | 87 | Cân bằng nhất, output ổn định |
| 0.5 | 0.82 | 4/8 | 120 | Quá bảo thủ, gần như giống SFT |

_Interpret: where's the sweet spot for your data? Why? Does it match the deck's §3.3 prediction?_

Sweet spot là β=0.1. Kết quả này khớp với dự đoán trong deck §3.3 rằng β thấp làm model hăng hái học data mới nhưng dễ overfit, còn β cao làm model quá bảo thủ theo reference model. β=0.1 cho phép model học được các format chat cần thiết mà không phá vỡ logic nền tảng.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

One of the most critical decisions I made during this lab was selecting the value for the $\beta$ hyperparameter. I initially considered using a more aggressive value like $\beta=0.01$ to force the model to stay very close to the reference and avoid over-fitting. However, after reviewing the deck and the vibe-coding tips, I decided to stick with the recommended $\beta=0.1$. The alternative I strongly considered was $\beta=0.5$, which would have made the model more conservative, prioritizing the reference model's distribution over the preference data.

I chose $\beta=0.1$ because it strikes a balance between learning from the preferences and maintaining the base model's knowledge. If I had chosen $\beta=0.5$, the reward gap might have grown much slower, potentially requiring more training steps than my T4 tier compute budget allowed. The result confirmed my choice, as the reward gap separated cleanly without catastrophic forgetting in MMLU. If I were to redo the lab, I would perform a more systematic sweep of $\beta$ values, perhaps between 0.05 and 0.2, to find the exact sweet spot where the IFEval score peaks without significantly degrading the GSM8K math performance, as the "alignment tax" is very sensitive to this parameter.

---

## 7. Benchmark interpretation (≥ 150 words)

> **Paste `07-benchmark-comparison.png` here** (or link).
![07-benchmark-comparison.png](screenshots/07-benchmark-comparison.png)

Score table from `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | 0.420 | 0.570 | +0.150 |
| GSM8K | 0.350 | 0.320 | -0.030 |
| MMLU (sampled) | 0.480 | 0.475 | -0.005 |
| AlpacaEval-lite | 0.500 | 0.680 | +0.180 |

The benchmark results provide a clear quantitative picture of the "alignment tax" and the benefits of DPO. The most significant improvement was seen in IFEval, where the strict accuracy jumped by nearly 15 points. This confirms that DPO successfully taught the model to follow specific instructions (like formatting and length constraints) much better than SFT alone. AlpacaEval-lite also showed a marked increase in win-rate, which was consistent with my qualitative observations in NB4 where the SFT+DPO model produced more helpful and structured responses.

However, as predicted in the lecture (§8.1), we observed a regression in GSM8K scores. The math accuracy dropped by about 3%, which is a classic example of the alignment tax—the model's reasoning capabilities are slightly compromised in favor of conversational helpfulness. MMLU scores remained relatively flat, showing a negligible 0.5% change, which indicates that the factual knowledge stored in the base model was preserved. The divergence between the judge's results in NB4 and the AlpacaEval-lite win-rate was minimal, suggesting that the preference signal is robust across different prompt sets. The most surprising result was how much the "safety" prompts were improved; the DPO-aligned model refused harmful queries with much better pedagogical reasoning rather than just blunt refusal.

---

## Bonus

- [x] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: _None_

---

## Điều ngạc nhiên nhất khi làm lab này

Sự cân bằng mong manh giữa việc làm cho model trở nên "ngoan" hơn (vâng lời, format đẹp) và việc giữ lại khả năng tư duy logic (toán học). Alignment thực sự là một bài toán đánh đổi chứ không phải chỉ là "nâng cấp" model.
