# CUDA-core Bound (FP32) Model — FLOPs per Cycle (per SM)

**Goal.** Provide a simple, architecture-aware upper bound on FP32 throughput delivered by CUDA cores on a *single* Streaming Multiprocessor (SM), expressed as **FLOPs per cycle**. This is a component-level bound you can combine with other bounds (e.g., memory, SFU) in a larger performance model.

---

## 1) Model at a Glance

We analyze execution in a sliding window of $k$ dynamic instructions. For each window $j$, we compute the FP32 CUDA-core bound:

$$
\boxed{\;\Phi_{\mathrm{SM,CUDA}}^{(j)} \;=\; \theta^{(j)}\, C_{\mathrm{fp32}}\, \big(1 + m_{\mathrm{FMA}}^{(j)}\big)\;}\quad\text{FLOPs/cycle}
$$

This equals the SM’s FP32 lane capacity $C_{\mathrm{fp32}}$ scaled by the *useful-lane fraction* $\theta$ (divergence/predication) and by FMA intensity (each FMA counts as two FLOPs).

> Sanity checks: If all threads are active ($\theta=1$) and all FP32 ops are FMAs ($m_{\mathrm{FMA}}=1$), then $\Phi=2\,C_{\mathrm{fp32}}$ FLOPs/cycle (peak). If half the lanes are active ($\theta=0.5$) and no FMAs ($m_{\mathrm{FMA}}=0$), then $\Phi=0.5\,C_{\mathrm{fp32}}$.

---

## 2) Notation

* $C_{\mathrm{fp32}}$: FP32 CUDA-core lanes per SM (hardware constant for the GPU).
* $N_{\mathrm{warp,fp32}}^{(j)}$: Count of **warp-level FP32 instructions** in window $j$ (i.e., how many FP32 warps are issued in the window).
* $\theta^{(j)}\in[0,1]$: Average *active-thread fraction* for those FP32 warps in the window (captures divergence/masking/predication). $\theta=1$ means all 32 lanes active on average.
* $m_{\mathrm{FMA}}^{(j)}\in[0,1]$: Fraction of the FP32 instructions in the window that are FMAs. Each active thread’s FMA contributes **2 FLOPs**.
* (Optional) $f_{\mathrm{fp32}}^{(j)}\in[0,1]$: Fraction of *all* dynamic instructions in the window that are FP32 (useful if you want the bound to reflect FP32 share within mixed windows).
* (Optional) $u_{\mathrm{issue}}^{(j)}\in[0,1]$: Effective FP32 issue utilization of the SM schedulers (captures dependency stalls, insufficient ILP/warp supply, scoreboard effects, etc.).

---

## 3) Derivation (per window $j$)

**E1) Thread-operations demand**

$$
 n_{\mathrm{ops}}^{(j)} = 32\,\theta^{(j)}\, N_{\mathrm{warp,fp32}}^{(j)}
$$

Each FP32 warp instruction covers up to 32 threads; only a fraction $\theta$ are active on average.

**E2) FLOP demand**

$$
 n_{\mathrm{FLOPs}}^{(j)} = n_{\mathrm{ops}}^{(j)}\,\big(1 + m_{\mathrm{FMA}}^{(j)}\big)
$$

FMAs contribute two FLOPs per active thread; non-FMA FP32 ops contribute one.

**E3) Minimum service time (cycles)**

$$
 T_{\mathrm{core}}^{(j)} = N_{\mathrm{warp,fp32}}^{(j)}\cdot\frac{32}{C_{\mathrm{fp32}}}\quad\text{(cycles)}
$$

A warp instruction consumes an FP32 issue slot regardless of lane masking. Thus the cycles scale with warp count and the SM’s FP32 lane width $C_{\mathrm{fp32}}$ (i.e., threads-per-warp divided by lanes-per-cycle).

**E4) Throughput bound (FLOPs/cycle)**

$$
 \Phi_{\mathrm{SM,CUDA}}^{(j)} = \frac{n_{\mathrm{FLOPs}}^{(j)}}{T_{\mathrm{core}}^{(j)}} = \theta^{(j)}\, C_{\mathrm{fp32}}\, \big(1 + m_{\mathrm{FMA}}^{(j)}\big)
$$

Note that $N_{\mathrm{warp,fp32}}$ cancels, as expected for a capacity bound.

---

## 4) Extended (practical) bound

If your window contains a mix of instruction types or experiences scheduler under-issue, apply:

$$
\boxed{\;\hat{\Phi}_{\mathrm{SM,CUDA}}^{(j)} = u_{\mathrm{issue}}^{(j)}\, f_{\mathrm{fp32}}^{(j)}\, \theta^{(j)}\, C_{\mathrm{fp32}}\, \big(1 + m_{\mathrm{FMA}}^{(j)}\big)\;}
$$

* $f_{\mathrm{fp32}}$ reduces the bound in proportion to the FP32 share of the window.
* $u_{\mathrm{issue}}$ reduces for dependency stalls, low occupancy/ILP, scoreboard waits, etc.

---

## 5) Assumptions & Scope

* Bounds **only** the FP32 CUDA-core pipelines. It ignores tensor cores, SFUs/transcendentals, and memory/LDST limits.
* Divergence/predication is captured via $\theta$; it does **not** shorten service time per warp.
* Architectural details like dual-issue (e.g., FP32+INT32), partitioned schedulers, or per-cycle warp issue are abstracted into $C_{\mathrm{fp32}}$ and (optionally) $u_{\mathrm{issue}}$.
* FLOP counting uses the conventional rule: **FMA = 2 FLOPs**.

---

## 6) Inputs & Measurement Hints

* **$\theta$**: average active-lane ratio across FP32 warps; compute as active-threads / 32 aggregated over FP32 warp instructions in the window.
* **$m_{\mathrm{FMA}}$**: fraction of FP32 instructions that are FMAs in the window: $m_{\mathrm{FMA}} = N_{\mathrm{FMA}}/(N_{\mathrm{FMA}}+N_{\mathrm{ADD}}+N_{\mathrm{MUL}}+\cdots)$.
* **$f_{\mathrm{fp32}}$** (optional): FP32 dynamic instruction share within the window.
* **$u_{\mathrm{issue}}$** (optional): ratio of FP32 issue cycles actually used vs. available in the window.

---

## 7) Worked Example

Suppose $C_{\mathrm{fp32}}=128$ lanes/SM, $\theta=0.75$, $m_{\mathrm{FMA}}=0.6$, and (optionally) $u_{\mathrm{issue}}=0.9$, $f_{\mathrm{fp32}}=0.8$.

**Core bound:**

$$
\Phi = 0.75\times128\times(1+0.6) = 153.6\;\text{FLOPs/cycle}
$$

**Extended bound:**

$$
\hat{\Phi} = 0.9\times0.8\times0.75\times128\times1.6 \approx 110.6\;\text{FLOPs/cycle}
$$

---

## 8) How to Use in a Full Model

* Compute $\Phi_{\mathrm{SM,CUDA}}^{(j)}$ (and other component bounds) over sliding windows; the minimum across components is the per-window performance ceiling.
* Keep the **FLOPs/cycle** unit for architectural clarity. If needed: FLOPs/s per SM = $\Phi\times f_{\mathrm{SM}}$ (SM clock in Hz); GPU-level ≈ SM-sum.

---

## 9) Summary

A per-SM CUDA-core FP32 throughput bound in FLOPs/cycle:

$$
\Phi_{\mathrm{SM,CUDA}}^{(j)} = \theta^{(j)}\, C_{\mathrm{fp32}}\, \big(1 + m_{\mathrm{FMA}}^{(j)}\big)\,,\quad\text{optionally}\;\times\; u_{\mathrm{issue}}^{(j)} f_{\mathrm{fp32}}^{(j)}.
$$

This preserves divergence effects, captures FMA weighting, and ties directly to architectural lane capacity. Use it as the CUDA-core roof within an instruction/roofline-style model.
