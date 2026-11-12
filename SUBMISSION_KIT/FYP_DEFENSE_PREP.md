# FYP DEFENCE PREP — Aasan Version

**Project:** Task-Oriented Deep Joint Source-Channel Coding for UAV Navigation
**Team:** Areesha Sohail, Amna Shahid, Hafiza Laiba Faisal
**Supervisor:** Ismat Hira — LCWU, Electrical Engineering
**Doc date:** 13 Nov 2026

> **Is doc ka maqsad:** viva / defence mein jo bhi poochha jaye, uska jawab aasan lafzon mein,
> sahi number ke saath, aur har equation ke har variable ke meaning ke saath.
>
> **Rule:** Jo cheez is doc mein nahi hai, us par "verified nahi kiya" keh dein. Guess na karein.

---

## 0. Is doc ko kaise use karein

| Kab | Kya karein |
|---|---|
| **3 din pehle** | Part 3 (equations) parh lein — variable tables yaad karne ki koshish na karein, samajhne ki karein |
| **2 din pehle** | Part 8 (results) + Part 14 (number sheet) — sirf 12 numbers yaad karein |
| **1 din pehle** | Part 10 (30 Q&A) + Part 11 (hard questions) — bol kar practice karein |
| **Subah** | Part 9 (claim rules) + Part 16 (checklist) — 10 minute |
| **Viva mein** | Part 0 ka rule: "verified nahi" kehna bura nahi, galat number batana bura hai |

---

## 1. 60-second opening (yaad kar lein)

### English script (~130 words)

> "We studied one question: **when** does a task-oriented neural joint code actually beat a
> conventional digital link for UAV navigation, under a **matched budget**?
>
> We send a 6-dimensional UAV state — position, velocity, and the goal — over AWGN and slow
> Rayleigh fading, using **k = 12 to 54 channel uses per control step**. Three links get exactly
> the same episodes, noise, fading, and bandwidth: a convolutional-coded QPSK link, a
> reconstruction DeepJSCC, and a task-oriented DeepJSCC trained on navigation success.
>
> We find a **bandwidth crossover**: the reconstruction neural link wins at **k ≤ 18**, the coded
> link wins at **k ≥ 24**. The task-trained link with control shaping sits at **0.996–0.999**.
> The frozen task loss **fails at 2–6%**, and we show that this is a **gradient-conditioning**
> problem, not a capacity limit. Finally we map the result onto a **3GPP UMa-aerial link budget**.
>
> The claim is **conditional**, not a new paradigm."

### Urdu version (agar panel Urdu mein poochay)

> "Hum ne ek sawal poochha: **kab** task-oriented neural link digital link se behtar hai, jab
> budget barabar ho? Teen links ko bilkul same episodes, noise, fading aur bandwidth diye.
> Result: kam bandwidth par neural jeetta hai, zyada par digital. Frozen loss 2–6% par fail hota
> hai — woh **training ka masla** hai, capacity ka nahi. Aerial link par bhi same ordering aati hai.
> Hamara claim conditional hai."

---

## 2. Project at a glance

| Cheez | Jawab |
|---|---|
| Thesis title | Task-Oriented Deep Joint Source-Channel Coding for UAV Navigation under Noisy and Fading Wireless Channels — with a 3GPP UMa-Aerial Link-Budget and Dispersion Extension |
| Paper title | *When Does Task-Oriented DeepJSCC Beat a Coded Digital Link? A Matched-Budget Study for UAV Navigation* |
| Source | 6-D UAV state (position, velocity, goal) |
| Budget | k = 12, 18, 24, 30, 54 channel uses per control step |
| Channels | AWGN, slow Rayleigh (block fading), 3GPP UMa-aerial (large-scale only) |
| Links | Digital (conv + QPSK), Reconstruction DeepJSCC, Task-oriented DeepJSCC |
| Main episodes | 5,000 per unit × 3 seeds = 15,000 per condition |
| Extension episodes | 1,000 per unit |
| Code | Python / PyTorch, CPU-only, deterministic, **111 tests** |

### Research questions (safe wording)

| # | Question |
|---|---|
| RQ1 | **Under what conditions, if any,** does a task-oriented DeepJSCC link outperform a matched-budget coded digital link? |
| RQ2 | How sensitive is it to bandwidth, SNR, fading, CSI error, and time correlation? |
| RQ3 | How does the result map onto a 3GPP UMa-aerial link budget? |

> **Zaroori:** RQ1 ko "kya task-oriented DeepJSCC behtar hai?" keh kar na poochhein — us mein
> jawab pehle se maan liya gaya hai. "Under what conditions" kehna sahi hai.

---

# PART 3 — SYSTEM MODEL: HAR EQUATION, HAR VARIABLE

## 3.1 State vector

```
s_t = [ x_t , y_t , vx_t , vy_t , xg , yg ]  ∈ R^6
```

| Symbol | Meaning | Unit |
|---|---|---|
| `s_t` | step `t` par poora state | — |
| `x_t`, `y_t` | UAV ki position | m |
| `vx_t`, `vy_t` | UAV ki velocity | m/s |
| `xg`, `yg` | goal ki position (episode bhar fixed) | m |
| `t` | discrete time step, 0 se 100 | — |
| `R^6` | 6 real numbers ka vector | — |

**Simple baat:** pehle 4 cheezein **badalti** hain (position, velocity), aakhri 2 **fixed** hain (goal).

---

## 3.2 Normalization

```
s̄_i = 2 · ( clip(s_i, a_i, b_i) − a_i ) / ( b_i − a_i ) − 1
```

| Symbol | Meaning | Value / Unit |
|---|---|---|
| `s_i` | i-th component ka raw value | m ya m/s |
| `a_i` | us component ka minimum | position/goal: −8 m; velocity: −5 m/s |
| `b_i` | us component ka maximum | position/goal: +8 m; velocity: +5 m/s |
| `clip(s_i, a_i, b_i)` | value ko `[a_i, b_i]` mein rokna | — |
| `s̄_i` | normalized value | **−1 se +1**, unit-less |

**Inverse (denormalize):**

```
s_i = a_i + (s̄_i + 1)/2 · ( b_i − a_i )      phir [a_i, b_i] par clip
```

**Yaad rakhein:**

- Position/goal ka span = 16 m → **1 normalized unit = 8 m**
- Velocity ka span = 10 m/s → **1 normalized unit = 5 m/s**
- Ranges **config** se aate hain, data se nahi (leakage na ho).

---

## 3.3 Dynamics (double integrator)

```
p_{t+1} = p_t + Δt · v_t + ½ · Δt² · u_t
v_{t+1} = v_t + Δt · u_t
```

| Symbol | Meaning | Unit |
|---|---|---|
| `p_t = [x_t, y_t]` | position | m |
| `v_t = [vx_t, vy_t]` | velocity | m/s |
| `u_t = [ux_t, uy_t]` | control = **commanded acceleration** | m/s² |
| `Δt` | ek step ka time | **0.1 s** |
| `t` | step index | 0 … 99 |

**Matrix form:**

```
s_{t+1} = A · s_t + B · u_t

A = [ I₂   Δt·I₂   0  ]        B = [ ½Δt²·I₂ ]
    [ 0     I₂     0  ]            [   Δt·I₂  ]
    [ 0     0      I₂ ]            [     0    ]
```

| Symbol | Meaning |
|---|---|
| `I₂` | 2×2 identity matrix ("koi change nahi") |
| `0` | 2×2 zero matrix |
| `A` | state transition matrix (purana state naya state banata hai) |
| `B` | control matrix (acceleration state mein kaise jata hai) |

**Kyun `½Δt²`?** Yeh **exact** solution hai jab `u` poore step ke liye constant ho. `u` do dafa
integrate hota hai: pehle velocity, phir position. Isi liye "double integrator".

**Simple baat:** control band kar dein to UAV rukta nahi — **inertia** ki wajah se aage slide karta hai.

---

## 3.4 Velocity limit

```
v ← v · min( 1 , v_max / ||v||₂ )

||v||₂ = sqrt( vx² + vy² )
```

| Symbol | Meaning | Value / Unit |
|---|---|---|
| `||v||₂` | velocity vector ki length | m/s |
| `v_max` | maximum speed | **5 m/s** |
| `min(1, …)` | agar speed limit ke andar hai to 1 (koi change nahi); bahar hai to chhota factor | — |

**Misaal:** `v = (4, 4)` → `||v|| = 5.66 > 5` → factor = 5/5.66 = 0.884 → naya `v = (3.54, 3.54)`.

> **Note:** yeh **norm** se limit hai, per-axis clip se nahi. Per-axis hota to `(4,4)` allow ho jata
> (length 5.66). Thesis mein sirf "5 m/s limit" likha hai — yeh detail viva mein bata dein.

---

## 3.5 PD controller

```
u_t = clip( −Kp · (p̂_t − ĝ_t) − Kv · v̂_t , −u_max , +u_max )
```

| Symbol | Meaning | Value / Unit |
|---|---|---|
| `p̂_t` | **estimated** position (channel se aayi) | m |
| `ĝ_t` | **estimated** goal | m |
| `v̂_t` | **estimated** velocity | m/s |
| `Kp` | proportional gain | **1.0** |
| `Kv` | derivative / damping gain | **2.0** |
| `u_max` | per-axis acceleration limit | **2.0 m/s²** |
| `clip(x, −a, +a)` | har axis ko `±a` mein rokna | — |

**Do hisse:**

- `−Kp(p̂ − ĝ)` = **"goal ki taraf kheencho"** (error jitna bara, force utna bara)
- `−Kv v̂` = **"speed ko damp karo"** (brake)

**Damping ratio:**

```
ζ = Kv / (2·√Kp) = 2 / (2·1) = 1.0     → critically damped
ω_n = √Kp = 1 rad/s                    → natural frequency
```

| Symbol | Meaning |
|---|---|
| `ζ` (zeta) | damping ratio; 1 = critically damped (overshoot nahi) |
| `ω_n` (omega) | system ki natural speed (rad/s) |

> **Honest baat:** `ζ = 1` sirf **small-signal, jab clip na lage** ke liye sahi hai. `u_max` aur
> `v_max` ki wajah se bade errors par system **nonlinear** ho jata hai (pehle full acceleration,
> phir normal PD).

**Important:** controller `p̂, ĝ, v̂` — yani **estimates** use karta hai, ground truth nahi.

---

## 3.6 Control error (ground truth vs estimate)

```
u_t = u_t*  −  Kp·(e_p − e_g)  −  Kv·e_v
```

| Symbol | Meaning |
|---|---|
| `u_t*` | agar state **bilkul sahi** milti to control | 
| `e_p = p̂_t − p_t` | position estimation error (m) |
| `e_g = ĝ_t − g_t` | goal estimation error (m) |
| `e_v = v̂_t − v_t` | velocity estimation error (m/s) |

**Simple baat:** control sirf **relative** position error `(e_p − e_g)` aur velocity error se
kharab hota hai. Agar `p̂` aur `ĝ` **dono** mein same galti ho (common-mode), to control par asar
**zero** — dono ek doosre ko cancel kar dete hain.

---

## 3.7 Steady-state offset (bias ka asar)

Jab UAV goal ke paas ruk jata hai: `u = 0` aur `v = 0`. Tab:

```
p_ss = g + ( b_g − b_p )
```

| Symbol | Meaning |
|---|---|
| `p_ss` | jahan UAV actually rukta hai | 
| `g` | **true** goal |
| `b_p` | position estimate ka constant bias (m) |
| `b_g` | goal estimate ka constant bias (m) |

**Simple baat:** decoded goal mein `0.5 m` se zyada bias ho to UAV **galat jagah** rukega, aur
success radius `0.5 m` se bahar chala jayega. Isi liye goal ka sahi decode hona zaroori hai.

> Note: PD mein **integral term nahi** hai, is liye constant bias khud-ba-khud theek nahi hota.

---

## 3.8 Encoder aur decoder

```
Encoder:  6  →  128  →  128  →  2k  real  →  k complex symbols
Decoder:  2k  →  128  →  128  →  6   (normalized state)
```

| Symbol | Meaning | Value |
|---|---|---|
| `k` | channel uses (complex symbols) per transmission | 12, 18, 24, 30, 54 |
| `2k` | real numbers (har symbol ke 2 hisse: I aur Q) | 24 … 108 |
| `128` | hidden layer width | fixed |
| `PReLU` | activation: negative side par bhi chhota learnable slope | — |

**Kyun `2k`?** Ek complex symbol = **2 real** numbers (`I` + `jQ`). Is liye `k` symbols ke liye
`2k` real outputs chahiye.

**Code reference:** `project/models/deepjscc_encoder.py`, `deepjscc_decoder.py`

---

## 3.9 Power normalization

```
z ← z̃ · sqrt(k · P) / sqrt( ||z̃||² + ε )

||z̃||² = Σ_{i=1..k} |z̃_i|²
```

| Symbol | Meaning | Value |
|---|---|---|
| `z̃` | encoder ka raw output (k complex) | — |
| `z` | normalized transmit symbols | — |
| `k` | symbols per codeword | 12 … 54 |
| `P` | average power per symbol | **1.0** |
| `ε` | numerical guard (divide-by-zero se bachne ke liye) | **1e−12** |
| `||z̃||²` | poore codeword ki energy | — |

**Kyun zaroori?**

1. SNR ka matlab tabhi banta hai jab `E[|z|²] = P`.
2. Warna encoder amplitude barha kar "free SNR" le leta.
3. Teeno links ka energy budget barabar rehta hai.

> **Detail:** yeh **per codeword** normalization hai (`dim=-1`), batch-level nahi.

---

## 3.10 AWGN channel

```
y = z + n
n ~ CN(0, σ²·I)
σ² = P · 10^(−SNR_dB / 10)
```

| Symbol | Meaning | Unit |
|---|---|---|
| `y` | received complex symbols | — |
| `z` | transmitted symbols | — |
| `n` | noise (complex Gaussian) | — |
| `CN(0, σ²I)` | circular complex Gaussian, mean 0, variance `σ²` per symbol | — |
| `σ²` | noise power per complex symbol | — |
| `P` | transmit power | 1.0 |
| `SNR_dB` | signal-to-noise ratio | dB |

**3 dB trap (viva favourite):**

```
Re(n) aur Im(n) ki variance = σ² / 2
taake E[|n|²] = σ²
```

Agar koi `σ²` ko per-component variance samajh le, to **3 dB** ka error ho jata hai.

---

## 3.11 Rayleigh fading + ZF equalization

```
y = h · z + n,        h ~ CN(0, 1)          (ek gain poore codeword ke liye)

ŷ = y · conj(h) / max( |h|² , ε_h )
```

| Symbol | Meaning | Value |
|---|---|---|
| `h` | complex fading gain, **ek** per codeword, k symbols par constant | CN(0,1) |
| `conj(h)` | complex conjugate | — |
| `|h|²` | channel gain ki power | — |
| `ε_h` | ZF guard (bahut chhote `|h|` par divide se bachne ke liye) | **1e−6** |

**Effective SNR:**

```
instantaneous:  SNR_inst = |h|² · P / σ²
average:        SNR_avg  = P / σ²          (kyunki E[|h|²] = 1)
```

**Simple baat:** AWGN mein sirf **noise** hai. Rayleigh mein ek **random gain** bhi hai. Kabhi
`|h|` chhota hota hai (deep fade) → ZF noise ko barha deta hai → outage.

---

## 3.12 SNR definition

```
SNR_dB = 10 · log10( P / σ² )
```

| Symbol | Meaning |
|---|---|
| `P` | transmit power |
| `σ²` | noise power |
| `log10` | base-10 logarithm |
| `10 ·` | power ratio ko dB mein badalna |

**Yaad rakhein:** 10× power = 10 dB. 100× power = 20 dB.

---

# PART 4 — DIGITAL LINK KA MATH (packet arithmetic)

```
n_src   = 6 · B                    (source bits)
n_info  = 6 · B + 6                (+ 6 tail bits)
n_coded = 2 · n_info               (rate 1/2)
n_pay   = n_coded / 2              (QPSK: 2 bits per symbol)
        = 6B + 6

Constraint:  n_pay ≤ k
```

| Symbol | Meaning | Value |
|---|---|---|
| `B` | bits per state component | 1, 2, 3, 4, 8 |
| `6` | state components | fixed |
| `6` | convolutional code ke tail bits (`K−1 = 7−1`) | fixed |
| `2` | code rate ka denominator (`R_c = 1/2`) | fixed |
| `2` | QPSK = 2 bits/symbol | fixed |
| `n_pay` | channel uses jo payload ko chahiye | — |
| `k` | available channel uses | — |

**Chune gaye pairs (exact match):**

| `B` | `n_pay = 6B + 6` | `k` |
|---:|---:|---:|
| 1 | 12 | 12 |
| 2 | 18 | 18 |
| 3 | 24 | 24 |
| 4 | 30 | 30 |
| 8 | 54 | 54 |

**k = 1, 2, 3 par kya hota hai?**

| `k` | chahiye (B=1) | available | deficit |
|---:|---:|---:|---:|
| 1 | 12 | 1 | 11 |
| 2 | 12 | 2 | 10 |
| 3 | 12 | 3 | 9 |

**Simple baat:** chhoti `k` par digital link **ban hi nahi sakta** — yeh "kamzor" nahi, **structurally
infeasible** hai. Isi liye M1 (k = 1–3) mein digital arm nahi hai.

---

# PART 5 — TASK LOSS: CHAAR TERMS

```
L_task = λd · L_d + λu · L_u + λv · L_v + λT · L_T
```

```
L_d = (1/T) · Σ_{t=1..T}   || p_t − g ||²
L_u = (1/T) · Σ_{t=0..T−1} || u_t ||²
L_v = (1/T) · Σ_{t=1..T}   || v_t ||²
L_T = || p_T − g ||²
```

| Symbol | Meaning | Unit |
|---|---|---|
| `L_d` | average squared distance to goal | m² |
| `L_u` | average squared control effort | (m/s²)² |
| `L_v` | average squared speed | (m/s)² |
| `L_T` | **terminal** squared distance (last step) | m² |
| `λd, λu, λv, λT` | weights (importance) | — |
| `p_t` | position at step t | m |
| `g` | goal position | m |
| `u_t` | applied control | m/s² |
| `v_t` | velocity | m/s |
| `T` | horizon = 100 steps | — |
| `||·||²` | vector ki length ka square | — |

**Index range ka matlab:**

- `L_d`, `L_v`: `t = 1 … T` — **t = 0 shamil nahi** (wo initial condition hai, hamari galti nahi).
- `L_u`: `t = 0 … T−1` — kyunki `T` controls hain (`T+1` positions ke liye).
- `L_T`: sirf **aakhri** step.

**Weights:**

| Arm | `λd` | `λu` | `λv` | `λT` | Total |
|---|---:|---:|---:|---:|---|
| Frozen | 1.0 | **0.0** | **0.0** | 2.0 | `L_d + 2L_T` |
| Shaped (λ-heavy) | 1.0 | **0.1** | **0.1** | 2.0 | `L_d + 0.1L_u + 0.1L_v + 2L_T` |

**Har term ka simple matlab:**

| Term | Simple meaning |
|---|---|
| `L_d` | "poore raste mein goal ke qareeb raho" |
| `L_u` | "zyada mehnat / zyada acceleration na karo" |
| `L_v` | "speed control mein rakho" |
| `L_T` | "aakhir mein goal par pohancho" |

**Do zaroori baatein:**

1. `λu = λv = 0` ka matlab: `L_u` aur `L_v` **compute hote hain magar seekhne mein use nahi hote** —
   sirf diagnostics hain.
2. Loss **true trajectory** par hai, decoded state par nahi. Gradient encoder tak sirf
   `u = f(ŝ)` ke zariye jata hai.

**Units ka issue (viva mein poochha ja sakta hai):** total loss **different units** ka sum hai
(`m²`, `m²/s⁴`, `m²/s²`). Is liye `λ = 0.1` ka koi absolute meaning nahi — woh 8 m workspace ke
scale ke saath tuned hai. 100 m workspace par same `0.1` ka matlab badal jayega.

---

# PART 6 — METRICS KA MATH

## 6.1 Success

```
success = 1  agar  ∃ t ∈ {0, 1, …, T} :  d_t ≤ r_g
d_t = || p_t − g ||₂
```

| Symbol | Meaning | Value |
|---|---|---|
| `success` | binary: 1 = goal mil gaya, 0 = nahi | 0/1 |
| `∃` | "kam se kam ek aisa t maujood hai" | — |
| `d_t` | step t par goal se distance | m |
| `p_t` | **true** position | m |
| `g` | **true** goal | m |
| `r_g` | success radius | **0.5 m** |
| `T` | horizon | **100** |

**Simple baat:** UAV **kisi bhi** step par 0.5 m ke andar aa jaye → success. "Aakhir mein rukna"
zaroori nahi.

**Time-to-goal:**

```
TTG = pehla t jahan d_t ≤ r_g ; agar kabhi nahi → T+1
```

## 6.2 Reconstruction MSE

```
MSE = (1 / (6·T)) · Σ_{t=1..T} || ŝ_t − s_t ||²
```

| Symbol | Meaning |
|---|---|
| `ŝ_t` | decoded (estimated) state |
| `s_t` | true state |
| `6` | state components |
| `T` | steps |
| note | **normalized** states par, t = 0 excluded |

**Warning:** yeh MSE **normalized** hai — isko direct metres mein na samjhein.

## 6.3 Control effort

```
control_effort = Σ_{t=0..T−1} || u_t ||²
```

| Symbol | Meaning | Unit |
|---|---|---|
| `u_t` | control at step t | m/s² |
| sum | poore episode ka total | (m/s²)² |

## 6.4 Failure

```
failure = (success == 0)
```

Failure ke do reason:

1. **Timeout** — 100 steps mein goal nahi mila.
2. **Workspace exit** — UAV `[−8, 8]²` se bahar chala gaya (evaluation mein episode freeze).

> **Nuance:** agar UAV pehle success kar chuka hai aur **baad mein** bahar chala jaye, to
> `success = 1` hi rehta hai. Exit success ko **override nahi** karta.

---

# PART 7 — AERIAL (3GPP UMa) KA MATH

```
d_3D = sqrt( d_2D² + h² )

p_1 = 4300 · log10(h) − 3800
d_1 = max( 460 · log10(h) − 700 , 18 )

P_LoS = d_1/d_2D + exp(−d_2D/p_1) · (1 − d_1/d_2D)

PL_LoS  = 28.0 + 22·log10(d_3D) + 20·log10(f_c)
PL_NLoS = −17.5 + (46 − 7·log10(h))·log10(d_3D) + 20·log10(40π·f_c/3)

PL_mean = P_LoS · PL_LoS + (1 − P_LoS) · PL_NLoS
```

| Symbol | Meaning | Unit |
|---|---|---|
| `d_2D` | UAV ki ground distance (horizontal) | m |
| `h` | UAV ki height | m |
| `d_3D` | 3-D distance | m |
| `f_c` | carrier frequency | **GHz** |
| `P_LoS` | line-of-sight hone ki probability | 0 … 1 |
| `PL_LoS` | LoS path loss | dB |
| `PL_NLoS` | NLoS path loss | dB |
| `PL_mean` | average path loss (LoS/NLoS mixture) | dB |
| `p_1`, `d_1` | 3GPP ke intermediate parameters | — |

**Pinned geometry** (`d_2D = 500 m`, `h = 50 m`, `f_c = 2 GHz`):

| Quantity | Value |
|---|---|
| `P_LoS` | 0.8887 |
| `PL_LoS` | 93.45 dB |
| `PL_NLoS` | 113.09 dB |
| `PL_mean` | 95.63 dB |

**Effective SNR mapping:**

```
SNR_eff = SNR_tx − PL
```

| Symbol | Meaning |
|---|---|
| `SNR_tx` | transmit power (normalized scale) |
| `PL` | path loss (dB) |
| `SNR_eff` | receiver par effective SNR |

**Mixture curve:**

```
SR(SNR_tx) = P_LoS · SR(SNR_tx − PL_LoS)
           + (1 − P_LoS) · SR(SNR_tx − PL_NLoS)
```

| Symbol | Meaning |
|---|---|
| `SR(·)` | success rate us effective SNR par |
| `P_LoS` | LoS episode ka weight |
| `1 − P_LoS` | NLoS episode ka weight |

**Required power gap (matched reliability):**

```
ΔP = P_req(neural, SR = 0.90) − P_req(digital, SR = 0.90)
```

| Symbol | Meaning |
|---|---|
| `P_req(...)` | us method ke liye **kitni power** chahiye target success rate par |
| `SR = 0.90` | target reliability |
| `ΔP` | horizontal gap (dB) — **same reliability** par comparison |

> **Zaroori:** yeh sab **normalised unit-gain** dB hain, **EIRP nahi**. Antenna gain, noise figure,
> cable loss shamil nahi.

---

# PART 8 — RESULTS: ASAN STORY

## Finding 1 — Bandwidth crossover (reconstruction arm)

AWGN, 0–20 dB ka mean, 3 seeds × 5,000 episodes:

| k | Reconstruction | Frozen task | Digital | Kaun jeeta |
|---:|---:|---:|---:|---|
| 12 | **0.242** | 0.028 | 0.062 | Neural (4×) |
| 18 | **0.254** | 0.018 | 0.164 | Neural |
| 24 | 0.233 | 0.034 | **0.306** | Digital |
| 30 | 0.279 | 0.021 | **0.566** | Digital |
| 54 | 0.246 | 0.016 | **0.995** | Digital |

**Simple baat:** kam bandwidth par digital link ko sirf **1 bit per component** milta hai (B=1) —
bohot rough quantization. Zyada bandwidth par uske paas 8 bits hote hain, to woh jeet jata hai.

**Rayleigh par bhi same ordering:** k = 54 par digital **0.941**, reconstruction **0.258**.

## Finding 2 — Frozen loss fail hota hai, λ fix karta hai

| Arm | k = 12 AWGN | k = 12 Rayleigh |
|---|---:|---:|
| Frozen (λu = λv = 0) | 0.020–0.048 (mean 0.033) | 0.033 |
| λ-light (0.01, 0.01) | 0.995 | **0.019** (fragile) |
| **λ-heavy (0.1, 0.1)** | **0.997** | **0.997** |

Kyun fail hota hai — **gradient**:

| Arm | p50 ‖g‖ | p99 ‖g‖ | non-finite steps | Converged |
|---|---:|---:|---:|---|
| Frozen (AWGN) | **3.0 × 10¹⁰** | 2.4 × 10¹⁸ | 75 | ✗ |
| Frozen (Rayleigh) | **8.1 × 10⁷** | 5.0 × 10¹⁸ | 100 | ✗ |
| λ-heavy | **3.3–4.1** | 57–113 | 0 | ✓ |

> **Note:** frozen arm ke yeh numbers **k = 18, width-256** cells se hain (A-9). 128-wide frozen
> history maujood nahi — yeh data limitation hai, chhupayi nahi gayi. λ-heavy numbers 128-wide hain.

**Simple baat:** frozen loss mein gradient **10–18 orders** tak phat jata hai; λ lagane se woh
**O(1)** ho jata hai. Yeh **capacity ka masla nahi** — 256-wide network bhi fail hua.

**A-16a (kaunsa term kaam karta hai):**

| Arm | k=12 AWGN | k=12 Ray | k=18 AWGN | k=18 Ray |
|---|---:|---:|---:|---:|
| `no_u` (control penalty hataya) | **0.016** | **0.016** | **failed** | **0.027** |
| `no_v` | 0.997 | 0.996 | 0.996 | 0.994 |
| `no_T` | 0.996 | 0.993 | 0.999 | 0.992 |

**Simple baat:** sirf **`λu`** active term hai. `λv` aur `λT` is operating point par cosmetic hain.

## Finding 3 — Aerial link

| Quantity | Value |
|---|---|
| λ-heavy, k = 54, SR ≥ 0.90 | **87 dB** |
| Digital, k = 54, SR ≥ 0.90 | **108 dB** |
| Gap | **21 dB** |
| λ-heavy, k = 18, SR ≥ 0.90 | 91 dB (digital yahan 0.90 tak pohanchta hi nahi) |
| 12 geometries par gap | **15–21 dB** @ 0.90; **12–15 dB** @ 0.99 |

**Simple baat:** same reliability par neural link ko **15–21 dB kam power** chahiye.

**Lekin:** yeh **ek hi performance curve** ko 12 geometries par remap kiya gaya hai — **independent
validation nahi**, sirf **link-budget sensitivity** hai.

## Finding 4 — Robustness

| Test | Result |
|---|---|
| 5G NR polar comparator | k = 12–30 par koi ordering change nahi; k = 54 par 0.909 vs 0.995 |
| 5G NR LDPC BG2 | k = 12–30 par significant difference nahi; k = 54 AWGN par 0.017 neeche |
| Doppler / time-correlated fading | λ arm ≤ 0.003 shifts; digital k = 54 par −0.028 |
| CSI error (σ up to 0.4) | success flat, **MSE +18%** |
| Uncoded QPSK (payload-maximal) | k = 12 par 0.526, k = 18 par 0.880, k = 54 par 0.884 |

> **Uncoded ka matlab:** digital ka low-bandwidth ceiling **partly framing/payload** ka masla hai,
> sirf "separation" ka nahi. Yeh claim ko **chhota** karta hai — aur hum khud batate hain.

---

# PART 9 — CLAIM RULES (10 rules, yaad kar lein)

| # | ✅ Keh sakte hain | ❌ Nahi kehna |
|---|---|---|
| 1 | "Reconstruction arm k ≤ 18 par jeetta hai, coded link k ≥ 24 par" | "Neural hamesha behtar hai" |
| 2 | "Yeh **specified fixed-rate baseline** (K=7 conv, QPSK, no AMC/HARQ) ke against hai" | "Tamam modern digital systems se behtar" |
| 3 | "15–21 dB **matched reliability** par, interpolated" | "Har operating point par 21 dB" |
| 4 | "A-17 ek shared curve ka **link-budget sensitivity** hai" | "12 geometries par independently validated" |
| 5 | "Frozen loss **optimisation** ki wajah se fail hota hai" | "Frozen loss **capacity** ki wajah se fail hota hai" |
| 6 | "Shaped arm **task ceiling** par hai (0.996–0.999)" | "Shaped arm digital se **significantly behtar** hai" |
| 7 | "Confidence intervals **overlap**; formal equivalence test nahi kiya" | "Dono **statistically equal** hain" |
| 8 | "**To our knowledge**, no prior work combines …" | "Yeh **first** hai" |
| 9 | "λ-shaping **ek diagnostic arm** hai, frozen protocol nahi" | "λ = 0.1 **optimal** hai" |
| 10 | "Yeh **conditional design rule** hai" | "Yeh **new paradigm** hai" |

**Agar phans jayein:** "Yeh number `<artifact>` se hai; main file dikha sakta hoon."

---

# PART 10 — 30 VIVA Q&A (short answers)

## A. Fundamentals (1–8)

**1. DeepJSCC kya hai?**
Ek encoder-decoder network jo source ko seedha channel symbols mein bhejta hai — beech mein bits nahi.

**2. Conventional digital se farq?**
Digital: compress → bits → channel code → modulate (do alag stages). DeepJSCC: ek hi network, channel training ke andar.

**3. Kya DeepJSCC Shannon capacity se aage jata hai?**
Nahi. Capacity same hai. Woh finite blocklength, low latency, aur task objective par **capacity ke qareeb pohanchne ka raasta** badalta hai.

**4. Kam bandwidth par neural kyun jeetta hai?**
Digital ko B=1 bit/component milta hai (bohot rough quantization) + 6 tail bits. Neural ke paas continuous symbols hain.

**5. Zyada bandwidth par digital kyun jeetta hai?**
Uske paas 8 bits/component hote hain → almost lossless. Neural reconstruction objective bottleneck ban jata hai.

**6. Task-oriented communication kya hai?**
Transmission ko downstream task (yahan: navigation success) par optimize karna, reconstruction par nahi.

**7. Goal ko kyun transmit karte hain?**
Controller ko `[p, v, g]` chahiye. Receiver self-contained rehta hai, koi side-channel nahi. **Cost:** 6 mein se 2 components constant hain → payload redundant.

**8. 6-D state kyun, images kyun nahi?**
UAV navigation ke liye compact kinematic state kaafi hai; yeh tractable hai aur literature mein is combination ka gap tha.

## B. System / model (9–16)

**9. Double integrator kya hai?**
`u → v → p`. Control acceleration hai; inertia hai; control band karein to UAV rukta nahi.

**10. `½Δt²` kyun?**
Kyunki `u` poore step constant hai — yeh exact integration hai, Euler nahi.

**11. Controller fixed kyun?**
Taake performance ka farq **communication** ki wajah se ho, control policy ki wajah se nahi.

**12. ζ = 1 ka matlab?**
Critically damped — overshoot nahi. **Caveat:** sirf small-signal; clip ke baad nonlinear.

**13. Normalization kyun?**
Sab components ko `[−1, 1]` mein laana, taake network scale se confuse na ho. Ranges config se, data se nahi.

**14. Power normalization kyun?**
SNR ka matlab tabhi banta hai jab `E[|z|²] = P`. Warna encoder amplitude barha kar cheating karta.

**15. AWGN vs Rayleigh?**
AWGN: sirf `+n`. Rayleigh: `h·z + n`, ek random gain per codeword → deep fade → outage.

**16. Perfect CSI + ZF ki limitation?**
Exact CSI assume karta hai; deep fade par ZF noise barha deta hai; no diversity; no CSIT.

## C. Loss / training (17–24)

**17. Chaar loss terms?**
`L_d` (distance), `L_u` (control), `L_v` (velocity), `L_T` (terminal). Weights `(1, 0, 0, 2)` frozen.

**18. Frozen loss kyun fail hota hai?**
Sirf distance/terminal signal hai; koi action penalty nahi. Gradient 10¹⁰–10¹⁸ tak phat jata hai; 75–100 non-finite steps.

**19. λ_u mathematically kya karta hai?**
Action par quadratic penalty: `∂L/∂u = 2λ_u·u + …`. Yeh gradient ko bound karta hai (trust-region jaisa).

**20. Kaunsa term active hai?**
`λ_u`. A-16a: sirf `λ_u` hatane se success 0.016–0.027 ho jati hai; `λ_v`/`λ_T` hatane se kuch nahi hota.

**21. Sirf zyada train karte to?**
Nahi. 12,000 steps → 0.035. Fixed 10 dB training → 0.050. Reconstruction auxiliary → 0.243.

**22. λ-light kyun fail hua?**
0.01 bohot kamzor hai; Rayleigh par 0.019 — degenerate. Fix "any positive λ" nahi hai.

**23. Kya yeh capacity ka masla hai?**
Nahi. 256-wide frozen network bhi 0.027/0.041 par fail hua.

**24. Kya λ general cure hai?**
Nahi. A-12 mein wahi λ arm 3GPP dispersion par diverge hui. Trigger = rare high-error episodes.

## D. Results / statistics (25–30)

**25. Crossover exactly kahan hai?**
k = 18 aur 24 ke beech, **reconstruction arm** ke liye.

**26. Shaped arm k = 54 par haarta hai?**
Nahi — ceiling par hai (0.998 vs 0.995), intervals overlap. Lekin **equivalence claim nahi**.

**27. 15–21 dB ka matlab?**
Matched reliability (0.90) par **interpolated required power** ka horizontal gap.

**28. A-17 independent validation hai?**
Nahi. Ek shared curve 12 geometries par remap — **sensitivity analysis**.

**29. McNemar test kyun?**
Success binary hai aur episodes paired hain. McNemar paired binary comparison ke liye hai.

**30. Multiple comparisons ka correction?**
Nahi kiya. Effects bohot bade hain (0.02 vs 0.99), is liye qualitative conclusion nahi badalta.

---

# PART 11 — 6 HARD QUESTIONS (model answers)

### H1. "Shaped arm 0.998, digital 0.995 — to aap ka contribution kya? Sirf tie?"

> "Contribution win nahi, **boundary** hai. Hum dikhate hain ke kab neural jeetta hai aur kab nahi.
> Shaped arm ceiling par hai, is liye benchmark usay rank nahi kar sakta — yeh hum **khud** likhte
> hain. Asli contributions: matched-budget protocol, crossover ki location, frozen-loss ka
> diagnosis, aur aerial link-budget mapping. Tie ko win banana hamara claim nahi."

### H2. "λ_u = 0.1 kaise chuna? Yeh tuning artefact hai?"

> "Wave-A mein hum ne 0.01 aur 0.1 dono chalaye. 0.01 AWGN par 0.995 deta hai magar Rayleigh par
> 0.019 — fragile. 0.1 dono par 0.997. A-16a term-removal confirm karta hai ke `λ_u` hi active
> term hai. Phir bhi yeh **ek hyperparameter** hai, hum ise optimal nahi kehte — aur 0.1 ka
> meaning workspace scale se juda hua hai."

### H3. "Aap ka digital baseline kamzor hai. Adaptive MCS/HARQ hota to?"

> "Bilkul, aur hum yeh paper mein likhte hain. Baseline **specified fixed-rate** hai — no AMC,
> no HARQ. Uncoded payload-maximal link bhi dikhaya: k = 12 par 0.062 se 0.526. Yani gap ka kuch
> hissa framing/rate adaptation ka hai. Hamara claim us baseline ke against hai, 'all digital'
> ke against nahi. Adaptive baseline future work hai."

### H4. "0.242 vs 0.062 — sirf 4×? Yeh to bara gap nahi."

> "k = 12 par digital ke paas B = 1 hai — 1 bit per component, yani 2 levels. Wahan 4× gap
> quantization ki wajah se hai. Ahem baat gap ka size nahi, **crossover ki location** hai: k = 18
> ke baad ordering palat jati hai. Aur k = 1–3 par digital link **structurally infeasible** hai."

### H5. "Gradient 10¹⁰ — yeh aap ke code ka bug ho sakta hai?"

> "Teen wajah se nahi lagta. Ek, 256-wide network bhi fail hua — parameter count masla nahi. Do,
> 75–100 non-finite steps record hue. Teen, longer training aur fixed-SNR training bhi fail hue,
> magar λ lagane se same architecture 0.997 par chala gaya. Hum curvature ya causal chain claim
> nahi karte — sirf gradient magnitude aur convergence ka evidence dete hain."

### H6. "Aap ne 3GPP channel simulate nahi kiya, sirf path loss lagaya?"

> "Sahi. Hum 3GPP ka **large-scale layer** use karte hain: LoS probability, LoS/NLoS path loss,
> aur unka mixture. Doppler, time-correlated shadowing, aur aerial small-scale fading shamil nahi.
> A-17 ek shared curve ka remap hai — sensitivity, validation nahi. Yeh limitations hum explicitly
> likhte hain."

---

# PART 12 — LIMITATIONS (khud pehle bol dein)

| # | Limitation | Ek line mein |
|---|---|---|
| 1 | Simulation-only | Koi flight test ya SDR nahi |
| 2 | 2-D kinematics | 3-D / 6-DoF flight dynamics nahi |
| 3 | Fixed PD controller | Joint control-communication co-design nahi |
| 4 | 3GPP = large-scale only | Doppler, shadowing correlation, aerial small-scale fading nahi |
| 5 | A-17 = shared-curve remap | Independent validation nahi |
| 6 | Digital baseline fixed-rate | No AMC / HARQ / soft combining |
| 7 | Task saturated | 0.996+ par benchmark rank nahi kar sakta |
| 8 | Extension blocks single-seed | Statistical strength kam |
| 9 | A-13 ek hi threshold | +20 dB, ek seed |
| 10 | Units mixed in loss | λ workspace-scale dependent |

**Kehne ka tareeqa:** "Yeh hamari study ki hadd hai, aur hum ise paper mein disclose karte hain."
Iske baad **kabhi** defend na karein — accept karein aur next step batayein.

---

# PART 13 — DEMO / LIVE PROOF (agar poochha jaye)

**Training live na chalayein** — 3,000 steps CPU par ghanton ka kaam hai.

| Kya dikhayein | Command / file | Time |
|---|---|---|
| Tests pass | `.\.venv\Scripts\python.exe -m pytest tests -q` | ~15 s |
| Completeness audit | `python artifacts/_audit_completeness.py` | ~5 s |
| Main numbers | `results/M2/step11_m2_summary.json` | instant |
| Gradient evidence | `results/MECHANISM/step_stability_summary.json` | instant |
| Aerial table | `results/AERIAL_GEOM/` | instant |
| Figures | `SUBMISSION_KIT/figures/*.png` | instant |

**Code walk-through ke liye 5 files:**

1. `project/uav/dynamics.py` — plant
2. `project/uav/controller.py` — PD
3. `project/losses/task_loss.py` — 4 terms
4. `project/models/reconstruction_jscc.py` — full link
5. `project/training/train_task_oriented.py` — closed loop

**Agar laptop fail ho jaye:** printed ONE_PAGER + 5 figures ka backup PDFrakhein.

---

# PART 14 — NUMBER CHEAT SHEET (sirf yeh 14 yaad karein)

| # | Number | Kya hai |
|---:|---|---|
| 1 | **6** | state components |
| 2 | **12–54** | k channel uses |
| 3 | **0.242 vs 0.062** | k = 12, recon vs digital |
| 4 | **0.254 vs 0.164** | k = 18 |
| 5 | **0.233 vs 0.306** | k = 24 (crossover ke baad) |
| 6 | **0.246 vs 0.995** | k = 54 |
| 7 | **2–6%** | frozen task loss |
| 8 | **0.997** | λ-heavy, k = 12 |
| 9 | **0.996–0.999** | shaped arm, baqi k |
| 10 | **7–10 orders** | median gradient gap |
| 11 | **87 vs 108 dB** | aerial, k = 54, SR 0.90 |
| 12 | **15–21 dB** | 12-geometry gap @ 0.90 |
| 13 | **0.5 m** | success radius |
| 14 | **111** | tests |

**Agar bhool jayein:** "Number artifact se confirm kar sakta hoon" — aur file khol dein.

---

# PART 15 — 8-SLIDE PLAN (10 minute)

| # | Slide | Ek line | Time |
|---|---|---|---|
| 1 | Title + problem | "UAV state bhejna hai; reconstruction ≠ navigation" | 1:00 |
| 2 | System + 3 arms | "Same episodes, same noise, same k — sirf objective badla" | 1:30 |
| 3 | **Crossover** | "k ≤ 18 neural, k ≥ 24 digital" | 2:00 |
| 4 | Task-arm fix | "Frozen 2–6%; λ-shaping se 0.997" | 1:30 |
| 5 | Mechanism | "Yeh capacity nahi — gradient conditioning hai" | 1:30 |
| 6 | Aerial | "87 vs 108 dB @ 0.90, k = 54" | 1:30 |
| 7 | Robustness | "4 comparators + Doppler + CSI — ordering nahi badalti" | 1:00 |
| 8 | Limitations + takeaway | "Conditional design rule, not a new paradigm" | 0:30 |

**2-minute version:** slide 3 → 4 → 6. Bas.

**Transition line har slide ke liye:** "Yeh result agla sawal kharra karta hai…" — is se flow aasan hota hai.

---

# PART 16 — CHECKLIST

## Print / submission se pehle fix karein

| # | Kya | Kahan | Status |
|---|---|---|---|
| 1 | Success definition "final distance ≤ 0.5 m" → **"at some step t ≤ T"** | paper §III.E (line 85) **aur** thesis §4.10 (line 547) | ⚠️ **fix karein** |
| 2 | Untrained baseline: paper kehta hai "not reported", thesis Table 5.5 mein **0.016** hai | paper §IV-B vs thesis | ⚠️ ek choose karein |
| 3 | Appendix A ke counts + search date | paper Appendix A | ⚠️ pending |
| 4 | Abstract 311 words → ≤250 | paper | ⚠️ pending |
| 5 | Thesis aerial ranges (`74–108`, `78–113`, `89–127`) re-derive | thesis | ⚠️ pending; **abhi quote na karein** |

## Day-before

- [ ] 8 slides PDF + PPTX dono
- [ ] ONE_PAGER 4 copies print
- [ ] Laptop + charger + HDMI adapter
- [ ] Backup USB mein figures + DOCX + PDF
- [ ] `pytest` ek dafa chala kar dekh lein
- [ ] 2-minute version bol kar timer lagayein
- [ ] Part 14 ke 14 numbers likh kar yaad karein

## Common mistakes (inko na karein)

| ❌ Ghalat | ✅ Sahi |
|---|---|
| "Neural hamesha behtar hai" | "k ≤ 18 par, reconstruction arm ke liye" |
| "First work" | "To our knowledge, no prior work combines…" |
| "Statistically equal" | "Confidence intervals overlap; no formal test" |
| "14 orders" | "7–10 orders (median)" |
| "Transmit SNR / EIRP" | "Normalised transmit power, not EIRP" |
| "Digital link k = 1 par kamzor hai" | "k = 1–3 par packet infeasible hai" |
| "λ = 0.1 optimal hai" | "λ = 0.1 diagnostic arm hai" |
| "3GPP channel simulate kiya" | "3GPP large-scale layer use kiya" |

---

# PART 17 — GLOSSARY (aasan matlab)

| Term | Aasan matlab |
|---|---|
| **DeepJSCC** | Neural network jo source ko seedha channel symbols mein bhejta hai |
| **Task-oriented** | Transmission ko final task (navigation) par optimize karna |
| **Reconstruction** | Sirf state ko sahi recover karna |
| **Channel use** | Ek complex symbol bhejne ka mauqa (k = kitne mauqe) |
| **AWGN** | Sirf add hone wala noise |
| **Rayleigh fading** | Signal ka random multiplicative gain |
| **ZF equalization** | Receiver `h` se divide kar ke fading hataata hai |
| **CSI** | Channel ki information |
| **SNR** | Signal power ÷ noise power |
| **Crossover** | Woh k jahan jeetne wala badal jata hai |
| **Ceiling** | Task ka max possible success (~0.999) |
| **Saturation** | Sab methods ceiling ke paas → farq pata nahi chalta |
| **Ablation** | Ek cheez hata kar dekhna ke us ka kya role tha |
| **Gradient conditioning** | Training ke gradients ka behaviour (bounded vs exploding) |
| **Non-finite** | `inf` ya `nan` — training toot jati hai |
| **Matched budget** | Dono links ko same channel uses / bandwidth |
| **EIRP** | Effective radiated power (hamara number **yeh nahi** hai) |
| **Wilson interval** | Success rate ka confidence interval |
| **McNemar test** | Paired binary comparison ka statistical test |
| **LoS / NLoS** | Line-of-sight / non-line-of-sight |
| **UMa-aerial** | 3GPP ka UAV ke liye urban macro channel model |

---

## Final line (agar sirf ek sentence kehte hain)

> "We show **when** task-oriented DeepJSCC is worth it — low bandwidth, low SNR, fading — and we
> show **why** it fails without control shaping: not capacity, but optimisation. At high bandwidth
> the coded link wins, and we say so."

**Aakhri yaad-dehani:** Panel ko impress **numbers** se nahi, **honest boundary** se hota hai.
Jo cheez aap nahi jaante, woh "verified nahi" keh dein — yeh weakness nahi, discipline hai.
