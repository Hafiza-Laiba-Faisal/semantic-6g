# Step 3 — Final Mathematical & Implementation Specification

## 1. Overall system

```text
UAV state s_t
      ¦
      ?
State normalization
      ¦
      ?
Task-Oriented DeepJSCC Encoder
      ¦
      ?
Power normalization
      ¦
      ?
Complex channel symbols z_t ? C^k
      ¦
      +-- AWGN
      ¦
      +-- Rayleigh fading + CSI equalization
      ¦
      ?
Received symbols
      ¦
      ?
DeepJSCC Decoder
      ¦
      ?
Navigation-relevant decoded state
      ¦
      ?
Controller p(·)
      ¦
      ?
Control action u_t
      ¦
      ?
2-D UAV dynamics
      ¦
      ?
next state / goal-reaching performance
```

The project is simulation-only and implemented in Python/PyTorch. 

---

# 2. UAV state

Use the project state:

$$
\boxed{
s_t=
[x_t,y_t,v_{x,t},v_{y,t},x_g,y_g]^T
\in\mathbb R^6
}
$$

where:

* \(x_t,y_t\): UAV position
* \(v_{x,t},v_{y,t}\): UAV velocity
* \(x_g,y_g\): goal position

The 2-D position+velocity portion is supported by the UAV communication/control literature; adding the goal coordinates is a **project design choice**, not something to claim as copied directly from one paper. 

---

# 3. UAV dynamics

Use the 2-D double-integrator model:

$$
\mathbf p_t=
\begin{bmatrix}
x_t\\y_t
\end{bmatrix},
\qquad
\mathbf v_t=
\begin{bmatrix}
v_{x,t}\\v_{y,t}
\end{bmatrix}
$$

and control:

$$
\mathbf u_t=
\begin{bmatrix}
a_{x,t}\\a_{y,t}
\end{bmatrix}.
$$

Continuous dynamics:

$$
\dot{\mathbf p}_t=\mathbf v_t
$$

$$
\dot{\mathbf v}_t=\mathbf u_t.
$$

Discrete dynamics:

$$
\boxed{
\mathbf p_{t+1}
=
\mathbf p_t+\Delta t\,\mathbf v_t
+\frac12\Delta t^2\mathbf u_t
}
$$

$$
\boxed{
\mathbf v_{t+1}
=
\mathbf v_t+\Delta t\,\mathbf u_t
}
$$

This is consistent with the double-integrator formulation extracted from the UAV modelling sources. 

For implementation, **\(\Delta t\), velocity limits and acceleration limits must be explicit project parameters**, not silently borrowed from the paper.

---

# 4. Goal and navigation quantities

Define goal:

$$
\mathbf p_g=[x_g,y_g]^T.
$$

Position error:

$$
\mathbf e_t=\mathbf p_t-\mathbf p_g.
$$

Distance to goal:

$$
\boxed{
d_t=\|\mathbf p_t-\mathbf p_g\|_2
}
$$

Goal success:

$$
d_t\le r_g
$$

where \(r_g\) is the declared goal radius.

---

# 5. Controller

Use a deterministic proportional/PD-style controller as the navigation interface.

A suitable project formulation is:

$$
\boxed{
\mathbf u_t^{*}
=
\operatorname{clip}
\left(
-K_p(\mathbf p_t-\mathbf p_g)
-K_v\mathbf v_t,
-u_{\max},u_{\max}
\right)
}
$$

This is the **reference/controller definition for the simulation**, not a claim that the exact \(K_p,K_v\) values come from the literature.

The important point for the experiment is that **all three communication systems feed the same controller**.

```text
Digital ? decoded state ? same controller
DeepJSCC reconstruction ? decoded state ? same controller
Task-Oriented DeepJSCC ? decoded/task state ? same controller
```

This prevents the controller itself from becoming an uncontrolled experimental variable.

---

# 6. State normalization

Because the original DeepJSCC paper normalizes image values to a bounded interval, but does **not** prescribe normalization for UAV states, the UAV state needs an explicit project-level normalization rule. 

Use:

$$
\boxed{
\bar{s}_{t,i}
=
2\frac{\operatorname{clip}(s_{t,i},a_i,b_i)-a_i}
{b_i-a_i}-1
}
$$

so every input component lies approximately in:

$$
[-1,1].
$$

The ranges \(a_i,b_i\) must be declared in the configuration file.

**Do not allow the code to calculate min/max separately from the test set.** That would leak test information.

---

# 7. DeepJSCC encoder

The original DeepJSCC mapping is:

$$
f_\theta:\mathbb R^n\rightarrow\mathbb C^k.
$$

For our low-dimensional state:

$$
\boxed{
f_\theta:\mathbb R^6\rightarrow\mathbb C^k
}
$$

The original image CNN architecture should **not** simply be copied because our source is a 6-D vector rather than an image. The source itself explicitly notes that the paper does not specify an MLP architecture for low-dimensional vector states. 

Therefore implement a compact MLP encoder:

$$
6\rightarrow h_1\rightarrow h_2\rightarrow 2k.
$$

For example:

```text
Linear(6, H)
PReLU
Linear(H, H)
PReLU
Linear(H, 2k)
```

The exact hidden dimension is a **project architecture parameter**, not a literature claim.

---

# 8. Real-to-complex conversion

The final network output contains \(2k\) real numbers.

Split:

$$
\tilde{\mathbf z}_{R}
\in\mathbb R^k,
\qquad
\tilde{\mathbf z}_{I}
\in\mathbb R^k
$$

and form:

$$
\boxed{
\tilde z_i=
\tilde z_{R,i}
+j\tilde z_{I,i}
}
$$

giving:

$$
\tilde{\mathbf z}\in\mathbb C^k.
$$

The source confirms the \(2k\)-real-output ? \(k\)-complex-symbol structure. The precise ordering of real/imaginary components is an implementation convention and should remain fixed throughout the project. 

---

# 9. Power normalization

This is **not optional**.

Use the DeepJSCC normalization:

$$
\boxed{
\mathbf z=
\sqrt{kP}
\frac{\tilde{\mathbf z}}
{\sqrt{\tilde{\mathbf z}^{*}\tilde{\mathbf z}+\epsilon}}
}
$$

The paper gives:

$$
\frac1k
E[\mathbf z^*\mathbf z]\le P
$$

and the corresponding normalization. 

Set:

$$
P=1
$$

to match the reference convention, unless the experiment explicitly defines another normalized power.

The small \(\epsilon\) is only a numerical implementation safeguard.

---

# 10. Bandwidth ratio

Define:

$$
\boxed{
\rho=\frac{k}{n}
}
$$

where:

$$
n=6
$$

for the six-dimensional source.

Thus:

$$
\boxed{
\rho=\frac{k}{6}
}
$$

if \(k\) denotes complex channel uses per state.

**Important:** don't accidentally count \(2k\) real neural outputs as \(2k\) channel uses. The system transmits \(k\) complex symbols.

The original DeepJSCC work defines \(k/n\) as the bandwidth-ratio convention. 

---

# 11. AWGN channel

Transmit:

$$
\mathbf z\in\mathbb C^k.
$$

AWGN:

$$
\boxed{
\mathbf y=\mathbf z+\mathbf n
}
$$

with:

$$
\boxed{
\mathbf n\sim
\mathcal{CN}(0,\sigma^2I_k)
}
$$

and:

$$
\boxed{
SNR_{dB}
=
10\log_{10}\left(\frac{P}{\sigma^2}\right)
}
$$

The reference DeepJSCC experiments use \(P=1\) and vary \(\sigma^2\) to obtain the desired SNR. 

Therefore:

$$
\boxed{
\sigma^2=
\frac{P}{10^{SNR_{dB}/10}}
}
$$

For complex Gaussian implementation:

$$
n_R,n_I
\sim
\mathcal N(0,\sigma^2/2).
$$

---

# 12. Rayleigh fading

Use:

$$
\boxed{
\mathbf y=h\mathbf z+\mathbf n
}
$$

where:

$$
h\sim\mathcal{CN}(0,1).
$$

The reference model is **slow Rayleigh fading**, meaning the same channel coefficient applies over the transmitted codeword rather than independently changing for every symbol. 

This distinction matters.

Do **not** implement:

```text
different h for every symbol
```

if we are reproducing the slow/block-fading setup.

---

# 13. CSI and equalization

For the main Rayleigh experiment, use perfect receiver CSI:

$$
\boxed{
\hat h=h
}
$$

and equalize:

$$
\boxed{
\tilde{\mathbf y}
=
\frac{\mathbf y}{h}
}
$$

before decoding.

Then:

$$
\tilde{\mathbf y}
=
\mathbf z+\frac{\mathbf n}{h}.
$$

CSI must be identical in the corresponding baseline systems.

**Do not give one communication method CSI while denying it to another.**

---

# 14. Task-oriented objective

This is the most important part.

The Shao et al. source establishes the task-oriented principle:

$$
Y\rightarrow X\rightarrow Z\rightarrow\hat Z\rightarrow\hat Y
$$

and an objective that emphasizes task information in the received representation rather than reconstruction of \(X\). 

But their \(Y\) is a classification target.

Therefore **do not claim that their VIB classification equation is our UAV navigation loss.**

Our task is closed-loop navigation.

The project objective should instead be based on navigation performance.

A practical differentiable training objective is:

$$
\boxed{
L_{\text{task}}
=
\lambda_d L_d+
\lambda_u L_u+
\lambda_v L_v+
\lambda_T L_T
}
$$

with components such as:

### Distance loss

$$
L_d
=
\frac1T
\sum_{t=1}^{T}
\|\mathbf p_t-\mathbf p_g\|_2^2
$$

### Control-effort loss

$$
L_u
=
\frac1T
\sum_{t=0}^{T-1}
\|\mathbf u_t\|_2^2
$$

### Velocity regularization

$$
L_v
=
\frac1T
\sum_{t=1}^{T}
\|\mathbf v_t\|_2^2
$$

### Terminal distance

$$
L_T=
\|\mathbf p_T-\mathbf p_g\|_2^2.
$$

Thus:

$$
\boxed{
L_{\text{TO}}
=
\lambda_dL_d+
\lambda_uL_u+
\lambda_vL_v+
\lambda_TL_T
}
$$

This is a **project-designed differentiable navigation objective inspired by the task-oriented/closed-loop-control literature**, not an equation we should falsely attribute verbatim to Shao or the DeepJSCC paper.

---

# 15. Task-Oriented DeepJSCC training

Training pipeline:

$$
s_t
\rightarrow
\text{encoder}
\rightarrow
z_t
\rightarrow
\text{channel}
\rightarrow
\hat z_t
\rightarrow
\text{decoder}
\rightarrow
\hat s_t
\rightarrow
\pi
\rightarrow
u_t
\rightarrow
\text{dynamics}
\rightarrow
L_{\text{TO}}.
$$

Backpropagation therefore goes:

```text
navigation loss
      ?
UAV dynamics
      ?
controller
      ?
decoder
      ?
channel
      ?
encoder
```

The channel must be implemented with differentiable operations during training.

---

# 16. Reconstruction DeepJSCC baseline

This is the second neural baseline.

Same:

* state
* normalization
* encoder capacity
* decoder capacity
* bandwidth
* channel
* SNR
* training episodes
* optimizer
* seeds

but the training objective becomes:

$$
\boxed{
L_{\text{recon}}
=
\frac16
\|\hat s_t-s_t\|_2^2
}
$$

or, preferably, normalized-state MSE:

$$
\boxed{
L_{\text{recon}}
=
\frac16
\|\hat{\bar{s}}_t-\bar{s}_t\|_2^2.
}
$$

Then evaluate its navigation performance **through the same controller and UAV simulator**.

This directly tests whether low reconstruction error necessarily means good navigation.

---

# 17. Conventional digital baseline

The literature supports the generic separated architecture:

$$
\boxed{
s
\rightarrow
\text{source coding}
\rightarrow
\text{channel coding}
\rightarrow
\text{modulation}
\rightarrow
\text{channel}
\rightarrow
\text{demodulation}
\rightarrow
\text{channel decoding}
\rightarrow
\text{source decoding}
}
$$

The original DeepJSCC paper uses JPEG/JPEG2000 + LDPC + BPSK/QAM for images, but **does not specify the appropriate quantizer for a six-dimensional continuous UAV state**. 

Therefore our vector digital baseline must explicitly define:

```text
state normalization
        ?
uniform scalar quantization
        ?
bits
        ?
LDPC
        ?
QPSK / selected modulation
        ?
channel
        ?
demodulation
        ?
LDPC decoding
        ?
dequantization
        ?
decoded state
        ?
same controller
```

The quantization rule and exact code/modulation configuration are **project design choices**, not claims from Bourtsoulatze et al.

---

# 18. Fair comparison rule

This is critical.

All three systems must have:

$$
\boxed{
\text{same source}
}
$$

$$
\boxed{
\text{same channel}
}
$$

$$
\boxed{
\text{same SNR}
}
$$

$$
\boxed{
\text{same bandwidth/channel-use budget}
}
$$

$$
\boxed{
\text{same UAV dynamics}
}
$$

$$
\boxed{
\text{same controller}
}
$$

$$
\boxed{
\text{same test episodes}
}
$$

$$
\boxed{
\text{same random seeds where applicable}
}
$$

Otherwise the performance comparison is confounded.

---

# 19. Evaluation metrics

Primary metric:

$$
\boxed{\text{Goal-reaching success rate}}
$$

$$
SR=
\frac{\text{successful episodes}}
{\text{total episodes}}
$$

where success means:

$$
d_t\le r_g
$$

within the maximum episode horizon.

Additional metrics:

### Final distance

$$
d_T=\|\mathbf p_T-\mathbf p_g\|_2
$$

### Average trajectory distance

$$
\bar d=
\frac1T\sum_{t=1}^{T}d_t
$$

### Control effort

$$
E_u=
\sum_{t=0}^{T-1}\|\mathbf u_t\|_2^2
$$

### Time-to-goal

Number of simulation steps required to satisfy:

$$
d_t\le r_g.
$$

### Reconstruction MSE

For the two systems where a decoded state exists:

$$
MSE=
\frac1{6T}
\sum_t
\|\hat s_t-s_t\|_2^2.
$$

This allows us to explicitly compare **communication fidelity vs actual task performance**.

---

# 20. Main experiments

Minimum experiment matrix:

| System                  | AWGN | Rayleigh |
| ----------------------- | ---: | -------: |
| Digital                 |    ? |        ? |
| Reconstruction DeepJSCC |    ? |        ? |
| Task-Oriented DeepJSCC  |    ? |        ? |

Across multiple SNR values.

Then evaluate at multiple bandwidth ratios:

$$
\rho=\frac{k}{6}.
$$

---

# 21. Ablations

At minimum:

### A. Task loss vs reconstruction loss

```text
Task-Oriented DeepJSCC
        vs
Reconstruction DeepJSCC
```

### B. Channel

```text
AWGN
vs
Rayleigh
```

### C. Bandwidth

Different \(k\) values.

### D. SNR

Multiple SNR operating points.

### E. Task-loss components

Remove individual terms:

```text
distance only
distance + control
distance + terminal
full loss
```

This tells us whether the claimed task-oriented benefit comes from a particular component.

---

# 22. Statistical reporting

Do not report one lucky training run.

For each major configuration:

$$
\boxed{
N_{\text{seeds}}\ge3
}
$$

and report:

$$
\text{mean}\pm\text{standard deviation}.
$$

For success rate, additionally report the number of successful episodes and total episodes.

The **same evaluation episodes** should be used across methods at each experimental condition.

---

# 23. What is officially frozen vs what still needs configuration

### Frozen now

These should **not** be changed while coding:

* 6-D UAV state
* 2-D dynamics
* complex \(k\)-symbol DeepJSCC
* \(2k\) real neural outputs
* power normalization
* \(k/6\) bandwidth convention
* AWGN equation
* SNR convention
* slow Rayleigh model
* perfect CSI for main Rayleigh experiment
* same controller for all methods
* navigation-based task objective
* reconstruction DeepJSCC baseline
* separated digital baseline
* goal-reaching as primary task metric

### Must be explicit in `config.py`

These are **parameters**, not hidden assumptions:

```text
dt
T_max
goal_radius
Kp
Kv
u_max
v_max

state ranges
hidden dimensions
k values
SNR values

lambda_distance
lambda_control
lambda_velocity
lambda_terminal

optimizer
learning rate
batch size
epochs
number of seeds

digital quantization bits
digital LDPC configuration
digital modulation
```

---

# 24. The final code architecture

Tell the coding AI to build this structure:

```text
project/
¦
+-- config.py
¦
+-- data/
¦   +-- state_generator.py
¦   +-- normalization.py
¦
+-- uav/
¦   +-- dynamics.py
¦   +-- controller.py
¦   +-- environment.py
¦
+-- channels/
¦   +-- awgn.py
¦   +-- rayleigh.py
¦   +-- complex_utils.py
¦
+-- models/
¦   +-- deepjscc_encoder.py
¦   +-- deepjscc_decoder.py
¦   +-- task_oriented.py
¦
+-- baselines/
¦   +-- reconstruction_jscc.py
¦   +-- digital.py
¦
+-- losses/
¦   +-- task_loss.py
¦   +-- reconstruction_loss.py
¦
+-- training/
¦   +-- train_task.py
¦   +-- train_reconstruction.py
¦   +-- common.py
¦
+-- evaluation/
¦   +-- evaluate.py
¦   +-- metrics.py
¦   +-- statistics.py
¦
+-- experiments/
¦   +-- awgn.py
¦   +-- rayleigh.py
¦   +-- ablations.py
¦
+-- main.py
```

---
