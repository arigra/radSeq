\subsection{Temporal Radar Sequence Generation via Diffusion Transformer}
\label{ssec: temporal generation}

While single-frame \gls{rd} map generation captures spatial radar characteristics, many radar applications such as target tracking and behavior recognition require temporally coherent sequences that preserve physical motion dynamics and inter-frame dependencies. To address this, we extend the diffusion framework to generate realistic temporal radar sequences. 

Recent advances in diffusion models have demonstrated the effectiveness of transformer based architectures for capturing long-range dependencies and complex correlations \cite{peebles2023scalable}. However, unlike natural images, \gls{rd} maps do not exhibit spatial correlations across distant regions. A target in one part of the map is physically independent of targets elsewhere, making full self-attention across all spatial patches within a single frame unnecessary. Nevertheless, \gls{rd} cells are not statistically independent, neighboring range and Doppler bins are locally correlated. Therefore, our model focuses self-attention primarily on temporal dependencies across frames, while preserving local spatial coherence through patch level representations.


Let $\mathbf{X} = \{\mathbf{x}_0^{(1)}, \mathbf{x}_0^{(2)}, \ldots, \mathbf{x}_0^{(L)}\}$ denote a clean radar sequence, where each $\mathbf{x}_0^{(\ell)} \in \mathbb{R}^{N \times K}$ represents the \gls{rd} map at time step $\ell$ and $L$ is the sequence length. During diffusion, we obtain the noisy sequence $\mathbf{X}_t = \{\mathbf{x}_t^{(1)}, \mathbf{x}_t^{(2)}, \ldots, \mathbf{x}_t^{(L)}\}$ at diffusion timestep $t$. Each frame $\mathbf{x}_t^{(\ell)} \in \mathbb{R}^{N \times K}$ is locally encoded into spatial tokens using small overlapping patches with stride $s < p$, where $p$ is the patch size. This overlapping design ensures that targets moving across patch boundaries remain visible in multiple adjacent patches simultaneously, maintaining temporal coherence as targets transition between spatial regions:
\begin{equation}
\mathbf{x}_t^{(\ell)} \rightarrow \{\mathbf{p}_1^{(\ell)}, \mathbf{p}_2^{(\ell)}, \ldots, \mathbf{p}_P^{(\ell)}\}, \quad \mathbf{p}_i^{(\ell)} \in \mathbb{R}^{d_p},
\end{equation}
where $P = \left\lfloor \frac{N - p}{s} + 1 \right\rfloor \times \left\lfloor \frac{K - p}{s} + 1 \right\rfloor$ is the number of patches per frame, and $d_p = p^2$ is the patch dimension. The overlap factor $\frac{p}{s}$ is chosen to balance computational efficiency with motion continuity.

These patches are linearly projected to the transformer embedding dimension $d$:
\begin{equation}
\mathbf{z}_i^{(\ell)} = \mathbf{W}_{\text{proj}} \mathbf{p}_i^{(\ell)} + \mathbf{b}_{\text{proj}}, \quad \mathbf{z}_i^{(\ell)} \in \mathbb{R}^d,
\end{equation}
where $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{d \times d_p}$ is the projection matrix and $\mathbf{b}_{\text{proj}}$ is the bias \cite{dosovitskiy2020image}.

\noindent To encode spatiotemporal structure, we inject three types of positional information into each token:
\begin{equation}
\tilde{\mathbf{z}}_i^{(\ell)} = \mathbf{z}_i^{(\ell)} + \mathbf{e}_{\text{spatial}}^{(i)} + \mathbf{e}_{\text{temporal}}^{(\ell)} + \mathbf{e}_{\text{diffusion}}(t),
\end{equation}
where $\mathbf{e}_{\text{spatial}}^{(i)} \in \mathbb{R}^d$ encodes the 2D spatial position of patch $i$ within the \gls{rd} map using learnable embeddings, $\mathbf{e}_{\text{temporal}}^{(\ell)} \in \mathbb{R}^d$ encodes the temporal position $\ell$ in the sequence using learnable embeddings, and $\mathbf{e}_{\text{diffusion}}(t) \in \mathbb{R}^d$ encodes the diffusion timestep via sinusoidal functions following~\cite{vaswani2017attention}. Unlike spatial and temporal embeddings, the diffusion timestep embedding uses fixed sinusoidal encoding rather than learned parameters because the model must generalize to arbitrary noise levels $t$ during the continuous diffusion process, whereas spatial and temporal positions come from a fixed discrete set during both training and inference.

\noindent The complete sequence is represented as a flattened token sequence:
\begin{equation}
\mathbf{Z}_t = \big[\tilde{\mathbf{z}}_1^{(1)}, \ldots, \tilde{\mathbf{z}}_P^{(1)}, \tilde{\mathbf{z}}_1^{(2)}, \ldots, \tilde{\mathbf{z}}_P^{(L)}\big] \in \mathbb{R}^{(L \cdot P) \times d}.
\end{equation}
The Diffusion Transformer processes $\mathbf{Z}_t$ through $B$ transformer blocks. Each block applies \gls{adaln} conditioned on the diffusion time step $t$ and optional conditioning variables $\mathbf{c}$:
\begin{equation}
\begin{aligned}
\mathbf{Y}_b &= \text{MSA}_{temp}\big(\text{adaLN}(\mathbf{Z}_b, t, \mathbf{c})\big) + \mathbf{Z}_b, \\
\mathbf{Z}_{b+1} &= \text{MLP}\big(\text{adaLN}(\mathbf{Y}_b, t, \mathbf{c})\big) + \mathbf{Y}_b,
\end{aligned}
\end{equation}
where $MSA_{temp}$ denotes temporal multi-head self-attention, and $\mathbf{Z}_b$ is the output of block $b$ (with $\mathbf{Z}_0 = \mathbf{Z}_t$).

The \gls{adaln} modulates the normalization parameters based on the diffusion time step:
\begin{equation}
\text{adaLN}(\mathbf{z}, t, \mathbf{c}) = \boldsymbol{\gamma}(t, \mathbf{c}) \odot \frac{\mathbf{z} - \mu(\mathbf{z})}{\sigma(\mathbf{z})} + \boldsymbol{\beta}(t, \mathbf{c}),
\end{equation}
where $\mu(\mathbf{z})$ and $\sigma(\mathbf{z})$ are the mean and standard deviation of $\mathbf{z}$, and $\boldsymbol{\gamma}(t, \mathbf{c})$ and $\boldsymbol{\beta}(t, \mathbf{c})$ are learned scale and shift parameters predicted by a time-conditioning network:
\begin{equation}
[\boldsymbol{\gamma}(t, \mathbf{c}), \boldsymbol{\beta}(t, \mathbf{c})] = \text{MLP}_{\text{cond}}\big([\mathbf{e}_{\text{diffusion}}(t), \phi(\mathbf{c})]\big),
\end{equation}
where $\phi(\mathbf{c})$ is an embedding of the conditioning vector \cite{peebles2023scalable}.

The multi-head self-attention mechanism operates solely along the temporal dimension, enabling each spatial token to attend to its temporal evolution across frames. Let $\mathbf{z}_{i,1:L} = [\tilde{\mathbf{z}}_i^{(1)}, \ldots, \tilde{\mathbf{z}}_i^{(L)}] \in \mathbb{R}^{L \times d}$ denote the sequence of embeddings corresponding to spatial patch $i$ across all $L$ frames. Temporal self-attention is computed independently for each spatial location:
\begin{equation}
\text{MSA}_{\text{temp}}(\mathbf{z}_{i,1:L}) = \text{Concat}\big(\text{head}_1^{(i)}, \ldots, \text{head}_H^{(i)}\big) \mathbf{W}^O,
\end{equation}
where each head $h$ follows the scaled dot-product attention mechanism \cite{vaswani2017attention}, attending only across time indices:
\begin{equation}
\text{head}_h^{(i)} = \text{Softmax}\!\left(\frac{\mathbf{Q}_h^{(i)} {\mathbf{K}_h^{(i)}}^\top}{\sqrt{d_h}}\right) \mathbf{V}_h^{(i)},
\end{equation}
with $\mathbf{Q}_h^{(i)}, \mathbf{K}_h^{(i)}, \mathbf{V}_h^{(i)} \in \mathbb{R}^{L \times d_h}$ representing the query, key, and value projections for spatial patch $i$. 
This temporal-only attention preserves the physical independence of distant spatial regions while effectively capturing target motion dynamics across frames.

After $B$ transformer blocks, the final token representations $\mathbf{Z}_B$ are projected back to patch space and reassembled into the sequence format. The predicted noise for each frame is obtained via:
\begin{equation}
\boldsymbol{\epsilon}_\theta(\mathbf{x}_t^{(\ell)}, t, \mathbf{c}) = \text{Unpatchify}\big(\mathbf{W}_{\text{out}} \mathbf{Z}_B^{(\ell)} + \mathbf{b}_{\text{out}}\big),
\end{equation}
where $\mathbf{Z}_B^{(\ell)}$ denotes the subset of tokens corresponding to frame $\ell$, and $\mathbf{W}_{\text{out}} \in \mathbb{R}^{d_p \times d}$ projects back to patch dimension.

The training objective combines per-frame denoising with spatiotemporal consistency:
\begin{equation}
\begin{aligned}
\mathcal{L}_{\text{DiT}} &= \mathbb{E}_{t, \mathbf{X}_0, \boldsymbol{\epsilon}} \Bigg[ \sum_{\ell=1}^{L} \Big\| \boldsymbol{\epsilon}^{(\ell)} - \boldsymbol{\epsilon}_\theta\big(\mathbf{x}_t^{(\ell)}, t, \mathbf{c}\big) \Big\|^2 \Bigg] = \mathbb{E}_{t, \mathbf{X}_0, \boldsymbol{\epsilon}} \big[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{X}_t, t, \mathbf{c})\|_F^2 \big],
\end{aligned}
\end{equation}
where $\|\cdot\|_F$ denotes the Frobenius norm over the entire sequence.

To enforce physical motion constraints and temporal smoothness, we augment the base diffusion loss with regularization terms that explicitly model radar-specific dynamics. First, we penalize large inter-frame variations in the predicted clean data $\hat{\mathbf{x}}_0^{(\ell)}$, which can be estimated from the noisy input via:
\begin{equation}
\hat{\mathbf{x}}_0^{(\ell)} = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t^{(\ell)} - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t^{(\ell)}, t, \mathbf{c}) \right).
\end{equation}

The temporal smoothness loss penalizes discontinuities:
\begin{equation}
\mathcal{L}_{\text{smooth}} = \mathbb{E} \Bigg[ \sum_{\ell=2}^{L} \omega_t \Big\| \hat{\mathbf{x}}_0^{(\ell)} - \hat{\mathbf{x}}_0^{(\ell-1)} \Big\|^2 \Bigg],
\end{equation}
where $\omega_t = (1 - \bar{\alpha}_t)$ weights the loss more heavily at later diffusion steps when predictions are more reliable.

Second, we enforce target trajectory consistency. 

Let $\mathcal{T}_k^{(\ell)} \in \mathbb{R}^2$ denote the spatial coordinates of target $k$ at frame $\ell$, 
obtained by detecting the local maxima in the predicted RD map $\hat{\mathbf{x}}_0^{(\ell)}$ above a 
threshold $\tau$.
The trajectory consistency loss enforces constant velocity motion:
\begin{equation}
\mathcal{L}_{\text{traj}} = \mathbb{E} \Bigg[ \sum_{k=1}^{M} \sum_{\ell=3}^{L} \Big\| \big(\mathcal{T}_k^{(\ell)} - \mathcal{T}_k^{(\ell-1)}\big) - \big(\mathcal{T}_k^{(\ell-1)} - \mathcal{T}_k^{(\ell-2)}\big) \Big\|^2 \Bigg],
\end{equation}
which penalizes abrupt velocity changes, ensuring targets follow physically plausible kinematic paths \cite{baisa2020derivation}.

Additionally, we introduce a Doppler consistency constraint. Since Doppler shift encodes radial velocity, targets with consistent motion should exhibit stable Doppler profiles modulo acceleration effects:
\begin{equation}
\mathcal{L}_{\text{Doppler}} = \mathbb{E} \Bigg[ \sum_{k=1}^{M} \sum_{\ell=2}^{L} \Big\| \mathbf{d}_k^{(\ell)} - \mathbf{d}_k^{(\ell-1)} \Big\|^2 \Bigg],
\end{equation}
where $\mathbf{d}_k^{(\ell)}$ represents the Doppler centroid of target $k$ at frame $\ell$.

The complete training objective becomes:
\begin{equation}
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DiT}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} + \lambda_{\text{traj}} \mathcal{L}_{\text{traj}} + \lambda_{\text{Doppler}} \mathcal{L}_{\text{Doppler}},
\end{equation}
where $\lambda_{\text{smooth}}, \lambda_{\text{traj}}, \lambda_{\text{Doppler}} > 0$ balance the different objective components.

The Diffusion Transformer naturally accommodates rich conditioning through the adaptive normalization mechanism \cite{peebles2023scalable}. 

We condition on several radar specific parameters, For controllable target dynamics, we provide kinematic parameters $\mathbf{c}_{\text{motion}} = \{(\mathbf{v}_k^{(0)}, \mathbf{a}_k)\}_{k=1}^M$, where $\mathbf{v}_k^{(0)} \in \mathbb{R}^2$ is the initial velocity and $\mathbf{a}_k \in \mathbb{R}^2$ is the acceleration of target $k$. These vectors are embedded and concatenated with the diffusion time embedding:
\begin{equation}
\mathbf{c} = \big[\phi_{\text{motion}}(\mathbf{v}_1^{(0)}, \mathbf{a}_1), \ldots, \phi_{\text{motion}}(\mathbf{v}_M^{(0)}, \mathbf{a}_M)\big],
\end{equation}
enabling generation of sequences where targets follow prescribed trajectories.

We also condition on scene attributes such as clutter intensity $\sigma_{\text{clutter}}$, \gls{scr}, and temporal clutter correlation $\rho_{\text{clutter}} \in [0,1]$:
\begin{equation}
\mathbf{c}_{\text{env}} = [\sigma_{\text{clutter}}, \text{SCR}, \rho_{\text{clutter}}]^\top,
\end{equation}
which controls background statistics across the sequence.

For multi-class target generation we use learnable class embeddings $\mathbf{e}_{\text{class}}^{(k)} \in \mathbb{R}^d$ for each target $k$, concatenated with other conditioning vectors.

The unified conditioning vector is:
\begin{equation}
\mathbf{c}_{\text{full}} = \text{MLP}_{\text{fusion}}\big([\mathbf{c}_{\text{motion}}, \mathbf{c}_{\text{env}}, \mathbf{e}_{\text{class}}]\big),
\end{equation}
which is fed into the adaLN modules throughout the transformer.


