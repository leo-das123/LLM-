"""
llm_explainer_ui.py

Visual UI (Streamlit) for the GPT-2 LLM introspection core.

Features:
    - Load GPT-2 once (cached) and introspect:
        * token embeddings
        * positional embeddings
        * per-layer Q/K/V
        * per-layer FFN internals (fc1, fc2)
        * logits, next token distribution

    - Modes:
        1) Single Forward Pass (8-step pipeline, all steps visible)
        2) Step-by-step Generation (per-token X-ray summary)

Run:
    streamlit run llm_explainer_ui.py
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import graphviz  # <-- NEW: for flowchart


# =============================================================================
# Introspection core (IntrospectGPT2)
# =============================================================================


class IntrospectGPT2:
    """
    GPT-2 core with deep introspection using forward hooks.

    Captures:
        - token embeddings (wte)
        - positional embeddings (wpe)
        - per-layer Q, K, V (from c_attn)
        - per-layer FFN outputs (fc1 = c_fc, fc2 = c_proj)
        - logits, attentions, hidden_states from the model outputs
    """

    def __init__(self, model_name: str = "gpt2") -> None:
        # 1) Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2) Model
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

        # 3) Config shortcuts
        self.config = self.model.config
        self.n_layer = self.config.n_layer
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        self.head_dim = self.n_embd // self.n_head

        # 4) Internal storage for hooks
        self.internals: Dict[str, Any] = {}
        self._hooks: List[Any] = []

        # 5) Register hooks once
        self._register_hooks()

    # ---------------------------------------------------------------------
    # Hook registration
    # ---------------------------------------------------------------------
    def _register_hooks(self) -> None:
        transformer = self.model.transformer

        # Prepare internal structure
        self.internals["token_embeddings"] = None
        self.internals["position_embeddings"] = None
        self.internals["layers"] = [
            {"Q": None, "K": None, "V": None, "fc1": None, "fc2": None}
            for _ in range(self.n_layer)
        ]

        # --- Embeddings hooks ---
        def token_embedding_hook(module, inp, out):
            # out: (batch, seq_len, n_embd)
            self.internals["token_embeddings"] = out.detach()

        def position_embedding_hook(module, inp, out):
            # out: (batch, seq_len, n_embd)
            self.internals["position_embeddings"] = out.detach()

        self._hooks.append(
            transformer.wte.register_forward_hook(token_embedding_hook)
        )
        self._hooks.append(
            transformer.wpe.register_forward_hook(position_embedding_hook)
        )

        # --- Per-layer Q/K/V and FFN hooks ---
        def make_c_attn_hook(layer_idx: int):
            def hook(module, inp, out):
                # out: (batch, seq_len, 3 * n_embd)
                q, k, v = out.split(self.n_embd, dim=-1)

                def reshape(x: torch.Tensor) -> torch.Tensor:
                    b, s, d = x.size()
                    x = x.view(b, s, self.n_head, self.head_dim)
                    return (
                        x.permute(0, 2, 1, 3)
                        .detach()
                    )  # (batch, head, seq, head_dim)

                self.internals["layers"][layer_idx]["Q"] = reshape(q)
                self.internals["layers"][layer_idx]["K"] = reshape(k)
                self.internals["layers"][layer_idx]["V"] = reshape(v)

            return hook

        def make_c_fc_hook(layer_idx: int):
            def hook(module, inp, out):
                # out: (batch, seq_len, intermediate_dim)
                self.internals["layers"][layer_idx]["fc1"] = out.detach()

            return hook

        def make_c_proj_hook(layer_idx: int):
            def hook(module, inp, out):
                # out: (batch, seq_len, n_embd)
                self.internals["layers"][layer_idx]["fc2"] = out.detach()

            return hook

        for layer_idx, block in enumerate(transformer.h):
            # Attention projection (QKV)
            self._hooks.append(
                block.attn.c_attn.register_forward_hook(make_c_attn_hook(layer_idx))
            )
            # FFN first linear
            self._hooks.append(
                block.mlp.c_fc.register_forward_hook(make_c_fc_hook(layer_idx))
            )
            # FFN second linear
            self._hooks.append(
                block.mlp.c_proj.register_forward_hook(make_c_proj_hook(layer_idx))
            )

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _encode(self, text: str, padding: bool = False) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=padding,
            truncation=True,
        )

    def _reset_internals(self) -> None:
        """
        Reset internals in-place so that hooks keep writing into the same objects.
        """
        if "layers" not in self.internals:
            self.internals["layers"] = [
                {"Q": None, "K": None, "V": None, "fc1": None, "fc2": None}
                for _ in range(self.n_layer)
            ]

        self.internals["token_embeddings"] = None
        self.internals["position_embeddings"] = None
        for layer in self.internals["layers"]:
            layer["Q"] = None
            layer["K"] = None
            layer["V"] = None
            layer["fc1"] = None
            layer["fc2"] = None

    # ---------------------------------------------------------------------
    # Forward with inspection
    # ---------------------------------------------------------------------
    def forward_with_inspection(self, text: str) -> Dict[str, Any]:
        """
        Run a single forward pass and capture:
          - inputs: {input_ids, attention_mask}
          - logits
          - attentions
          - hidden_states
          - internals: embeddings, Q/K/V, FC1/FC2
        """
        self._reset_internals()
        enc = self._encode(text, padding=False)
        input_ids = enc["input_ids"]
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
            )

        return {
            "inputs": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            "logits": outputs.logits,
            "attentions": outputs.attentions,
            "hidden_states": outputs.hidden_states,
            "internals": self.internals,
        }

    # ---------------------------------------------------------------------
    # Text generation
    # ---------------------------------------------------------------------
    def generate_text(
        self,
        text: str,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 2,
    ) -> str:
        """
        Generate a continuation from the given text using sampling.
        """
        enc = self._encode(text, padding=True)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        with torch.no_grad():
            out_ids = self.model.generate(**gen_kwargs)

        # Decode only the new tokens
        input_len = input_ids.shape[1]
        generated_ids = out_ids[0, input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# =============================================================================
# Introspection configuration
# =============================================================================


@dataclass
class IntrospectionConfig:
    layers_to_show: int = 3
    heads_to_show: int = 4
    num_dims_to_show: int = 8
    top_keys_per_head: int = 4
    topk_next_token: int = 10

    def clip_to_model(self, n_layer: int, n_head: int) -> "IntrospectionConfig":
        return IntrospectionConfig(
            layers_to_show=min(self.layers_to_show, n_layer),
            heads_to_show=min(self.heads_to_show, n_head),
            num_dims_to_show=self.num_dims_to_show,
            top_keys_per_head=self.top_keys_per_head,
            topk_next_token=self.topk_next_token,
        )


# =============================================================================
# Streamlit helpers: cache model
# =============================================================================


@st.cache_resource(show_spinner=True)
def load_llm() -> IntrospectGPT2:
    return IntrospectGPT2("gpt2")


# =============================================================================
# Visualization helpers
# =============================================================================


def plot_bar(values: List[float], labels: List[str], title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots()
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    st.pyplot(fig)


def plot_heatmap(matrix: np.ndarray, x_labels: List[str], y_labels: List[str], title: str):
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    fig.colorbar(cax)
    fig.tight_layout()
    st.pyplot(fig)


# =============================================================================
# NEW: Flowchart + simple explanation of the 8 steps
# =============================================================================

def render_pipeline_flowchart():
    """
    Show a visual flowchart of the 8 conceptual steps for the transformer forward pass,
    using Streamlit's built-in Graphviz support (no extra dependencies).
    """
    dot_source = r"""
    digraph G {
        rankdir=LR;
        node [shape=box, style="rounded,filled", color="#1f77b4", fontname="Helvetica", fontsize=10];
        edge [fontname="Helvetica", fontsize=9];

        raw  [label="Raw text\n(user prompt)", shape=oval, color="#2ca02c", style="filled"];
        s1   [label="1. Tokenization\nText → subword tokens"];
        s2   [label="2. Numericalization\nTokens → token IDs"];
        s3   [label="3. Embeddings\nIDs → dense vectors"];
        s4   [label="4. Positional encodings\nAdd position info"];
        s5   [label="5. Self-attention\nQ/K/V → context"];
        s6   [label="6. Feed-forward\nFC1 → GELU → FC2"];
        s7   [label="7. Logits & softmax\nScores → probabilities"];
        s8   [label="8. Decoding\nPick next token"];
        out  [label="Updated text\n(prompt + next token)", shape=oval, color="#ff7f0e", style="filled"];

        raw -> s1 -> s2 -> s3 -> s4 -> s5 -> s6 -> s7 -> s8 -> out;
    }
    """
    st.graphviz_chart(dot_source)

def render_pipeline_text_explanation():
    """
    Short, easy explanation of each step, in human language.
    """
    st.markdown("#### 8-step overview (easy explanation)")
    st.markdown(
        """
1. **Tokenization** – Split your raw text into subword pieces (like `Hel`, `lo`, `!`) that GPT-2 actually knows.  
2. **Numericalization** – Turn each token into a number (its vocabulary ID), so the model can work with it.  
3. **Embeddings** – Look up each ID in a big table to get a high-dimensional vector: this is the model’s “meaning” of that token.  
4. **Positional encodings** – Add another vector that encodes **where** the token is in the sequence (1st, 2nd, 3rd, …), so order matters.  
5. **Self-attention** – For every token, build Q/K/V vectors and let it “look at” other tokens with different strengths (attention weights). Combine their values into a **context vector**.  
6. **Feed-forward network** – Pass each token through a small MLP (FC1 → GELU → FC2) to non-linearly mix and transform features.  
7. **Logits → probabilities** – Use a final linear layer to get one score per vocabulary token, then softmax to convert scores into a probability distribution over the next token.  
8. **Decoding** – Apply a decoding strategy (here: greedy pick of highest probability) to choose the actual next token, and append it to the text.
        """
    )


# =============================================================================
# Main app logic – Single forward (show all 8 steps)
# =============================================================================


def run_single_forward(llm: IntrospectGPT2, prompt: str, cfg: IntrospectionConfig):
    cfg = cfg.clip_to_model(llm.n_layer, llm.n_head)
    info = llm.forward_with_inspection(prompt)

    input_ids = info["inputs"]["input_ids"][0]
    tokens = llm.tokenizer.convert_ids_to_tokens(input_ids)
    logits = info["logits"][:, -1, :]
    probs = torch.softmax(logits, dim=-1)[0]

    internals = info["internals"]
    token_embeds = internals.get("token_embeddings")
    pos_embeds = internals.get("position_embeddings")
    layer_internals = internals.get("layers", [])

    st.markdown("### Full 8-Step Transformer Pipeline (single forward pass)")

    # --- NEW: high-level flowchart + explanation ------------------------
    with st.expander("High-level 8-step pipeline (flowchart view)", expanded=True):
        render_pipeline_flowchart()
        render_pipeline_text_explanation()
    # --------------------------------------------------------------------

    # ------------------------------------------------------------------ #
    # STEP 1 + 2
    # ------------------------------------------------------------------ #
    with st.expander("Step 1 & 2 – Tokenization and Numericalization", expanded=True):
        st.markdown("#### Step 1: Tokenization (Text → tokens)")
        st.write("GPT-2 uses BPE (byte-pair encoding) tokens. Here is your prompt as GPT-2 tokens:")
        df_tokens = pd.DataFrame(
            {
                "Position": list(range(len(tokens))),
                "Token": tokens,
                "Token ID": input_ids.tolist(),
            }
        )
        st.dataframe(df_tokens, use_container_width=True)

        st.markdown("#### Step 2: Numericalization (Tokens → IDs)")
        st.write("Same information as a simple Python list of token IDs:")
        st.code(input_ids.tolist())

    # ------------------------------------------------------------------ #
    # STEP 3 – EMBEDDINGS
    # ------------------------------------------------------------------ #
    with st.expander("Step 3 – Embeddings (Token IDs → dense vectors)", expanded=True):
        st.write(
            "Each token ID is mapped into a high-dimensional vector (the word embedding). "
            "We inspect the norms and the first few dimensions."
        )
        if token_embeds is None:
            st.warning("No token embeddings captured.")
        else:
            emb = token_embeds[0]  # (seq, dim)
            norms = [float(v.norm().item()) for v in emb]

            st.caption("Embedding norm per token (magnitude of each embedding vector).")
            plot_bar(
                norms,
                [f"{i}:{t}" for i, t in enumerate(tokens)],
                "Embedding norms for tokens",
                "Token index",
                "L2 norm",
            )

            idx = st.slider(
                "Inspect embedding for token position",
                0,
                len(tokens) - 1,
                0,
                key="emb_token_idx",
            )
            vec = emb[idx]
            dims = list(range(cfg.num_dims_to_show))
            vals = vec[: cfg.num_dims_to_show].tolist()
            st.markdown(
                f"**Token [{idx}] = `{tokens[idx]}` – first {cfg.num_dims_to_show} embedding dimensions**"
            )
            plot_bar(
                vals,
                [str(d) for d in dims],
                "First N embedding dimensions",
                "Embedding dimension index",
                "Value",
            )

    # ------------------------------------------------------------------ #
    # STEP 4 – POSITIONAL ENCODINGS
    # ------------------------------------------------------------------ #
    with st.expander("Step 4 – Positional encodings (add order/position to tokens)", expanded=True):
        st.write(
            "The model adds a positional embedding to each token embedding so it knows "
            "which token comes first, second, etc."
        )
        if pos_embeds is None:
            st.warning("No positional embeddings captured.")
        else:
            pos = pos_embeds[0]  # (seq, dim)
            norms = [float(v.norm().item()) for v in pos]
            st.caption("Norm of the positional embedding for each position in the sequence.")
            plot_bar(
                norms,
                [str(i) for i in range(len(pos))],
                "Positional embedding norms",
                "Position index",
                "L2 norm",
            )

            idx = st.slider(
                "Inspect positional embedding for position",
                0,
                len(pos) - 1,
                0,
                key="pos_idx",
            )
            vec = pos[idx]
            dims = list(range(cfg.num_dims_to_show))
            vals = vec[: cfg.num_dims_to_show].tolist()
            st.markdown(
                f"**Position [{idx}] – first {cfg.num_dims_to_show} positional dimensions**"
            )
            plot_bar(
                vals,
                [str(d) for d in dims],
                "First N positional embedding dimensions",
                "Dimension index",
                "Value",
            )

    # ------------------------------------------------------------------ #
    # STEP 5 – SELF-ATTENTION
    # ------------------------------------------------------------------ #
    with st.expander("Step 5 – Self-Attention (Q/K/V → attention weights → context)", expanded=True):
        st.write(
            "Each attention layer/head turns token representations into Q (queries), "
            "K (keys), and V (values). The last token queries all positions and "
            "produces attention weights, then a context vector."
        )

        if not layer_internals:
            st.warning("No layer internals captured.")
        else:
            layer_idx = st.slider(
                "Layer to inspect",
                0,
                min(cfg.layers_to_show, len(layer_internals)) - 1,
                0,
                key="attn_layer_idx",
            )
            layer_data = layer_internals[layer_idx]
            Q = layer_data["Q"]
            K = layer_data["K"]
            V = layer_data["V"]

            if Q is None or K is None or V is None:
                st.warning(f"No Q/K/V captured for layer {layer_idx}.")
            else:
                Qb = Q[0]  # (head, seq, head_dim)
                Kb = K[0]
                Vb = V[0]
                n_heads = Qb.shape[0]

                head_idx = st.slider(
                    "Attention head to inspect",
                    0,
                    min(cfg.heads_to_show, n_heads) - 1,
                    0,
                    key="attn_head_idx",
                )

                q_head = Qb[head_idx]  # (seq, head_dim)
                k_head = Kb[head_idx]
                v_head = Vb[head_idx]

                st.markdown(f"**Layer {layer_idx}, Head {head_idx}**")

                last_q = q_head[-1]
                last_k = k_head[-1]
                st.write(
                    f"Norm(Q_last) = {float(last_q.norm()):.4f}, "
                    f"Norm(K_last) = {float(last_k.norm()):.4f}"
                )

                head_dim = q_head.shape[-1]
                scores = q_head @ k_head.t()  # (seq, seq)
                scores = scores / math.sqrt(head_dim)
                weights = torch.softmax(scores[-1], dim=-1)  # (seq,)

                top_k = min(cfg.top_keys_per_head, len(tokens))
                vals, idxs = torch.topk(weights, k=top_k)
                df_top = pd.DataFrame(
                    {
                        "Position": idxs.tolist(),
                        "Token": [tokens[i] for i in idxs.tolist()],
                        "Attention weight": [float(v) for v in vals],
                    }
                )
                st.markdown("**Top-attended tokens for the LAST sequence position**")
                st.dataframe(df_top, use_container_width=True)

                matrix = weights.unsqueeze(0).numpy()  # (1, seq)
                plot_heatmap(
                    matrix,
                    x_labels=[f"{i}:{t}" for i, t in enumerate(tokens)],
                    y_labels=["Last token query"],
                    title="Attention weights (LAST token over all positions)",
                )

                context = weights.unsqueeze(0) @ v_head
                context = context.squeeze(0)
                st.markdown("**Context vector (weighted sum of V)**")
                dims = list(range(cfg.num_dims_to_show))
                vals = context[: cfg.num_dims_to_show].tolist()
                plot_bar(
                    vals,
                    [str(d) for d in dims],
                    "First N context dimensions",
                    "Dimension index",
                    "Value",
                )

    # ------------------------------------------------------------------ #
    # STEP 6 – FEED-FORWARD
    # ------------------------------------------------------------------ #
    with st.expander("Step 6 – Feed-Forward Network (FC1 → activation → FC2)", expanded=True):
        st.write(
            "Each transformer layer has a small MLP that processes each token independently "
            "after attention. We inspect the norms of FC1 and FC2 outputs for the last token."
        )
        if not layer_internals:
            st.warning("No FFN internals captured.")
        else:
            norms_fc1 = []
            norms_fc2 = []
            layer_ids = []

            max_layers = min(cfg.layers_to_show, len(layer_internals))
            for i in range(max_layers):
                layer_data = layer_internals[i]
                fc1 = layer_data["fc1"]
                fc2 = layer_data["fc2"]
                if fc1 is None or fc2 is None:
                    continue
                fc1_last = fc1[0, -1]
                fc2_last = fc2[0, -1]
                norms_fc1.append(float(fc1_last.norm()))
                norms_fc2.append(float(fc2_last.norm()))
                layer_ids.append(i)

            if not layer_ids:
                st.warning("No FC1/FC2 data available.")
            else:
                df_ffn = pd.DataFrame(
                    {
                        "Layer": layer_ids,
                        "FC1 norm (last token)": norms_fc1,
                        "FC2 norm (last token)": norms_fc2,
                    }
                )
                st.dataframe(df_ffn, use_container_width=True)

                fig, ax = plt.subplots()
                x = np.arange(len(layer_ids))
                width = 0.35
                ax.bar(x - width / 2, norms_fc1, width, label="FC1")
                ax.bar(x + width / 2, norms_fc2, width, label="FC2")
                ax.set_xticks(x)
                ax.set_xticklabels([str(i) for i in layer_ids])
                ax.set_xlabel("Layer")
                ax.set_ylabel("L2 norm")
                ax.set_title("FFN norms per layer (last token)")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)

    # ------------------------------------------------------------------ #
    # STEP 7 – LOGITS → PROBABILITIES
    # ------------------------------------------------------------------ #
    with st.expander("Step 7 – Logits → probabilities (next-token distribution)", expanded=True):
        st.write(
            "The final layer outputs raw scores (logits) for each vocabulary token. "
            "Softmax turns them into a probability distribution over the next token."
        )
        topk = torch.topk(probs, k=min(cfg.topk_next_token, probs.numel()))
        ids = topk.indices.tolist()
        ps = topk.values.tolist()
        texts = [llm.tokenizer.decode([tid]) for tid in ids]
        df_next = pd.DataFrame(
            {
                "Token ID": ids,
                "Token (decoded)": texts,
                "Probability": [float(p) for p in ps],
            }
        )
        st.dataframe(df_next, use_container_width=True)

        plot_bar(
            [float(p) for p in ps],
            [f"{tid}:{t}" for tid, t in zip(ids, texts)],
            "Top-K next-token probabilities",
            "Token",
            "Probability",
        )

    # ------------------------------------------------------------------ #
    # STEP 8 – DECODING
    # ------------------------------------------------------------------ #
    with st.expander("Step 8 – Decoding (choose next token)", expanded=True):
        st.write(
            "A decoding strategy (here: greedy) chooses one token from the probability distribution "
            "and appends it to the sequence as the next token."
        )
        greedy_id = torch.argmax(probs, dim=-1).item()
        greedy_tok = llm.tokenizer.convert_ids_to_tokens([greedy_id])[0]
        st.markdown(
            f"**Greedy next token:** id={greedy_id}, token=`{greedy_tok}`, "
            f"prob={float(probs[greedy_id]):.6f}"
        )


# =============================================================================
# Step-by-step generation (summary view)
# =============================================================================


def run_step_by_step(llm: IntrospectGPT2, prompt: str, cfg: IntrospectionConfig, num_steps: int):
    """
    Generate tokens one at a time, and for EACH generation step,
    show the full 8-step pipeline (with sub-steps) directly in the UI.

    No selectbox, no manual step selection — everything is visible at once.
    """
    cfg = cfg.clip_to_model(llm.n_layer, llm.n_head)
    current_text = prompt

    st.markdown("### Step-by-step generation (8-step pipeline for each new token)")
    st.write(
        "At each step we run a forward pass on the current text, inspect all 8 steps, "
        "then append the greedy next token to the text."
    )
    st.write(f"**Initial prompt:** `{prompt}`")

    # --- NEW: high-level flowchart at top of step-by-step mode ----------
    with st.expander("High-level 8-step pipeline (flowchart view)", expanded=True):
        render_pipeline_flowchart()
        render_pipeline_text_explanation()
    # --------------------------------------------------------------------

    # For each generation step
    for step in range(num_steps):
        st.markdown("---")
        st.markdown(f"## Generation step {step + 1}")
        st.markdown(f"**Input text at this step:** `{current_text}`")

        # Forward pass with introspection
        info = llm.forward_with_inspection(current_text)

        input_ids = info["inputs"]["input_ids"][0]
        tokens = llm.tokenizer.convert_ids_to_tokens(input_ids)
        logits = info["logits"][:, -1, :]
        probs = torch.softmax(logits, dim=-1)[0]

        internals = info["internals"]
        token_embeds = internals.get("token_embeddings")
        pos_embeds = internals.get("position_embeddings")
        layer_internals = internals.get("layers", [])

        # ================================================================
        # STEP 1 & 2 – TOKENIZATION + NUMERICALIZATION
        # ================================================================
        with st.expander(
            f"Step 1 & 2 – Tokenization and Numericalization (step {step + 1})",
            expanded=True,
        ):
            st.markdown("#### Step 1: Tokenization (Text → tokens)")
            st.write("GPT-2 BPE tokens for the current text:")
            df_tokens = pd.DataFrame(
                {
                    "Position": list(range(len(tokens))),
                    "Token": tokens,
                    "Token ID": input_ids.tolist(),
                }
            )
            st.dataframe(df_tokens, use_container_width=True)

            st.markdown("#### Step 2: Numericalization (Tokens → IDs)")
            st.write("Same information as a list of IDs:")
            st.code(input_ids.tolist())

        # ================================================================
        # STEP 3 – EMBEDDINGS
        # ================================================================
        with st.expander(
            f"Step 3 – Embeddings (Token IDs → dense vectors) [step {step + 1}]",
            expanded=True,
        ):
            st.write(
                "Each token ID is mapped into a high-dimensional vector (the word embedding). "
                "We inspect norms and first few dimensions."
            )
            if token_embeds is None:
                st.warning("No token embeddings captured.")
            else:
                emb = token_embeds[0]  # (seq, dim)
                norms = [float(v.norm().item()) for v in emb]

                st.caption("Embedding norm per token (magnitude of each embedding vector).")
                plot_bar(
                    norms,
                    [f"{i}:{t}" for i, t in enumerate(tokens)],
                    f"Embedding norms for tokens (step {step + 1})",
                    "Token index",
                    "L2 norm",
                )

                idx = st.slider(
                    "Inspect embedding for token position",
                    0,
                    len(tokens) - 1,
                    0,
                    key=f"emb_token_idx_step_{step}",
                )
                vec = emb[idx]
                dims = list(range(cfg.num_dims_to_show))
                vals = vec[: cfg.num_dims_to_show].tolist()
                st.markdown(
                    f"**Token [{idx}] = `{tokens[idx]}` – first "
                    f"{cfg.num_dims_to_show} embedding dimensions**"
                )
                plot_bar(
                    vals,
                    [str(d) for d in dims],
                    f"First N embedding dimensions (step {step + 1})",
                    "Embedding dimension index",
                    "Value",
                )

        # ================================================================
        # STEP 4 – POSITIONAL ENCODINGS
        # ================================================================
        with st.expander(
            f"Step 4 – Positional encodings (add order/position) [step {step + 1}]",
            expanded=True,
        ):
            st.write(
                "The model adds a positional embedding to each token embedding so it knows "
                "which token comes first, second, etc."
            )
            if pos_embeds is None:
                st.warning("No positional embeddings captured.")
            else:
                pos = pos_embeds[0]  # (seq, dim)
                norms = [float(v.norm().item()) for v in pos]
                st.caption("Norm of the positional embedding for each position.")
                plot_bar(
                    norms,
                    [str(i) for i in range(len(pos))],
                    f"Positional embedding norms (step {step + 1})",
                    "Position index",
                    "L2 norm",
                )

                idx = st.slider(
                    "Inspect positional embedding for position",
                    0,
                    len(pos) - 1,
                    0,
                    key=f"pos_idx_step_{step}",
                )
                vec = pos[idx]
                dims = list(range(cfg.num_dims_to_show))
                vals = vec[: cfg.num_dims_to_show].tolist()
                st.markdown(
                    f"**Position [{idx}] – first {cfg.num_dims_to_show} positional dimensions**"
                )
                plot_bar(
                    vals,
                    [str(d) for d in dims],
                    f"First N positional embedding dimensions (step {step + 1})",
                    "Dimension index",
                    "Value",
                )

        # ================================================================
        # STEP 5 – SELF-ATTENTION
        # ================================================================
        with st.expander(
            f"Step 5 – Self-Attention (Q/K/V → weights → context) [step {step + 1}]",
            expanded=True,
        ):
            st.write(
                "Each attention layer/head turns token representations into Q (queries), "
                "K (keys), and V (values). The last token queries all positions and "
                "produces attention weights, then a context vector."
            )

            if not layer_internals:
                st.warning("No layer internals captured.")
            else:
                import math

                layer_idx = st.slider(
                    "Layer to inspect",
                    0,
                    min(cfg.layers_to_show, len(layer_internals)) - 1,
                    0,
                    key=f"attn_layer_idx_step_{step}",
                )
                layer_data = layer_internals[layer_idx]
                Q = layer_data["Q"]
                K = layer_data["K"]
                V = layer_data["V"]

                if Q is None or K is None or V is None:
                    st.warning(f"No Q/K/V captured for layer {layer_idx}.")
                else:
                    Qb = Q[0]  # (head, seq, head_dim)
                    Kb = K[0]
                    Vb = V[0]
                    n_heads = Qb.shape[0]

                    head_idx = st.slider(
                        "Attention head to inspect",
                        0,
                        min(cfg.heads_to_show, n_heads) - 1,
                        0,
                        key=f"attn_head_idx_step_{step}",
                    )

                    q_head = Qb[head_idx]  # (seq, head_dim)
                    k_head = Kb[head_idx]
                    v_head = Vb[head_idx]

                    st.markdown(f"**Layer {layer_idx}, Head {head_idx}**")

                    last_q = q_head[-1]
                    last_k = k_head[-1]
                    st.write(
                        f"Norm(Q_last) = {float(last_q.norm()):.4f}, "
                        f"Norm(K_last) = {float(last_k.norm()):.4f}"
                    )

                    head_dim = q_head.shape[-1]
                    scores = q_head @ k_head.t()  # (seq, seq)
                    scores = scores / math.sqrt(head_dim)
                    weights = torch.softmax(scores[-1], dim=-1)  # (seq,)

                    top_k = min(cfg.top_keys_per_head, len(tokens))
                    vals, idxs = torch.topk(weights, k=top_k)
                    df_top = pd.DataFrame(
                        {
                            "Position": idxs.tolist(),
                            "Token": [tokens[i] for i in idxs.tolist()],
                            "Attention weight": [float(v) for v in vals],
                        }
                    )
                    st.markdown("**Top-attended tokens for the LAST position**")
                    st.dataframe(df_top, use_container_width=True)

                    matrix = weights.unsqueeze(0).numpy()  # (1, seq)
                    plot_heatmap(
                        matrix,
                        x_labels=[f"{i}:{t}" for i, t in enumerate(tokens)],
                        y_labels=["Last token query"],
                        title=f"Attention weights (LAST token) – step {step + 1}",
                    )

                    context = weights.unsqueeze(0) @ v_head
                    context = context.squeeze(0)
                    st.markdown("**Context vector (weighted sum of V)**")
                    dims = list(range(cfg.num_dims_to_show))
                    vals = context[: cfg.num_dims_to_show].tolist()
                    plot_bar(
                        vals,
                        [str(d) for d in dims],
                        f"First N context dimensions (step {step + 1})",
                        "Dimension index",
                        "Value",
                    )

        # ================================================================
        # STEP 6 – FEED-FORWARD
        # ================================================================
        with st.expander(
            f"Step 6 – Feed-Forward Network (FC1 → activation → FC2) [step {step + 1}]",
            expanded=True,
        ):
            st.write(
                "Each transformer layer has an MLP that processes each token independently "
                "after attention. We inspect the norms of FC1 and FC2 outputs for the last token."
            )
            if not layer_internals:
                st.warning("No FFN internals captured.")
            else:
                norms_fc1 = []
                norms_fc2 = []
                layer_ids = []

                max_layers = min(cfg.layers_to_show, len(layer_internals))
                for i in range(max_layers):
                    layer_data = layer_internals[i]
                    fc1 = layer_data["fc1"]
                    fc2 = layer_data["fc2"]
                    if fc1 is None or fc2 is None:
                        continue
                    fc1_last = fc1[0, -1]
                    fc2_last = fc2[0, -1]
                    norms_fc1.append(float(fc1_last.norm()))
                    norms_fc2.append(float(fc2_last.norm()))
                    layer_ids.append(i)

                if not layer_ids:
                    st.warning("No FC1/FC2 data available.")
                else:
                    df_ffn = pd.DataFrame(
                        {
                            "Layer": layer_ids,
                            "FC1 norm (last token)": norms_fc1,
                            "FC2 norm (last token)": norms_fc2,
                        }
                    )
                    st.dataframe(df_ffn, use_container_width=True)

                    fig, ax = plt.subplots()
                    x = np.arange(len(layer_ids))
                    width = 0.35
                    ax.bar(x - width / 2, norms_fc1, width, label="FC1")
                    ax.bar(x + width / 2, norms_fc2, width, label="FC2")
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(i) for i in layer_ids])
                    ax.set_xlabel("Layer")
                    ax.set_ylabel("L2 norm")
                    ax.set_title(
                        f"FFN norms per layer (last token) – step {step + 1}"
                    )
                    ax.legend()
                    fig.tight_layout()
                    st.pyplot(fig)

        # ================================================================
        # STEP 7 – LOGITS → PROBABILITIES
        # ================================================================
        with st.expander(
            f"Step 7 – Logits → probabilities (next-token distribution) [step {step + 1}]",
            expanded=True,
        ):
            st.write(
                "The final layer outputs raw scores (logits). Softmax turns them "
                "into probabilities over the vocabulary."
            )
            topk = torch.topk(probs, k=min(cfg.topk_next_token, probs.numel()))
            ids = topk.indices.tolist()
            ps = topk.values.tolist()
            texts = [llm.tokenizer.decode([tid]) for tid in ids]
            df_next = pd.DataFrame(
                {
                    "Token ID": ids,
                    "Token (decoded)": texts,
                    "Probability": [float(p) for p in ps],
                }
            )
            st.dataframe(df_next, use_container_width=True)

            plot_bar(
                [float(p) for p in ps],
                [f"{tid}:{t}" for tid, t in zip(ids, texts)],
                f"Top-K next-token probabilities (step {step + 1})",
                "Token",
                "Probability",
            )

        # ================================================================
        # STEP 8 – DECODING
        # ================================================================
        with st.expander(
            f"Step 8 – Decoding (choose next token) [step {step + 1}]",
            expanded=True,
        ):
            greedy_id = torch.argmax(probs, dim=-1).item()
            greedy_tok = llm.tokenizer.convert_ids_to_tokens([greedy_id])[0]
            st.markdown(
                f"**Greedy next token:** id={greedy_id}, token=`{greedy_tok}`, "
                f"prob={float(probs[greedy_id]):.6f}"
            )

        # ---- Append token for next step ----
        greedy_id = torch.argmax(probs, dim=-1).item()
        current_text = current_text + llm.tokenizer.decode([greedy_id])

    # After all steps
    st.markdown("---")
    st.markdown("### Final result after step-by-step generation")
    st.write(f"**Final text:** `{current_text}`")



# =============================================================================
# Streamlit app
# =============================================================================


def main():
    st.set_page_config(
        page_title="GPT-2 LLM Introspection Lab",
        layout="wide",
    )

    st.title("GPT-2 LLM Introspection Lab")
    st.caption(
        "Visual X-ray of the 8-step transformer pipeline: "
        "tokens → embeddings → attention → FFN → logits → next token."
    )

    llm = load_llm()

    # Sidebar: configuration
    st.sidebar.header("Introspection settings")

    cfg = IntrospectionConfig(
        layers_to_show=st.sidebar.slider(
            "Layers to show", min_value=1, max_value=llm.n_layer, value=3
        ),
        heads_to_show=st.sidebar.slider(
            "Heads per layer", min_value=1, max_value=llm.n_head, value=4
        ),
        num_dims_to_show=st.sidebar.slider(
            "Vector dimensions to show", min_value=2, max_value=32, value=8
        ),
        top_keys_per_head=st.sidebar.slider(
            "Top attended positions per head", min_value=1, max_value=16, value=4
        ),
        topk_next_token=st.sidebar.slider(
            "Top-K next-token probabilities", min_value=5, max_value=50, value=10
        ),
    )

    st.sidebar.markdown("---")
    mode = st.sidebar.radio(
        "Mode",
        ["Single forward pass (show all 8 steps)", "Step-by-step generation (summary)"],
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Model: GPT-2 small\n"
        f"Layers: {llm.n_layer}, Heads: {llm.n_head}, Embedding dim: {llm.n_embd}"
    )

    # Main prompt input
    prompt = st.text_area("Enter your prompt:", value="The meaning of life is", height=100)

    if not prompt.strip():
        st.warning("Enter a prompt to start introspection.")
        return

    if mode.startswith("Single forward pass"):
        if st.button("Run single forward-pass introspection"):
            run_single_forward(llm, prompt, cfg)
    else:
        num_steps = st.number_input(
            "Number of tokens to generate (steps)",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
        )
        if st.button("Run step-by-step generation"):
            run_step_by_step(llm, prompt, cfg, num_steps=num_steps)


if __name__ == "__main__":
    main()
