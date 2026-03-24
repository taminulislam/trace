"""Module 3.5: LLaVA-1.5 7B with LoRA.

Vision encoder: CLIP ViT-L/14 -> FROZEN throughout all training
Language model: Vicuna-7B with LoRA (rank=16, alpha=32, targets: q_proj, v_proj, k_proj)
Custom projection MLP: concat([ATF_fused(256), temporal_emb(256)]) -> D=4096 visual tokens
(This replaces LLaVA's standard CLIP->MLP projection -- key structural novelty)
Output: class label token + natural language narration tokens
"""

import torch
import torch.nn as nn


class TRACEProjectionMLP(nn.Module):
    """Custom projection that replaces LLaVA's standard CLIP->MLP projection.

    Takes the concatenation of ATF fused features (D=256) and temporal
    embeddings (D=256) -> total 512-dim input, and projects to the LLM's
    hidden dimension (4096 for Vicuna-7B) as visual tokens.

    This is the key structural novelty of TRACE: instead of using
    CLIP visual features, we inject domain-specific thermal+temporal features.

    Args:
        input_dim: ATF_dim + temporal_dim (default 512 = 256+256)
        hidden_dim: intermediate MLP dimension
        output_dim: LLM hidden size (4096 for Vicuna-7B)
        num_tokens: number of visual tokens to produce (default 32)
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024,
                 output_dim: int = 4096, num_tokens: int = 32):
        super().__init__()
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim * num_tokens),
        )

    def forward(self, atf_fused: torch.Tensor,
                temporal_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atf_fused: (B, 256) from ATF module
            temporal_emb: (B, 256) from temporal encoder

        Returns:
            (B, num_tokens, output_dim) visual token embeddings for LLM
        """
        x = torch.cat([atf_fused, temporal_emb], dim=-1)  # (B, 512)
        x = self.proj(x)  # (B, output_dim * num_tokens)
        B = x.shape[0]
        return x.view(B, self.num_tokens, -1)  # (B, num_tokens, 4096)


class TRACELLaVA(nn.Module):
    """TRACE LLaVA integration module.

    Wraps LLaVA-1.5 7B with:
    - Custom projection MLP (replacing standard CLIP->MLP)
    - LoRA adapters on Vicuna-7B (rank=16, alpha=32, targets: q_proj, v_proj, k_proj)
    - CLIP ViT-L/14 vision encoder (FROZEN, kept for compatibility but not used
      for primary features)

    This class handles model construction and LoRA setup.
    The actual forward pass follows the LLaVA pipeline but substitutes
    our custom visual tokens.

    Args:
        llava_model_name: HuggingFace model name for LLaVA-1.5
        lora_rank: LoRA rank (default 16)
        lora_alpha: LoRA alpha (default 32)
        lora_targets: list of module names to apply LoRA
        atf_dim: ATF fused feature dimension (default 256)
        temporal_dim: temporal embedding dimension (default 256)
        num_visual_tokens: number of visual tokens from projection (default 32)
    """

    def __init__(
        self,
        llava_model_name: str = "liuhaotian/llava-v1.5-7b",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_targets: list = None,
        atf_dim: int = 256,
        temporal_dim: int = 256,
        num_visual_tokens: int = 32,
    ):
        super().__init__()
        if lora_targets is None:
            lora_targets = ["q_proj", "v_proj", "k_proj"]

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_targets = lora_targets
        self.llava_model_name = llava_model_name

        # Custom projection MLP
        self.projection = TRACEProjectionMLP(
            input_dim=atf_dim + temporal_dim,
            output_dim=4096,
            num_tokens=num_visual_tokens,
        )

        # LLaVA model and LoRA will be initialized via setup_model()
        self.llava_model = None
        self.tokenizer = None

    def setup_model(self):
        """Load LLaVA model and apply LoRA adapters.

        Call this after __init__ when running on GPU with sufficient memory.
        Separated from __init__ to allow model definition without loading 7B weights.
        """
        from transformers import AutoTokenizer
        from llava.model import LlavaLlamaForCausalLM
        from peft import LoraConfig, get_peft_model

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llava_model_name, use_fast=False
        )
        self.llava_model = LlavaLlamaForCausalLM.from_pretrained(
            self.llava_model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        # Freeze CLIP vision encoder
        if hasattr(self.llava_model, "model") and hasattr(self.llava_model.model, "vision_tower"):
            for param in self.llava_model.model.vision_tower.parameters():
                param.requires_grad = False

        # Apply LoRA to language model
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_targets,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llava_model = get_peft_model(self.llava_model, lora_config)
        self.llava_model.print_trainable_parameters()

    def forward(self, atf_fused: torch.Tensor, temporal_emb: torch.Tensor,
                input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None) -> dict:
        """
        Args:
            atf_fused: (B, 256) from ATF module
            temporal_emb: (B, 256) from temporal encoder
            input_ids: (B, seq_len) tokenized text input
            attention_mask: (B, seq_len) attention mask
            labels: (B, seq_len) target token ids for causal LM loss

        Returns:
            dict with "loss" and "logits"
        """
        # Get custom visual tokens from our projection
        visual_tokens = self.projection(atf_fused, temporal_emb)  # (B, num_tokens, 4096)

        # Get text embeddings from LLM
        text_embeds = self.llava_model.get_model().embed_tokens(input_ids)

        # Prepend visual tokens to text embeddings
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        # Adjust attention mask and labels for prepended visual tokens
        B, num_vis = visual_tokens.shape[:2]
        vis_attn = torch.ones(B, num_vis, device=input_ids.device, dtype=attention_mask.dtype)
        if attention_mask is not None:
            attention_mask = torch.cat([vis_attn, attention_mask], dim=1)

        if labels is not None:
            # Don't compute loss on visual token positions
            vis_labels = torch.full((B, num_vis), -100, device=labels.device, dtype=labels.dtype)
            labels = torch.cat([vis_labels, labels], dim=1)

        outputs = self.llava_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def generate(self, atf_fused: torch.Tensor, temporal_emb: torch.Tensor,
                 prompt_ids: torch.Tensor, max_new_tokens: int = 128,
                 **kwargs) -> torch.Tensor:
        """Generate narration text given visual features and a prompt.

        Args:
            atf_fused: (B, 256)
            temporal_emb: (B, 256)
            prompt_ids: (B, seq_len) tokenized prompt
            max_new_tokens: maximum generation length

        Returns:
            (B, gen_len) generated token ids
        """
        visual_tokens = self.projection(atf_fused, temporal_emb)
        prompt_embeds = self.llava_model.get_model().embed_tokens(prompt_ids)
        inputs_embeds = torch.cat([visual_tokens, prompt_embeds], dim=1)

        return self.llava_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
