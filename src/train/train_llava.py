"""Stage 4: LoRA LLaVA Fine-tuning (4-GPU DDP).

Freeze TRACE segmentation branch, TemporalEncoder, ATF.
Train: LoRA adapters on Vicuna-7B + custom projection MLP.
Loss: Causal LM cross-entropy (next-token prediction).
10 epochs, lora_lr=2e-4, projection_lr=1e-4, batch_size=4, grad_accum=8.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ThermalNarrationDataset
from src.models.atf import AsymmetricThermalFusion
from src.models.llava_lora import TRACELLaVA
from src.models.trace_model import TRACE
from src.models.temporal_encoder import TemporalEncoder
from src.utils.config import LLaVAConfig
from src.utils.trainer import (CheckpointManager, ETATracker, MetricsLogger,
                                cleanup_ddp, get_cosine_schedule_with_warmup,
                                get_sampler, is_main_process, print_header,
                                set_seed, setup_ddp, unwrap_model,
                                wrap_model_ddp)


@torch.no_grad()
def generate_and_evaluate(llava, seg_model, atf_module, temporal_model,
                          val_ds, device, max_new_tokens=64, eval_samples=128):
    """Generate descriptions for a subset of val samples and compute
    BERTScore (F1), ROUGE-L, and BLEU-4.
    """
    raw = unwrap_model(llava)
    raw.eval()

    references, hypotheses = [], []
    n = min(eval_samples, len(val_ds))
    indices = list(range(n))

    for idx in indices:
        sample = val_ds[idx]
        overlay = sample["overlay"].unsqueeze(0).to(device)
        mask    = sample["mask"].unsqueeze(0).to(device)
        ref_text = sample["description"]

        with autocast(dtype=torch.bfloat16):
            seg_out     = seg_model(overlay, binary_mask=mask)
            stage4_feat = seg_out["stage4_features"]
            bg_overlay  = overlay * (1.0 - mask)
            atf_out     = atf_module(mask, stage4_feat, bg_overlay)
            clip        = overlay.unsqueeze(1).repeat(1, 16, 1, 1, 1)
            temp_out    = temporal_model(clip)
            gen_ids     = raw.generate(
                atf_out["fused"], temp_out["temporal_embedding"],
                max_new_tokens=max_new_tokens,
            )
        hyp = raw.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        hypotheses.append(hyp)
        references.append(ref_text)

    # BERTScore F1
    P, R, F1 = bert_score_fn(hypotheses, references,
                              lang="en", device=str(device), verbose=False)
    bert_f1 = F1.mean().item()

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(scorer.score(r, h)["rougeL"].fmeasure
                  for r, h in zip(references, hypotheses)) / len(references)

    # BLEU-4
    smooth = SmoothingFunction().method1
    refs_tok  = [[r.split()] for r in references]
    hyps_tok  = [h.split() for h in hypotheses]
    bleu4 = corpus_bleu(refs_tok, hyps_tok,
                        smoothing_function=smooth)

    return {
        "val/BERTScore_F1": bert_f1,
        "val/ROUGE_L":      rouge_l,
        "val/BLEU_4":       bleu4,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seg_checkpoint", type=str, default=None)
    parser.add_argument("--temporal_checkpoint", type=str, default=None)
    parser.add_argument("--fusion_checkpoint", type=str, default=None)
    args = parser.parse_args()

    rank, world_size, device = setup_ddp()
    config = LLaVAConfig()
    set_seed(config.seed + rank)
    print_header("Stage 4: LLaVA LoRA Fine-tuning", config)
    if is_main_process():
        print(f"World size: {world_size} GPUs")

    # ---- Load frozen upstream models ----
    seg_model = TRACE(num_seg_classes=1, decode_dim=256, use_aux_mask=True).to(device)
    seg_ckpt = args.seg_checkpoint or config.segmentation_checkpoint
    if seg_ckpt:
        ckpt = torch.load(seg_ckpt, map_location=device, weights_only=False)
        seg_model.load_state_dict(ckpt["model_state_dict"])
        if is_main_process():
            print(f"Loaded seg checkpoint: {seg_ckpt}")
    seg_model.eval()
    for p in seg_model.parameters():
        p.requires_grad = False

    temporal_model = TemporalEncoder(output_dim=256).to(device)
    temp_ckpt = args.temporal_checkpoint or config.temporal_checkpoint
    if temp_ckpt:
        ckpt = torch.load(temp_ckpt, map_location=device, weights_only=False)
        temporal_model.load_state_dict(ckpt["model_state_dict"])
        if is_main_process():
            print(f"Loaded temporal checkpoint: {temp_ckpt}")
    temporal_model.eval()
    for p in temporal_model.parameters():
        p.requires_grad = False

    atf_module = AsymmetricThermalFusion(feature_dim=256, stage4_channels=512).to(device)
    fusion_ckpt = args.fusion_checkpoint or config.fusion_checkpoint
    if fusion_ckpt:
        ckpt = torch.load(fusion_ckpt, map_location=device, weights_only=False)
        atf_state = {k.replace("atf.", ""): v for k, v in ckpt["model_state_dict"].items()
                     if k.startswith("atf.")}
        atf_module.load_state_dict(atf_state)
        if is_main_process():
            print(f"Loaded ATF checkpoint: {fusion_ckpt}")
    atf_module.eval()
    for p in atf_module.parameters():
        p.requires_grad = False

    # ---- LLaVA model ----
    llava = TRACELLaVA(
        llava_model_name=config.llava_model_name,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_targets=config.lora_targets,
        num_visual_tokens=config.num_visual_tokens,
    )
    llava.setup_model()
    llava = wrap_model_ddp(llava, device)

    # Data
    raw_llava = unwrap_model(llava)
    train_ds = ThermalNarrationDataset(
        img_size=(224, 224), tokenizer=raw_llava.tokenizer,
        max_text_len=config.max_text_len)
    val_ds = ThermalNarrationDataset(
        img_size=(224, 224), tokenizer=raw_llava.tokenizer,
        max_text_len=config.max_text_len)
    train_sampler = get_sampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=config.num_workers, pin_memory=config.pin_memory,
                              drop_last=True)
    if is_main_process():
        print(f"Train: {len(train_ds)} narration pairs | Val (gen eval): {len(val_ds)}")

    # Optimizer: separate LRs for LoRA and projection
    lora_params = [p for n, p in llava.named_parameters()
                   if "lora" in n.lower() and p.requires_grad]
    proj_params = [p for n, p in llava.named_parameters()
                   if "projection" in n.lower() and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": config.lora_lr},
        {"params": proj_params, "lr": config.projection_lr},
    ], weight_decay=config.weight_decay)

    scaler = GradScaler()
    total_steps = config.epochs * (len(train_loader) // config.gradient_accumulation)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, total_steps)
    eta = ETATracker(total_steps)

    ckpt_mgr = CheckpointManager(config.checkpoint_dir, config.stage_name,
                                  keep_last_n=config.keep_last_n_checkpoints)
    logger = MetricsLogger(config.log_dir, config.stage_name,
                           use_wandb=config.use_wandb,
                           wandb_project=config.wandb_project)

    global_step = 0
    start_epoch = 0

    if args.resume or args.resume_from:
        path = args.resume_from
        info = ckpt_mgr.load(path, llava, optimizer, scheduler, scaler) if path else ckpt_mgr.load_latest(llava, optimizer, scheduler, scaler)
        if info:
            start_epoch = info["epoch"] + 1
            global_step = info["global_step"]

    for epoch in range(start_epoch, config.epochs):
        llava.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        total_loss = 0.0

        for i, batch in enumerate(train_loader):
            overlay = batch["overlay"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = input_ids.clone()

            with autocast(dtype=torch.bfloat16):
                with torch.no_grad():
                    seg_out = seg_model(overlay, binary_mask=mask)
                    stage4_feat = seg_out["stage4_features"]
                    bg_overlay = overlay * (1.0 - mask)
                    atf_out = atf_module(mask, stage4_feat, bg_overlay)
                    clip = overlay.unsqueeze(1).repeat(1, 16, 1, 1, 1)
                    temp_out = temporal_model(clip)

                out = llava(atf_out["fused"], temp_out["temporal_embedding"],
                            input_ids, attention_mask, labels)
                loss = out["loss"] / config.gradient_accumulation

            scaler.scale(loss).backward()

            if (i + 1) % config.gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

                if global_step % config.log_every_n_steps == 0 and is_main_process():
                    eta_str, elapsed = eta.step(global_step)
                    lr = optimizer.param_groups[0]["lr"]
                    print(f"  [E{epoch:02d} S{global_step:05d}] "
                          f"loss={loss.item() * config.gradient_accumulation:.4f} "
                          f"lr={lr:.2e} ETA={eta_str}")
                    logger.log({"train/loss": loss.item() * config.gradient_accumulation,
                                 "train/lr": lr}, step=global_step)

            total_loss += loss.item() * config.gradient_accumulation

        avg_loss = total_loss / len(train_loader)
        if is_main_process():
            print(f"  [Epoch {epoch:02d}] avg_loss={avg_loss:.4f}")
            logger.log({"train/epoch_loss": avg_loss}, step=global_step)

            # Generation quality evaluation (BLEU, ROUGE-L, BERTScore)
            gen_metrics = generate_and_evaluate(
                llava, seg_model, atf_module, temporal_model,
                val_ds, device)
            print(f"           BERTScore_F1={gen_metrics['val/BERTScore_F1']:.4f} "
                  f"ROUGE_L={gen_metrics['val/ROUGE_L']:.4f} "
                  f"BLEU_4={gen_metrics['val/BLEU_4']:.4f}")
            logger.log({**{"train/epoch_loss": avg_loss}, **gen_metrics}, step=global_step)

            if (epoch + 1) % config.save_every_n_epochs == 0:
                ckpt_mgr.save(llava, optimizer, scheduler, scaler, epoch,
                              global_step, {"train/loss": avg_loss, **gen_metrics})

    logger.finish()
    cleanup_ddp()
    if is_main_process():
        print("\nStage 4 complete.")


if __name__ == "__main__":
    main()
