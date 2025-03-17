import ultimate_rvc.rvc.infer.logger_config
import logging
logger = logging.getLogger(__name__)

from typing import Optional
import torch
from ultimate_rvc.rvc.lib.algorithm.commons import rand_slice_segments, slice_segments
from ultimate_rvc.rvc.lib.algorithm.encoders import PosteriorEncoder, TextEncoder
from ultimate_rvc.rvc.lib.algorithm.generators.hifigan import HiFiGANGenerator
from ultimate_rvc.rvc.lib.algorithm.generators.hifigan_mrf import HiFiGANMRFGenerator
from ultimate_rvc.rvc.lib.algorithm.generators.hifigan_nsf import HiFiGANNSFGenerator
from ultimate_rvc.rvc.lib.algorithm.generators.refinegan import RefineGANGenerator
from ultimate_rvc.rvc.lib.algorithm.residuals import ResidualCouplingBlock
import numpy as np


def debug_tensor(name, tensor):
    """
    Logs detailed statistics for a tensor.
    """
    if tensor is None:
        logger.debug("%s: None", name)
        return
    try:
        t_np = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        logger.debug("%s: shape=%s, min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                     name, tensor.shape, np.min(t_np), np.max(t_np),
                     np.mean(t_np), np.std(t_np))
    except Exception as e:
        logger.debug("%s: Could not compute statistics due to: %s", name, e)

class Synthesizer(torch.nn.Module):
    """
    Base Synthesizer model with detailed debugging logs.
    """
    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        use_f0: bool,
        text_enc_hidden_dim: int = 768,
        vocoder: str = "HiFi-GAN",
        randomized: bool = True,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        # Log all initialization parameters.
        logger.debug("Initializing Synthesizer with parameters:")
        logger.debug("spec_channels: %d", spec_channels)
        logger.debug("segment_size: %d", segment_size)
        logger.debug("inter_channels: %d", inter_channels)
        logger.debug("hidden_channels: %d", hidden_channels)
        logger.debug("filter_channels: %d", filter_channels)
        logger.debug("n_heads: %d", n_heads)
        logger.debug("n_layers: %d", n_layers)
        logger.debug("kernel_size: %d", kernel_size)
        logger.debug("p_dropout: %f", p_dropout)
        logger.debug("resblock: %s", resblock)
        logger.debug("resblock_kernel_sizes: %s", resblock_kernel_sizes)
        logger.debug("resblock_dilation_sizes: %s", resblock_dilation_sizes)
        logger.debug("upsample_rates: %s", upsample_rates)
        logger.debug("upsample_initial_channel: %d", upsample_initial_channel)
        logger.debug("upsample_kernel_sizes: %s", upsample_kernel_sizes)
        logger.debug("spk_embed_dim: %d", spk_embed_dim)
        logger.debug("gin_channels: %d", gin_channels)
        logger.debug("sr: %d", sr)
        logger.debug("use_f0: %s", use_f0)
        logger.debug("text_enc_hidden_dim: %d", text_enc_hidden_dim)
        logger.debug("vocoder: %s", vocoder)
        logger.debug("randomized: %s, checkpointing: %s", randomized, checkpointing)

        self.segment_size = segment_size
        self.use_f0 = use_f0
        self.randomized = randomized

        # Initialize the text encoder (posterior encoder for phoneme features).
        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            text_enc_hidden_dim,
            f0=use_f0,
        )
        logger.info("TextEncoder (enc_p) initialized.")

        logger.info("Using %s vocoder", vocoder)
        # Initialize the decoder based on the vocoder type and F0 usage.
        if use_f0:
            if vocoder == "MRF HiFi-GAN":
                self.dec = HiFiGANMRFGenerator(
                    in_channel=inter_channels,
                    upsample_initial_channel=upsample_initial_channel,
                    upsample_rates=upsample_rates,
                    upsample_kernel_sizes=upsample_kernel_sizes,
                    resblock_kernel_sizes=resblock_kernel_sizes,
                    resblock_dilations=resblock_dilation_sizes,
                    gin_channels=gin_channels,
                    sample_rate=sr,
                    harmonic_num=8,
                    checkpointing=checkpointing,
                )
                logger.info("Initialized HiFiGANMRFGenerator for F0 usage.")
            elif vocoder == "RefineGAN":
                self.dec = RefineGANGenerator(
                    sample_rate=sr,
                    downsample_rates=upsample_rates[::-1],
                    upsample_rates=upsample_rates,
                    start_channels=16,
                    num_mels=inter_channels,
                    checkpointing=checkpointing,
                )
                logger.info("Initialized RefineGANGenerator for F0 usage.")
            else:
                self.dec = HiFiGANNSFGenerator(
                    inter_channels,
                    resblock_kernel_sizes,
                    resblock_dilation_sizes,
                    upsample_rates,
                    upsample_initial_channel,
                    upsample_kernel_sizes,
                    gin_channels=gin_channels,
                    sr=sr,
                    checkpointing=checkpointing,
                )
                logger.info("Initialized HiFiGANNSFGenerator for F0 usage.")
        elif vocoder == "MRF HiFi-GAN":
            logger.error("MRF HiFi-GAN does not support training without pitch guidance.")
            self.dec = None
        elif vocoder == "RefineGAN":
            logger.error("RefineGAN does not support training without pitch guidance.")
            self.dec = None
        else:
            self.dec = HiFiGANGenerator(
                inter_channels,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
                checkpointing=checkpointing,
            )
            logger.info("Initialized HiFiGANGenerator for non-F0 training.")

        # Initialize the posterior encoder for mel spectrograms.
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,  # fixed parameter: number of layers or kernel size for PosteriorEncoder?
            1,
            16,
            gin_channels=gin_channels,
        )
        logger.info("PosteriorEncoder (enc_q) initialized.")

        # Initialize the flow network for residual coupling.
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            3,
            gin_channels=gin_channels,
        )
        logger.info("ResidualCouplingBlock (flow) initialized.")

        # Initialize speaker embedding module.
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)
        logger.info("Speaker embedding (emb_g) initialized with %d embeddings.", spk_embed_dim)

    def _remove_weight_norm_from(self, module):
        for hook in module._forward_pre_hooks.values():
            if getattr(hook, "__class__", None).__name__ == "WeightNorm":
                torch.nn.utils.remove_weight_norm(module)
                logger.debug("Removed weight norm from module: %s", module)

    def remove_weight_norm(self):
        for module in [self.dec, self.flow, self.enc_q]:
            self._remove_weight_norm_from(module)
        logger.info("Weight norm removed from dec, flow, and enc_q modules.")

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self

    def forward(self, phone: torch.Tensor, phone_lengths: torch.Tensor,
                pitch: torch.Tensor | None = None, pitchf: torch.Tensor | None = None,
                y: torch.Tensor | None = None, y_lengths: torch.Tensor | None = None,
                ds: torch.Tensor | None = None):
        logger.debug("Synthesizer.forward: phone shape: %s, phone_lengths: %s", phone.shape, phone_lengths)
        
        if ds is not None:
            g = self.emb_g(ds).unsqueeze(-1)
            debug_tensor("Speaker embedding (g)", g)
        else:
            g = None

        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        debug_tensor("enc_p: m_p", m_p)
        debug_tensor("enc_p: logs_p", logs_p)
        debug_tensor("enc_p: x_mask", x_mask)
        
        if y is not None:
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            debug_tensor("enc_q: z", z)
            debug_tensor("enc_q: m_q", m_q)
            debug_tensor("enc_q: logs_q", logs_q)
            debug_tensor("enc_q: y_mask", y_mask)
            
            z_p = self.flow(z, y_mask, g=g)
            debug_tensor("Flow output (z_p)", z_p)
            
            if self.randomized:
                z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
                debug_tensor("Random slice (z_slice)", z_slice)
                if self.use_f0:
                    pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                    debug_tensor("Sliced pitchf", pitchf)
                    o = self.dec(z_slice, pitchf, g=g)
                else:
                    o = self.dec(z_slice, g=g)
                debug_tensor("Decoder output (o)", o)
                return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
            else:
                if self.use_f0:
                    o = self.dec(z, pitchf, g=g)
                else:
                    o = self.dec(z, g=g)
                debug_tensor("Decoder output (o)", o)
                return o, None, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
        else:
            logger.debug("No mel spectrogram (y) provided; returning only encoder outputs.")
            return None, None, x_mask, None, (None, None, m_p, logs_p, None, None)

    @torch.jit.export
    def infer(self, phone: torch.Tensor, phone_lengths: torch.Tensor,
            pitch: torch.Tensor | None = None, nsff0: torch.Tensor | None = None,
            sid: torch.Tensor = None, rate: torch.Tensor | None = None):
        logger.debug("Synthesizer.infer: phone shape: %s, phone_lengths: %s", phone.shape, phone_lengths)
        g = self.emb_g(sid).unsqueeze(-1)
        debug_tensor("infer: Speaker embedding (g)", g)
        
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        debug_tensor("infer: m_p", m_p)
        debug_tensor("infer: logs_p", logs_p)
        debug_tensor("infer: x_mask", x_mask)
        
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        debug_tensor("infer: z_p", z_p)
        
        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p, x_mask = z_p[:, :, head:], x_mask[:, :, head:]
            if self.use_f0 and nsff0 is not None:
                nsff0 = nsff0[:, head:]
            logger.debug("After rate adjustment: z_p shape: %s, x_mask shape: %s", z_p.shape, x_mask.shape)
        
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        debug_tensor("infer: Flow reverse output (z)", z)
        
        if self.use_f0:
            o = self.dec(z * x_mask, nsff0, g=g)
        else:
            o = self.dec(z * x_mask, g=g)
        debug_tensor("infer: Decoder output (o)", o)
        
        return o, x_mask, (z, z_p, m_p, logs_p)
 
