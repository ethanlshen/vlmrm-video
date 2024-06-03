from typing import List, Optional, Tuple, overload

import torch
import torch.distributed as dist
import torch.nn as nn

from torchvision.models.video.s3d import S3D_Weights
from vlmrm.contrib.open_clip.transform import video_transform
from vlmrm.trainer.config import CLIPRewardConfig

from mmpt.models import MMPTModel

MMPT_PATH = "/home/ethans/classes/rl/project/fairseq/examples/MMPT"

class CLIPEmbed(nn.Module):
    def __init__(self, clip_model, video_encoder):
        super().__init__()
        self.clip_model = clip_model.cuda().eval()
        self.video_encoder = video_encoder.cuda().eval()
        self.transform = S3D_Weights.KINETICS400_V1.transforms(resize_size=(224, 224))
        self.target = None
        self.baseline = None
        self.max_video_len = 32  # From yaml
        
    @torch.inference_mode()
    def forward(self, x):
        if len(x.shape) == 6:
            bsz, sec, fps, x_dim, y_dim, rgb_dim = x.shape
            x = x.reshape(bsz, sec * fps, x_dim, y_dim, rgb_dim)
            x = x.permute(0, 1, 4, 2, 3)
        with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
            x = self.transform(x) # Normalize video
            x = x.permute(0, 2, 3, 4, 1) # Reverse permutation
            # torch.save(x.reshape(-1, 224, 224, 3), "frames_post.pt")
            # torch.save(255 * x.reshape(-1, 224, 224, 3), "frames_post_255.pt")
            x = x.reshape(bsz, sec, fps, x.shape[2], x.shape[3], x.shape[4])
            vfeats, vmasks = self.process_video(x)
            x = self.clip_model.forward_video(vfeats, 
                                              vmasks, 
                                              self.target[0], 
                                              self.target[1]) # Encode
        return x

    def process_video(self, video_frames):
        bsz = video_frames.size(0)
        assert bsz == 1, "only bsz=1 is supported now."
        seq_len = video_frames.size(1)
        video_frames = video_frames.view(-1, *video_frames.size()[2:])
        video_frames = video_frames.permute(0, 4, 1, 2, 3).to(torch.float16)
        video_frames = video_frames.cuda()
        vfeats = self.video_encoder(video_frames)
        vfeats = vfeats['video_embedding']
        vfeats = vfeats.view(bsz, seq_len, vfeats.size(-1))
        padding = torch.zeros(
            bsz, self.max_video_len - seq_len, vfeats.size(-1))
        vfeats = torch.cat([vfeats, padding.cuda()], dim=1)
        vmasks = torch.cat([
            torch.ones((bsz, seq_len), dtype=torch.bool),
            torch.zeros((bsz, self.max_video_len - seq_len), dtype=torch.bool)
            ],
            dim=1
        )
        return vfeats, vmasks

class CLIPReward(nn.Module):
    def __init__(
        self,
        *,
        model: CLIPEmbed,
        alpha: float,
        target_prompts: torch.Tensor,
        baseline_prompts: torch.Tensor,
    ) -> None:
        """CLIP Reward function that modifies the CLIP vector space by
        projecting all vectors onto the line spanned by the prompt and
        a baseline prompt. The alpha parameter controls the degree of
        projection. A value of 0.0 means that the reward function is
        equivalent to the CLIP reward function. A value of 1.0 means
        that the vector space is completely projected onto the line
        and becomes a 1D space. Any value in between is a linear
        interpolation between the two.

        Args:
            model (str): CLIP model.
            device (str): Device to use.
            alpha (float, optional): Coeefficient of projection.
            target_prompts (torch.Tensor): Tokenized prompts describing
                the target state.
            baseline_prompts (torch.Tensor): Tokenized prompts describing
                the baseline state.
        """
        super().__init__()
        self.embed_module = model
        # Embed text. We unpackage prompt to access caps and cmasks.
        # caps and cmasks
        self.target_prompts = target_prompts
        self.baseline_prompts = baseline_prompts
        self.embed_module.target = target_prompts
        self.embed_module.baseline = baseline_prompts
        # embedding prep
        target = self.embed_prompts(*target_prompts).mean(dim=0, keepdim=True)
        baseline = self.embed_prompts(*baseline_prompts).mean(dim=0, keepdim=True)      
        direction = target - baseline
        # Register them as buffers so they are automatically moved around.
        self.register_buffer("target", target)
        self.register_buffer("baseline", baseline)
        self.register_buffer("direction", direction)

        self.alpha = alpha
        projection = self.compute_projection(alpha) # Projection
        self.register_buffer("projection", projection)
    
    def compute_projection(self, alpha: float) -> torch.Tensor:
        projection = self.direction.T @ self.direction / torch.norm(self.direction) ** 2
        identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
        projection = alpha * projection + (1 - alpha) * identity
        return projection

    def update_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        self.projection = self.compute_projection(alpha)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = 1 - (torch.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2
        return y

    @staticmethod
    def tokenize_prompts(x: List[str]) -> torch.Tensor:
        """Tokenize a list of prompts."""
        # Tokenizer
        pass

    def embed_prompts(self, caps, cmasks) -> torch.Tensor:
        """Embed a list of prompts."""
        with torch.no_grad():
            x = self.embed_module.clip_model.forward_text(caps, cmasks).float()
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def embed_video(self, x):
        return self.embed_module.forward(x)


def load_reward_model(
    model_name, target_prompts, baseline_prompts, alpha, cache_dir: Optional[str] = None
):
    config_path = f"{MMPT_PATH}/projects/retri/videoclip/how2.yaml"
    print("config start " + config_path)
    mmpt, tokenizer, aligner = MMPTModel.from_pretrained(config=config_path,
                                                          checkpoint=f"{MMPT_PATH}/runs/retri/videoclip/checkpoint_best.pt")
    # Aligner
    print("aligning text")
    target_ctext = aligner._build_text_seq(
        tokenizer(target_prompts, add_special_tokens=False)["input_ids"][0]
    )
    baseline_ctext = aligner._build_text_seq(
        tokenizer(baseline_prompts, add_special_tokens=False)["input_ids"][0]
    )
    target_ctext = (target_ctext[0][None, :].cuda(), target_ctext[1][None, :].cuda())
    baseline_ctext = (baseline_ctext[0][None, :].cuda(), baseline_ctext[1][None, :].cuda())
    # MMPT.model is actual model with text, video encoding. We pass that into CLIPEmbed
    # as clip_model.
    model = CLIPEmbed(mmpt.model, mmpt.video_encoder)
    model = CLIPReward(
        model=model,
        alpha=alpha,
        target_prompts=target_ctext,
        baseline_prompts=baseline_ctext,
    )
    print("reward model loaded")
    return model.eval()


def load_reward_model_from_config(config: CLIPRewardConfig) -> CLIPReward:
    return load_reward_model(
        model_name=config.pretrained_model,
        target_prompts=config.target_prompts,
        baseline_prompts=config.baseline_prompts,
        alpha=config.alpha,
        cache_dir=config.cache_dir,
    )


def compute_rewards(
    model: CLIPEmbed,
    frames: torch.Tensor,
    batch_size: int,
    num_workers: int,
    worker_frames_tensor=None,
) -> torch.Tensor:
    assert frames.device == torch.device("cpu")
    assert batch_size % num_workers == 0
    n_samples = len(frames)
    print(f"frames {frames.shape}")
    rewards = torch.zeros(n_samples, device=torch.device("cpu"))
    model = model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            frames_batch = frames[i : i + batch_size]
            # Reshape to 5 second clips
            _, x_dim, y_dim, rgb_dim = frames_batch.shape
            # print(frames_batch)
            # torch.save(frames_batch, "frames.pt")
            frames_batch = frames_batch.reshape(1, 6, 30, x_dim, y_dim, rgb_dim)
            # Calc reward
            print(f"Computing reward for batch {i} with shape {frames_batch.shape}")
            rewards_batch = dist_worker_compute_reward(
                rank=0,
                reward_model=model,
                render_dim=frames.shape[1:],
                batch_size=batch_size // num_workers,
                num_workers=num_workers,
                frames=frames_batch,
                worker_frames_tensor=worker_frames_tensor,
            )
            rewards_batch = rewards_batch.cpu()
            rewards[i : i + batch_size] = rewards_batch 
    return rewards


@overload
def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: torch.Tensor,
) -> torch.Tensor:
    ...


@overload
def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: None = None,
) -> None:
    ...


def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames=None,
    worker_frames_tensor=None,
) -> Optional[torch.Tensor]:
    if rank == 0:
        t = frames.shape
        if frames is None:
            raise ValueError("Must pass render result on rank=0")
        # if len(frames) != num_workers * batch_size:
        #     print(num_workers)
        #     print(batch_size)
        #     print(len(frames))
        #     raise ValueError("Must pass render result with correct batch size")
        if t[0] * t[1] * t[2] != num_workers * batch_size:
            raise ValueError("Must pass render result with correct batch size")
        scatter_list = [t.cuda(rank) for t in torch.chunk(frames, num_workers, dim=0)]
    else:
        scatter_list = []

    worker_frames = worker_frames_tensor if worker_frames_tensor is not None else torch.zeros((batch_size, *render_dim), dtype=torch.uint8).cuda(rank)
    # dist.scatter(worker_frames, scatter_list=scatter_list, src=0) 
    with torch.no_grad():
        
        embeddings = reward_model.embed_module(frames)
        rewards = reward_model(embeddings)

    def zero_t():
        return torch.zeros_like(rewards)

    recv_rewards = [zero_t() for _ in range(num_workers)] if rank == 0 else []
    dist.gather(rewards, gather_list=recv_rewards, dst=0)

    if rank == 0:
        return torch.cat(recv_rewards, dim=0).cuda(rank)
