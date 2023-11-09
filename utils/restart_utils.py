import torch
from diffusers import DDIMScheduler
from tqdm import tqdm
from .attn_utils import register_attention_editor_diffusers


@torch.no_grad()
def restart(x_prev_updated, t_min, t_max, model, args):
    """This function conduct *SINGLE STEP* restart strategy.
    """
    strategy = args.restart_strategy
    scheduler: DDIMScheduler = model.scheduler

    if strategy == 'ddim_inversion':
        # TODO:
        pass

    elif strategy == 'ddpm':
        # t_start = scheduler.timesteps[timestep]
        # t_end = scheduler.timesteps[timestep + 1]
        # alpha_cumprod = scheduler.alphas_cumprod[timestep]
        noise = torch.randn_like(x_prev_updated)
        beta_t = scheduler.betas[t_min]
        x_restart = x_prev_updated * torch.sqrt(1 - beta_t) + beta_t * noise

    elif strategy == 'restart':
        # TODO:
        # rho = 7
        # t_max = timestep

        # prev_timestep = min(timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 999)
        # t_min = min(timestep - scheduler.config.num)
        pass

    else:
        raise ValueError(f'Invalid restart strategy {strategy}.')

    return x_restart


@torch.no_grad()
def restart_from_t(input_latent, t_min, t_max, model, text_embeddings, args,
                   editor=None):
    """
      +--------------- drag ---------------+
      |                                    |
     z_t_max -> z_t_max - 1 -> ... -> z_t_min -> z_t_min - 1-> ... -> z_0
      |                                                               |
      +----------------------------- Restart -------------------------+

    t_min = t_max + drag_step - 1
    t_min: the timestep to restart from
    t_max: the timestep to end at

    """
    strategy = args.restart_strategy
    scheduler: DDIMScheduler = model.scheduler
    t_max_idx = torch.nonzero(scheduler.timesteps == t_max)[0][0]
    t_min_idx = torch.nonzero(scheduler.timesteps == t_min)[0][0]

    if strategy == 'add_noise_to_t':
        """
        +------------ 1. drag ---------------+
        |                                    |
        z_t_max -> z_t_max - 1 -> ... -> z_t_min -> z_t_min - 1-> ... -> z_0
        |                                    |                           |
        +---- 2. Add Noise (Restart) --------+                           |
        |                                                                |
        +--------------------- 3. Denoising wo/Guidance -----------------+
        """
        alpha_cumprod_min = scheduler.alphas_cumprod[t_min]
        alpha_cumprod_max = scheduler.alphas_cumprod[t_max]

        alpha_cumprod = alpha_cumprod_max / alpha_cumprod_min
        factor_1 = alpha_cumprod ** 0.5
        factor_2 = (1 - alpha_cumprod) ** 0.5
        noise = torch.randn_like(input_latent)
        input_latent = factor_1 * input_latent + factor_2 * noise

        return input_latent

    if args.restart_with_masactrl:
        # TODO: finish this function.
        assert editor is not None, (
            'Editor must be passed since we want to apply MasaCtrl in Restart.')
        register_attention_editor_diffusers(model, editor, 'lora_attn_proc')

    # 1. denoising to z_0
    denoising_timesteps = scheduler.timesteps[t_min_idx:]
    # TODO: should we add masactrl here?
    for t in tqdm(denoising_timesteps):
        unet_output = model.unet(
            input_latent,
            t,
            encoder_hidden_states=text_embeddings,
            return_intermediates=False)
        input_latent, _ = model.step(unet_output, t, input_latent)

    # 2. restart to z_t_max
    inverse_timesteps = scheduler.timesteps[t_max_idx:].flip(0)
    if strategy == 'ddim_inversion':
        print(f'Restart - DDIM: {inverse_timesteps}')
        desc = 'Restart - DDIM Inversion'
        for t in tqdm(inverse_timesteps, desc=desc):
            unet_output = model.unet(
                input_latent,
                t,
                encoder_hidden_states=text_embeddings,
                return_intermediates=False)

            input_latent, _ = model.inv_step(unet_output, t, input_latent)

    elif strategy == 'add_noise':
        print('Restart - Add Noise')
        noise = torch.randn_like(input_latent)
        input_latent = scheduler.add_noise(input_latent, noise, t_max)

    return input_latent
