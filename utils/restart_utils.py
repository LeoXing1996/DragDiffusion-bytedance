import torch
from diffusers import DDIMScheduler
from tqdm import tqdm


@torch.no_grad()
def restart(x_prev_updated, t_min, t_max, model, args):
    """This function conduct *SINGLE STEP* restart strategy.
    """
    strategy = args.restart_strategy
    scheduler: DDIMScheduler = model.scheduler

    if strategy == 'ddim_inverse':
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
def restart_from_t(input_latent, t_min, t_max, model, text_embeddings, args):
    """
      +- drag (t_min = t_max + drag_step) -+
      |                                    |
     z_t_max -> z_t_max - 1 -> ... -> z_t_min -> z_t_min - 1-> ... -> z_0
      |                                                               |
      +----------------------------- Restart -------------------------+

    t_min: the timestep to restart from
    t_max: the timestep to end at

    """
    strategy = args.restart_strategy
    scheduler: DDIMScheduler = model.scheduler
    t_max_idx = torch.nonzero(scheduler.timesteps == t_max)[0][0]
    t_min_idx = torch.nonzero(scheduler.timesteps == t_min)[0][0]

    # 1. denoising to z_0
    denoising_timesteps = scheduler.timesteps[t_min_idx+1:]
    for t in tqdm(denoising_timesteps):
        unet_output = model.unet(
            input_latent,
            t,
            encoder_hidden_states=text_embeddings,
            return_intermediates=False)
        input_latent, _ = model.step(unet_output, t, input_latent)

    # 2. restart to z_t_max
    inverse_timesteps = scheduler.timesteps[t_max_idx:].flip(0)
    if strategy == 'ddim_inverse':
        desc = f'Restart - DDIM: {inverse_timesteps}'
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
