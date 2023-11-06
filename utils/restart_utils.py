import torch
from diffusers import DDIMScheduler

@torch.no_grad()
def restart(x_prev_updated, timestep, model, args):
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
        beta_t = scheduler.betas[timestep]
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
