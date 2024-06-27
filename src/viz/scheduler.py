import src.config as conf
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.scheduler import create_noise_schedule

def plot_alphas_schedules(num_steps):
    schedules = ['cosine', 'linear']
    colors = ['b', 'r']  # Blue for cosine, Red for linear
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Alpha Schedules Comparison')

    for schedule_type, color in zip(schedules, colors):
        noise_schedule = create_noise_schedule(num_steps, schedule_type)
        
        # Plot alphas
        ax1.plot(noise_schedule['alphas'], color=color, label=f'{schedule_type.capitalize()}')
        
        # Plot cumulative alphas
        ax2.plot(noise_schedule['alphas_cumprod'], color=color, label=f'{schedule_type.capitalize()}')

    # Customize left plot (Alphas)
    ax1.set_title('Alphas')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Alpha')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Customize right plot (Cumulative Alphas)
    ax2.set_title('Cumulative Alphas')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative Alpha')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("/diffusion/results/scheduler/alphas_comparison.png")
    plt.close()

# Call the function to plot
plot_alphas_schedules(conf.T)

def plot_betas_comparison():
    num_steps = conf.T
    
    # Cosine schedule
    betas_t_cosine = create_noise_schedule(num_steps, "cosine")
    
    # Linear schedule
    betas_t_linear = create_noise_schedule(num_steps, "linear")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(betas_t_cosine["betas"] / betas_t_cosine["betas"][0], label='Cosine Schedule')
    plt.plot(betas_t_linear["betas"] / betas_t_linear["betas"][0], label='Linear Schedule')
    
    plt.title('Comparison of Cosine and Linear Betas Schedules')
    plt.xlabel('Timestep')
    plt.ylabel('Betas')
    plt.legend()
    plt.grid(True)
    
    plt.savefig("/diffusion/results/scheduler/betas.png")
    plt.close()
    
    
plot_betas_comparison()