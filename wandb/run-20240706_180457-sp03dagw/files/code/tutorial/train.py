from sindy_rl.dynamics import EnsembleSINDyDynamicsModel
from Ambiente_SOMN.Somn import Somn
import wandb
from sindy_rl.env import rollout_env
from sindy_rl.registry import DMCEnvWrapper
from sindy_rl.policy import RandomPolicy

dyna_config = {
    'callbacks': 'project_cartpole',
    'dt': 1,
    'discrete': True,
    
    # Optimizer config 
    'optimizer': {
      'base_optimizer': {
        'name': 'STLSQ',
        'kwargs': {
          'alpha': 5.0e-5,
          'threshold': 7.0e-3,
            },
      },
      # Ensemble Optimization config
      'ensemble': {
        'bagging': True,
        'library_ensemble': True,
        'n_models': 20,
      },
    },
    # Dictionary/Libary Config
    'feature_library': {
      'name': 'affine', # use affine functions
      'kwargs': {
        'poly_deg': 2,
        'n_state': 5 ,
        'n_control': 1,
        'poly_int': True,
        'tensor': True,
      }
    }
}

dyn_model = EnsembleSINDyDynamicsModel(dyna_config)

env = Somn(
                Y=10,
                M=10,
                N=10,
                MAXDO=100,
                MAXAM=2,
                MAXPR=2,
                MAXPE=10,
                MAXFT=5,
                MAXMT=3,
                MAXTI=2,
                MAXEU = 5, 
                atraso=-1,
                objetivo=0
            )


# Initialize a new wandb run
if len(wandb.patched["tensorboard"]) > 0:
    wandb.tensorboard.unpatch()
#wandb.tensorboard.patch(root_logdir="/content/drive/MyDrive/SOMN2/runs")
wandb.tensorboard.patch(root_logdir="./runs")

run1 = wandb.init(project='Fred_test_SU', #NOME DO PROJETO
                        # config=config_PPO,
                        group=f"priorizando lucro", #GRUPOS A SEREM ADCIONADOS NO WANDB
                    #   name=f"PPO (teste 12, atraso = {atraso:02d}, run: {x + 1:02d})",
                        name=f"PPO (teste 16, experimento 1, run 1",
                    #   name="run_test_SU", #NOME DA EXECUÇÃO
                        save_code=True,
                        reinit=True
    )


cart_env = env



random_policy = RandomPolicy(cart_env.action_space)
traj_obs, traj_acts, traj_rews = rollout_env(cart_env, random_policy, 
                                             n_steps = 8000, n_steps_reset=1000)

train_obs = traj_obs[:-1]
test_obs = traj_obs[-1]

train_acts = traj_acts[:-1]
test_acts = traj_acts[-1]
dyn_model.fit(train_obs, train_acts)

dyn_model.set_median_coef_()
dyn_model.print()

run1.finish()