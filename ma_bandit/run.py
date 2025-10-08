import os
import torch
import numpy as np
import networkx as nx
import federatedbandit.agent as fba
import federatedbandit.env as fbe
from PIL import Image
from tqdm import tqdm
from matplotlib import cm
from torch.utils.data import DataLoader


def main(config):
    use_cuda = torch.cuda.is_available()
    config['device'] = torch.device("cuda" if use_cuda else "cpu")
    rng = torch.Generator(device=config['device'])
    rng.manual_seed(config['seed'])

    # Create dataset
    env = config['env'].split('-')[0]
    if env == "HomoBandit":
        train_data = fbe.HomoBandit(
            config['horizon'], 
            config['n_agents'], 
            config['n_arms'],
            np.random.default_rng(
                int(config['env'].split('-')[-1]) # seed of the loss tensor
            )
        )
    elif env == 'HalfActBandit':
        train_data = fbe.StoActBandit(
            config['horizon'], 
            config['n_agents'], 
            config['n_arms'],
            config['n_agents']//2,               # activation size
            np.random.default_rng(
                int(config['env'].split('-')[1]) # seed of the loss tensor
            )
        )
    elif env == 'HalfFixActBandit':
        train_data = fbe.FixActBandit(
            config['horizon'], 
            config['n_agents'], 
            config['n_arms'],
            config['n_agents']//2,               # activation size
            np.random.default_rng(
                int(config['env'].split('-')[1]) # seed of the loss tensor
            )
        )
    else:
        raise NotImplementedError("The "+env+" environment has not been implemented.")
    
    train_loader = DataLoader(
        train_data,
        batch_size=1, 
        shuffle=False
    )
    # compute cumulative loss of the best arm in hindsight
    best_cumu_loss = train_data.cumloss_of_best_arm()

    # Specify communcation network
    network = config['network'].split('-')[0]
    if network == 'COMPLETE':
        graph = nx.complete_graph(config['n_agents'])
    elif network == 'GRID':
        graph = nx.grid_graph([
            int(np.sqrt(config['n_agents'])),
            int(np.sqrt(config['n_agents']))
        ])
    elif network == 'PETERSEN':
        graph = nx.petersen_graph()

    # Create GSE
    agent = fba.GSE(
        config['n_agents'],
        config['n_arms'],
        graph,
        config['network'],
        config['p'],
        config['horizon'],
        device=config['device']
    )

    # Create new algorithm here
    regret = []
    cumu_loss = 0
    rounds = len(train_loader)
    for i, loss_matrix in tqdm(enumerate(train_loader), total=rounds):
        L_t = torch.squeeze(loss_matrix, 0).to(config['device'])
        # make actions
        actions = agent.action(i)
        # compute cumulative losses
        cumu_loss += torch.matmul(
            torch.mean(L_t, dim=0),
            torch.transpose(actions.float(), 1, 0)
        )
        # update
        agent.update(L_t, actions)

        if i in config['snapshots']:
            regret.append(torch.mean(cumu_loss).item() - best_cumu_loss[i])
    return regret


def get_plot_points(n_steps):
    points = []
    for i in range(1, 101):
        if i * i < n_steps:
            points.append(i * i)
        else:
            break
    next_point = 10000
    while True:
        next_point *= 2
        if next_point < n_steps:
            points.append(next_point)
        else:
            break
    points.append(n_steps)
    return points
def filename(conf):
    return "%s_%s_N%d_K%d_T%d_p%0.1f" % (config['proj'],config['network'], conf['n_agents'], 
        conf['n_arms'], conf['horizon'], conf['p'])

if __name__ == "__main__":
    # repeated group simulations
    config = dict(
        proj = 'GSE-hete',
        env = 'None',
        network = 'None',
        n_agents = 16,
        n_arms = 5,                 
        horizon = 10000,                  
        gamma = 0.01,
        seed = None,
        p = None,
    )
    neworks = ['COMPLETE']
    edge_probs = [0.9]

    n_reps = 20
    plot_points = get_plot_points(config['horizon'])
    reg = np.zeros(shape=(len(plot_points)-1, n_reps))
    config['snapshots'] = plot_points

    for e in neworks:
        config['network'] = e
        for p in edge_probs:
            config['p'] = p
            for s in range(n_reps):
                config['env'] = 'HalfActBandit-' + str(s)
                config['seed'] = s
                reg[:, s] = main(config)
            mean = reg.mean(1)
            std = reg.std(1)
            DIR_PATH = os.path.dirname(os.path.realpath(__file__))
            os.chdir(DIR_PATH + "/../")

            if not os.path.exists('data'):
                os.makedirs('data')
            with open('data/' + filename(config), 'w') as file:
                file.write("# with %d runs\n" % (n_reps))
                file.writelines(["%d %f %f\n" % (plot_points[i], mean[i], std[i]) for i in range(len(plot_points)-1)])

