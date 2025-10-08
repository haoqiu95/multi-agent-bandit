import torch, cvxpy
import numpy as np
import networkx as nx
import random
col_softmax = torch.nn.Softmax(dim=1)

inf = 1e10

class GSE:
    def __init__(self, n_agents, n_arms, W, network, p, horizon, device) -> None:
        self.Z = torch.zeros([n_agents, n_arms], device=device)
        self.graph = W
        self.p = p
        self.X0 = torch.ones([n_agents, n_arms], device=device) / n_arms
        self.device = device
        self.counts = torch.zeros([n_agents, n_arms], device=device)
        self.cum_L = torch.zeros([n_agents, n_arms], device=device)
        self.conf = torch.zeros([n_agents, n_arms], device=device)
        self.L = n_agents
        self.K = self.Z.shape[-1]
        self.n_agents = n_agents
        self.T = horizon
        self.network = network

    def action(self, t):
        A = torch.argmin(self.counts, dim=1, keepdim=True)
        A_one_hot = torch.nn.functional.one_hot(A, num_classes=self.K).squeeze(1)
        return A_one_hot

    def gossip(self):
        subg = self.graph.copy()
        edges_to_remove = [edge for edge in subg.edges if random.random() < self.p]
        subg.remove_edges_from(edges_to_remove)
        W = np.eye(self.n_agents) - nx.laplacian_matrix(subg).toarray()/self.n_agents
        return W

    def update(self, loss_matrix, actions):
        L_t = loss_matrix.to(self.device)

        #maintain pre_mean
        pre_mean = self.cum_L/torch.maximum(self.counts,torch.tensor(1.0))

        #update counts and cum loss
        self.counts += actions
        self.cum_L += L_t * actions

        log_T = torch.log(torch.tensor(self.T))

        # different graph structures
        if self.network == "COMPLETE":
            L = self.K * log_T/self.p 
            tau_star = log_T/(self.p)
        elif self.network == "GRID":
            L = self.K * log_T/self.p 
            tau_star = self.n_agents*log_T/(self.p) 
        elif self.network == "PETERSEN":
            L = self.K * log_T/self.p 
            tau_star = self.n_agents*log_T/(self.p) 

        #update mean 
        mean = self.cum_L/torch.maximum(self.counts,torch.tensor(1.0))
        conf_counts = self.n_agents * torch.maximum(self.counts- L, torch.tensor(1.0, device=self.counts.device))
        
        self.conf = torch.sqrt(log_T/conf_counts) + (tau_star + np.sqrt(self.n_agents))/conf_counts
        
        W = torch.tensor(self.gossip())
        # gossip step.
        self.Z = torch.mm(W.float(), self.Z.float()) + mean - pre_mean

        # arm elimination
        glcb = self.Z - self.conf
        gucb = self.Z + self.conf

        mask_1 = self.counts > (inf-1)
        gucb[mask_1] = inf 
        gucb_min, _ = torch.min(gucb, dim=1, keepdim=True)
        mask_2 = glcb > gucb_min
        self.counts[mask_2]=inf

        #intersection
        eliminated = torch.where(self.counts == inf, self.counts, torch.tensor(0))
        mask_3 = W.float() > 0


        eliminated_inter = torch.stack([
            eliminated[row_mask].max(dim=0).values
            for row_mask in mask_3
        ])

        self.counts = torch.max(self.counts, eliminated_inter)


class CommNet:
    def __init__(self, nx_graph) -> None:
        self.comm_net = nx_graph
    
    def max_deg_gossip(self, spectral_gap=False):
        degrees = [val for (node, val) in self.comm_net.degree()]
        max_deg = max(degrees)
        D = np.diag(degrees)
        A = nx.to_numpy_array(self.comm_net)
        P = np.eye(len(degrees)) - (D - A) / (max_deg+1)
        # spectral gap
        if spectral_gap:
            return P, compute_spectral_gap(P)
        return P

    def fast_gossip(self, algo, spectral_gap=False):
        if algo == 'SDP':
            comple_graph = nx.complement(self.comm_net)
            n = self.comm_net.number_of_nodes()
            P = cvxpy.Variable((n, n))
            e = np.ones(n)
            obj = cvxpy.Minimize(cvxpy.norm(P - 1.0/n))
            cnsts = [
                P@e==e,
                P.T == P,
                P >= 0
            ]
            for u, v in comple_graph.edges():
                if u != v: cnsts.append(P[u, v] == 0)
            prob = cvxpy.Problem(obj, cnsts)
            prob.solve()
            # spectral gap
            if spectral_gap:
                return P.value, compute_spectral_gap(P.value)
            return P.value
        else:
            raise NotImplementedError("The "+algo+" method has not been implemented.")



def cube_root_scheduler(gamma=0.01):
    '''Generates a series of exploration ratios'''
    step = 1
    while True:
        yield gamma / step ** (1/3) 
        step += 1

def compute_spectral_gap(P):
    singular_values = np.linalg.svd(P, compute_uv=False, hermitian=True)
    gap = 1 - singular_values[1]
    return gap

def fedexp3_ub_exact(n_epochs, n_agents, n_arms, spectral_gap, lr_array, gamma_array):
    C_w = 3 + min(
        2 * np.log(n_epochs) + np.log(n_agents),
        np.sqrt(n_agents)
    ) / spectral_gap
    lr_last = lr_array[-1]
    gamma_last = gamma_array[-1]
    cum_reg = np.log(n_arms) / lr_last
    for lr, gamma in zip(lr_array, gamma_array):
        first = n_arms**2/2 * lr / gamma
        second = n_arms**2 / gamma_last * C_w * lr
        third = gamma
        ins = first + second + third
        cum_reg += ins
        yield cum_reg

if __name__ == "__main__":
    # no communication
    N = 4
    adj = np.zeros([N, N])
    g = CommNet(
        nx.from_numpy_array(adj)
    )
    print(g.max_deg_gossip())

    # complete graph
    adj = np.ones([N, N])
    for i in range(N):
        adj[i][i] = 0
    g = CommNet(
        nx.from_numpy_array(adj)
    )
    print(g.max_deg_gossip())
    
    # sqrt(N)-by-sqrt(N) grid
    g = CommNet(nx.grid_graph([
        int(np.sqrt(N)),
        int(np.sqrt(N))
    ]))
    print(g.max_deg_gossip())
    
    # random geometric graphs with connectivity radius
    # r = 2 sqrt(log(n)^2 / n)
    r = np.sqrt(np.log(N) ** 1.1 / N)
    g = CommNet(nx.random_geometric_graph(
        N, r
    ))
    print(g.max_deg_gossip())

    
