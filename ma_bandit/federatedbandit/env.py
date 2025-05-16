import os
import datetime
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class HomoBandit(Dataset):
    def __init__(self, n_epochs, n_agents, n_arms, rng) -> None:
        super().__init__()
        global_means = np.linspace(0, 1, n_arms)
        L = np.array(
            [rng.binomial(1, p, size=(n_epochs, n_agents)) for p in global_means], 
            dtype=np.float32
        )
        self.data = np.transpose(L, (1, 2, 0))
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]

    def cumloss_of_best_arm(self):
        true_loss = np.mean(self.data, axis=1)
        cum_losses = np.cumsum(true_loss, axis=0)
        best_arm = np.argmin(cum_losses[-1,])
        return cum_losses[:,best_arm]
        

class StoActBandit(HomoBandit):
    def __init__(self, n_epochs, n_agents, n_arms, activate_size, rng) -> None:
        super().__init__(n_epochs, n_agents, n_arms, rng)
        for t in range(n_epochs):
            non_selected_idx = rng.choice(
                n_agents,
                size=n_agents - activate_size,
                replace=False
            )
            self.data[t,non_selected_idx,:] = 0

class FixActBandit(HomoBandit):
    def __init__(self, n_epochs, n_agents, n_arms, activate_size, rng) -> None:
        super().__init__(n_epochs, n_agents, n_arms, rng)
        self.data[:,activate_size:,:] = 0
            

class MovieLens(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_pickle(
            os.path.join(
                os.path.dirname(__file__),
                '../MovieLens/MovieLens_loss.pkl'
            )
        )
        start = self.data.index.values[0][0]
        end = self.data.index.values[-1][0]
        self.start_date = datetime.datetime.strptime(start, '%Y-%m-%d').date()
        self.end_date = datetime.datetime.strptime(end, '%Y-%m-%d').date()
        self.genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
            'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
            'Documentary', '(no genres listed)'
        ]
        delta = self.end_date - self.start_date
        self.T = delta.days + 1
        self.N = 610
        self.K = len(self.genres)

    def __len__(self):
        return self.T

    def get_armId(self, genre):
        return self.genres.index(genre)
    
    def __getitem__(self, index):
        today = self.start_date + datetime.timedelta(days=index)
        losses = np.zeros((self.N, self.K))
        for i in range(self.N):
            for j, genre in enumerate(self.genres):
                key = (str(today), i+1, genre)
                if key in self.data.index:
                    losses[i, j] = self.data['loss'][key]
        return losses


