import torch
from federated.federated_learning import FederatedLearning
from .robust_aggregation import krum_aggregation, median_aggregation, trimmed_mean_aggregation

class RobustFederatedLearning(FederatedLearning):
    def update_global_model(self, client_weights, client_sizes, method='krum', **kwargs):
        """
        Update the global model using various robust aggregation methods
        method: 'krum', 'median', 'trimmed_mean'
        kwargs: additional parameters, such as num_adv, trim_ratio, etc.
        """
        if method == 'krum':
            agg_weights = krum_aggregation(client_weights, kwargs.get('num_adv', 1))
        elif method == 'median':
            agg_weights = median_aggregation(client_weights)
        elif method == 'trimmed_mean':
            agg_weights = trimmed_mean_aggregation(client_weights, kwargs.get('trim_ratio', 0.1))
        else:
            raise NotImplementedError(f"Unknown aggregation method: {method}")
        self.global_model.load_state_dict(agg_weights) 