import flwr as fl
import sys
import numpy as np



#controller code
from controller import server_rounds
from controller import server_grpc_max_message_length
from controller import server_address



#########################################################################

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

# Create strategy and run server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address =server_address , 
        config=fl.server.ServerConfig(num_rounds=server_rounds) ,  
        grpc_max_message_length = server_grpc_max_message_length *server_grpc_max_message_length*server_grpc_max_message_length ,
        strategy = strategy
)
