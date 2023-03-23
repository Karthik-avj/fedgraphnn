To run the project: sh run_fed_subgraph_link_pred.sh 4

Parameters you can change: 
In the config/simulation/fedml_config.yaml, you can change the following parameters:

To run another model change the 'model' parameter to:
For GCN: "gcn", GIN: "gin", ChebConv: "cheb", GraphSage: "sage", GAT: "gat"

To change number of clients wunning the federated setting:
change 'client_num_in_total' and 'client_num_per_round'

To change the dataset change 'dataset' to:
'ciao' or 'epinion'

What each file does:
data/data_loader manipulates the dataset to group users that can be used by the federated setting
fedml_subgraph_link_prediction.py looks at the args and calls the appropriate model, federated learning agents
trainer/fed_subgraph_lp_trainer.py train the fedml federated model
trainer/fed_subgraph_lp_aggregator.py aggregates the fedml federated model for the global model let say gcn
model/gcn_link.py is the gcn(global) model, similarly for other models
config/simulation/fedml_config.yaml stores the parameters used by the model
