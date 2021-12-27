import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='model and training confg')
    subparsers = parser.add_subparsers()
    parser_v1 = subparsers.add_parser('v1')
    parser_v2 = subparsers.add_parser('v2')
    # Training confg
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--load_from_path', type=str, default=None)
    parser.add_argument('--save_to_path', type=str, default="checkpoints/lastest/")
    
    # V1 
    parser_v1.add_argument('--model', type=str, default="model_v1") # DO NOT EDIT THIS
    # Categorical Embeddings
    parser_v1.add_argument('--embed_loc_size', type=int, default=5)
    parser_v1.add_argument('--embed_time_size', type=int, default=10)
    # GRU
    parser_v1.add_argument('--num_hiddens', type=int, default=8)
    parser_v1.add_argument('--num_layers', type=int, default=2)
    parser_v1.add_argument('--GRU_dropout', type=float, default=0)
    # MLP
    parser_v1.add_argument('--l_dim', type=int, default=16)

    # V2
    parser_v2.add_argument('--model', type=str, default="model_v2") # DO NOT EDIT THIS
    # Categorical Embeddings
    parser_v2.add_argument('--embed_loc_size', type=int, default=5)
    parser_v2.add_argument('--embed_time_size', type=int, default=10)
    # GRU
    parser_v2.add_argument('--num_hiddens', type=int, default=8)
    parser_v2.add_argument('--num_layers', type=int, default=2)
    # MLP
    parser_v2.add_argument('--l_dim', type=int, default=16)
    # Attention
    parser_v2.add_argument('--dropout', type=float, default=0)

    # dataset-specific: DO NOT EDIT THEM
    parser.add_argument('--loc_dim', type=int, default=10)
    parser.add_argument('--time_dim', type=int, default=24)
    parser.add_argument('--X_dim', type=int, default=49)
    parser.add_argument('--output_dim', type=int, default=1)

    args = parser.parse_args()
    print(args)
    return args
