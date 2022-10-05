from models.admit import *
from args import Helper
from utils.train_helper import *
from utils.eval_helper import *
from utils.model_helper import *
from data.dataset import data1
from copy import deepcopy
from utils.data_helper import *
import os
import logging
from sklearn.preprocessing import StandardScaler

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    helper = Helper()
    args = helper.config
    args.device = device
    args.args_to_dict = helper.args_to_dict
    setup_seed(args.seed)

    logger = None
    if args.log:
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        
        handler = logging.FileHandler("{}/log_{}.txt".format(args.log_dir, args.local_time))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info(str(args.args_to_dict))

    n_train, n_eval = int(args.n_data * 0.67), int(args.n_data * 0.23)
    n_test = args.n_data - n_train - n_eval

    if args.load_data:
        try:
            train_data = load_train(args)
            eval_data = load_eval(args)
            test_data = load_test(args)
            print('load train and test data succesfully')
        except Exception as e:
            print('error in load data : {}'.format(e))
            exit()
    # create new data for training and testing
    else:
        train_data = data1(n_train)
        eval_data = data1(n_eval)
        test_data = data1(n_test)

        if args.save_data:
            save_train(args, train_data)
            save_eval(args, eval_data)
            save_test(args, test_data)
            print('save train and test data succesfully')

    if args.scale:
        args.scaler = StandardScaler().fit(train_data.y.reshape(-1,1))

    model = ADMIT(args)
    model.to(device)

    for epoch, model, loss in train(model, train_data, args):
        print(epoch, loss)
        if epoch % args.verbose == 0:
            _, mse, _ = eval(model, args, train_data, eval_data, test_data)
            print('eval_mse {:.5f}'.format(mse))
            
            if logger:
                logger.info('epoch: {}, eval_mse {:.5f}'.format(epoch, mse))

if __name__ == "__main__":      
    main()