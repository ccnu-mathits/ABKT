# -*- coding: utf-8 -*-

"""

--------------------------------------

    Author: Xiaoxuan Shen
    
    Date:   2022/4/21 14:25
    
--------------------------------------

"""
import argparse
import os
from CMF_KM import CMF
from GMF_AM import GMF_BOOSTING

def parse_args():
    """
    Parse the embedding arguments.

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="ABKT")

    parser.add_argument('--dataset', default='AICFE',
                        help='Choose dataset. Default is "AICFE", choose from ["ASSISTment2009","AICFE"]. ')

    parser.add_argument('--type', default='math',
                        help='The subset of the dataset.'
                             '["RandomIterateSection","RandomChildOrderSection"] for dataset ASSISTment2009'
                             '["math","phy"] for dataset AICFE')

    # hyper-parameters in knowledge module
    parser.add_argument('--KM_k', type=int, default=5,
                        help='The dimensionality of knowledge module, namely k_K, Default is 5')

    parser.add_argument('--KM_guess', type=float, default=0.25,
                        help='The surmise of knowledge module. Default is 0.25.')

    parser.add_argument('--use_pertrained_model', type=bool, default=True,
                        help='Weather use protrained knowledge module or not. Default is True.')

    # hyper-parameters in ability module
    parser.add_argument('--AM_k', type=int, default=32,
                        help='The dimensionality of ability model, namely k_A,. Default is 32.')

    parser.add_argument('--AM_lambda', type=float, default=0.1,
                        help='The control factor of regularization term in ability model. Default is 0.1. ')

    parser.add_argument('--AM_layer', type=int, default=1,
                        help='The depth of feature aggregation in the ability module, namly l. '
                             'Default is 0.1. Choose from [1,2,3] ')

    # hyper-parameters in boosting model
    parser.add_argument('--pretrain_clip', type=float, default=0.4,
                        help='The clip range, namly _mu. Default is 0.4. ')
    parser.add_argument('--joint_model', default="add",
                        help='The type of the joint model. Default is "add". Choose from ["add","mul"]. ')

    # hyper-parameters in training process
    parser.add_argument('--device', default="cuda:0",
                        help='The working device of pytorch, '
                             'Default is "cpu", choose from ["cpu","cuda:0","cuda:1",...] ')

    return parser.parse_args()


def main():
    """
    Pipeline for ABKT
    """
    args = parse_args()
    print("Checking the per-trained knowledge model...")
    KM_path = './Models/'+str(args.dataset)+'-'+str(args.type)+'/CMF-k-'+str(args.KM_k)+'-'+str(args.KM_guess)+'-earlystop'
    if os.access(KM_path, os.F_OK) and args.use_pertrained_model:
        print("per-trained knowledge model is existent...")
    else:
        print("per-trained knowledge model is not existent...")
        print("Training the knowledge model...")
        cmf = CMF(
            dataset=args.dataset,
            type=args.type,
            k_hidden_size=args.KM_k,
            guess=args.KM_guess,
            device=args.device,
        )
        cmf.train()
        cmf.log_result()
    print("Training the boosted ability model...")
    gmf = GMF_BOOSTING(
        dataset=args.dataset,
        type=args.type,
        CMF_k=args.KM_k,
        CMF_guess=args.KM_guess,
        embedding_k=args.AM_k,
        m_lambda=args.AM_lambda,
        GMF_layer=args.AM_layer,
        pretrain_clip=args.pretrain_clip,
        combine=args.joint_model,
        device=args.device,
    )
    gmf.train()
    gmf.log_result()



if __name__ == "__main__":
    main()