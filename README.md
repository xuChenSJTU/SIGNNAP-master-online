# SIGNN-master-online
**This is a Pytorch implementation of our paper "Learning Node Representations against Perturbations in Graph Neural Networks"**    
  
**1. Datasets:**   
We use six benchmarks: Pubmed, Facbook, Coauthor-CS, Coauthor-Phy.  Amazon-Computer and Amazon-Photo
The transductive learning setting is conducted on Pubmed, Facbook, Coauthor-CS, Amazon-Computer, Amazon-Photo, and the inductive learning setting is conducted on Coauthor-Phy.  

**2. Requirements:**  
(1) GPU machine  
(2) Necessary packages:  
python3.5; pytorch=1.4.0; tqdm=4.27.0; tensorboardX=1.8; pandas=0.25; numpy=1.15; networkx=2.2; logger=1.4; scipy=1.1; scikit-learn=0.20  

**3. Notes:**  
(1) Note that some of the codes may be redundant and are not useful, we will polish them later.    
(2) Note that some of the codes are based on different released codes, so the same arguement in different .py files may have different meanings.     
(3) Note that the following comands consist of both training and evaluation, if you do not want to train Ours, you can directly use the trained models we provided in xxxx/src/saved_models/.    
Then, you can directly check the ############################### Evaluation for Ours ######################################## part and run the evaluation results.     
For example:     
--> CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=pubmed --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1    
--> CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=facebook_page --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1    
--> CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=coauthor_cs --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1    
--> CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=coauthor_phy --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1      
--> CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=amazon_computer --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     
--> CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=coauthor_phy --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0      

**4. We will illustrate how to run the codes as following.**    

In order to run our codes, you need to get in the: xxxx/src/ file first.      
**(1) On Pubmed dataset**      
############################### Baselines ########################################  
GCN: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=pubmed --sampling_percent=1.0 --type=multigcn --nbaseblocklayer=0  

ResGCN: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=pubmed --sampling_percent=1.0 --type=resgcn --nbaseblocklayer=1  

JKNet: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=pubmed --sampling_percent=1.0 --type=densegcn --nbaseblocklayer=1  

GraphSage: CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_graphsage.py --dataset=pubmed --sampling_percent=1.0 --nhiddenlayer=2  

GAT: CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_gat.py --dataset=pubmed --sampling_percent=1.0 --nhiddenlayer=1  

DropEdge(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=pubmed --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0  

DropEdge(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=pubmed --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1  

DropEdge(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=pubmed --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1  

DropEdge(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_graphsage.py --dataset=pubmed --sampling_percent=0.7 --nhiddenlayer=2  

DropEdge(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_gat.py --dataset=pubmed --sampling_percent=0.7 --nhiddenlayer=1  

DGI: CUDA_VISIBLE_DEVICES=your_gpu_num python train_DGI.py --dataset=pubmed  

############################### Ours ########################################  
Ours(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=pubmed --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1  

Ours(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=pubmed --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=pubmed --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours_graphsage.py --dataset=pubmed --sampling_percent=0.7 --nhiddenlayer=2 --nce_k=1024 --nce_t=0.1  

Ours(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours_gat.py --dataset=pubmed --sampling_percent=0.7 --nhiddenlayer=1 --nce_k=1024 --nce_t=0.1  

############################### Evaluation for Ours ########################################  
Since our model is an unsupervised learning one, we provide the following code for evaluation after the model training.  

Ours(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=pubmed --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1  

Ours(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=pubmed --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=pubmed --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervised_graphsage.py --dataset=pubmed --sampling_percent=0.7 --nhiddenlayer=2 --nce_k=1024 --nce_t=0.1  

Ours(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervised_gat.py --dataset=pubmed --sampling_percent=0.7 --nhiddenlayer=1 --nce_k=1024 --nce_t=0.1  


**(2) On Facebook dataset**  
############################### Baselines ########################################  
GCN: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=facebook_page --sampling_percent=1.0 --type=multigcn --nbaseblocklayer=0  

ResGCN: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=facebook_page --sampling_percent=1.0 --type=resgcn --nbaseblocklayer=1  

JKNet: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=facebook_page --sampling_percent=1.0 --type=densegcn --nbaseblocklayer=1  

GraphSage: CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_graphsage.py --dataset=facebook_page --sampling_percent=1.0 --nhiddenlayer=2  

GAT: CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_gat.py --dataset=facebook_page --sampling_percent=1.0 --nhiddenlayer=1  

DropEdge(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=facebook_page --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0  

DropEdge(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=facebook_page --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1  

DropEdge(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=facebook_page --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1  

DropEdge(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_graphsage.py --dataset=facebook_page --sampling_percent=0.7 --nhiddenlayer=2  

DropEdge(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_gat.py --dataset=facebook_page --sampling_percent=0.7 --nhiddenlayer=1  

DGI: CUDA_VISIBLE_DEVICES=your_gpu_num python train_DGI.py --dataset=facebook_page  

############################### Ours ########################################  
Ours(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=facebook_page --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1  

Ours(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=facebook_page --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=facebook_page --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours_graphsage.py --dataset=facebook_page --sampling_percent=0.7 --nhiddenlayer=2 --nce_k=1024 --nce_t=0.1  

Ours(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours_gat.py --dataset=facebook_page --sampling_percent=0.7 --nhiddenlayer=1 --nce_k=1024 --nce_t=0.1  

############################### Evaluation for Ours ########################################  
Since our model is an unsupervised learning one, we provide the following code for evaluation after the model training.  

Ours(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=facebook_page --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1  

Ours(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=facebook_page --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=facebook_page --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervised_graphsage.py --dataset=facebook_page --sampling_percent=0.7 --nhiddenlayer=2 --nce_k=1024 --nce_t=0.1  

Ours(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervised_gat.py --dataset=facebook_page --sampling_percent=0.7 --nhiddenlayer=1 --nce_k=1024 --nce_t=0.1  

**(3) On Coauthor-CS dataset**  
############################### Baselines ########################################  
GCN: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=coauthor_cs --sampling_percent=1.0 --type=multigcn --nbaseblocklayer=0  

ResGCN: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=coauthor_cs --sampling_percent=1.0 --type=resgcn --nbaseblocklayer=1  

JKNet: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=coauthor_cs --sampling_percent=1.0 --type=densegcn --nbaseblocklayer=1  

GraphSage: CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_graphsage.py --dataset=coauthor_cs --sampling_percent=1.0 --nhiddenlayer=2  

GAT: CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_gat.py --dataset=coauthor_cs --sampling_percent=1.0 --nhiddenlayer=1  

DropEdge(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=coauthor_cs --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0  

DropEdge(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=coauthor_cs --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1  

DropEdge(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=coauthor_cs --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1  

DropEdge(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_graphsage.py --dataset=coauthor_cs --sampling_percent=0.7 --nhiddenlayer=2  

DropEdge(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_gat.py --dataset=coauthor_cs --sampling_percent=0.7 --nhiddenlayer=1  

DGI: CUDA_VISIBLE_DEVICES=your_gpu_num python train_DGI.py --dataset=coauthor_cs  

############################### Ours ########################################  
Ours(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=coauthor_cs --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1  

Ours(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=coauthor_cs --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=coauthor_cs --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours_graphsage.py --dataset=coauthor_cs --sampling_percent=0.7 --nhiddenlayer=2 --nce_k=1024 --nce_t=0.1  

Ours(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours_gat.py --dataset=coauthor_cs --sampling_percent=0.7 --nhiddenlayer=1 --nce_k=1024 --nce_t=0.1  

############################### Evaluation for Ours ########################################  
Since our model is an unsupervised learning one, we provide the following code for evaluation after the model training.  

Ours(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=coauthor_cs --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1  

Ours(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=coauthor_cs --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=coauthor_cs --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervised_graphsage.py --dataset=coauthor_cs --sampling_percent=0.7 --nhiddenlayer=2 --nce_k=1024 --nce_t=0.1  

Ours(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervised_gat.py --dataset=coauthor_cs --sampling_percent=0.7 --nhiddenlayer=1 --nce_k=1024 --nce_t=0.1  

**(4) On Coauthor-Phy dataset**  
############################### Baselines ########################################  
GCN: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=coauthor_phy --sampling_percent=1.0 --type=multigcn --nbaseblocklayer=0  

ResGCN: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=coauthor_phy --sampling_percent=1.0 --type=resgcn --nbaseblocklayer=1  

JKNet: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=coauthor_phy --sampling_percent=1.0 --type=densegcn --nbaseblocklayer=1  

GraphSage: CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_graphsage.py --dataset=coauthor_phy --sampling_percent=1.0 --nhiddenlayer=2  

GAT: CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_gat.py --dataset=coauthor_phy --sampling_percent=1.0 --nhiddenlayer=1  

DropEdge(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=coauthor_phy --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0  

DropEdge(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=coauthor_phy --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1  

DropEdge(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=coauthor_phy --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1  

DropEdge(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_graphsage.py --dataset=coauthor_phy --sampling_percent=0.7 --nhiddenlayer=2  

DropEdge(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_gat.py --dataset=coauthor_phy --sampling_percent=0.7 --nhiddenlayer=1  

DGI: CUDA_VISIBLE_DEVICES=your_gpu_num python train_DGI.py --dataset=coauthor_phy  

############################### Ours ########################################  
Ours(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=coauthor_phy --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1  

Ours(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=coauthor_phy --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=coauthor_phy --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours_graphsage.py --dataset=coauthor_phy --sampling_percent=0.7 --nhiddenlayer=2 --nce_k=1024 --nce_t=0.1  

Ours(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours_gat.py --dataset=coauthor_phy --sampling_percent=0.7 --nhiddenlayer=1 --nce_k=1024 --nce_t=0.1  

############################### Evaluation for Ours ########################################  
Since our model is an unsupervised learning one, we provide the following code for evaluation after the model training.  

Ours(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=coauthor_phy --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1  

Ours(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=coauthor_phy --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=coauthor_phy --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1  

Ours(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervised_graphsage.py --dataset=coauthor_phy --sampling_percent=0.7 --nhiddenlayer=2 --nce_k=1024 --nce_t=0.1  

Ours(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervised_gat.py --dataset=coauthor_phy --sampling_percent=0.7 --nhiddenlayer=1 --nce_k=1024 --nce_t=0.1  


**(5) On Amazon-Com dataset**  
############################### Baselines ########################################  
GCN: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=amazon_computer --sampling_percent=1.0 --type=multigcn --nbaseblocklayer=0 --weight_decay=0.0    

ResGCN: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=amazon_computer --sampling_percent=1.0 --type=resgcn --nbaseblocklayer=1 --weight_decay=0.0      

JKNet: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=amazon_computer --sampling_percent=1.0 --type=densegcn --nbaseblocklayer=1 --weight_decay=0.0     

GraphSage: CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_graphsage.py --dataset=amazon_computer --sampling_percent=1.0 --nhiddenlayer=2 --weight_decay=0.0     

GAT: CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_gat.py --dataset=amazon_computer --sampling_percent=1.0 --nhiddenlayer=1 --weight_decay=0.0     

DropEdge(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=amazon_computer --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --weight_decay=0.0     

DropEdge(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=amazon_computer --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1 --weight_decay=0.0     

DropEdge(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=amazon_computer --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1 --weight_decay=0.0     

DropEdge(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_graphsage.py --dataset=amazon_computer --sampling_percent=0.7 --nhiddenlayer=2  --weight_decay=0.0    

DropEdge(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_gat.py --dataset=amazon_computer --sampling_percent=0.7 --nhiddenlayer=1 --weight_decay=0.0     

DGI: CUDA_VISIBLE_DEVICES=your_gpu_num python train_DGI.py --dataset=amazon_computer --weight_decay=0.0     

############################### Ours ########################################  
Ours(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=amazon_computer --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=amazon_computer --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=amazon_computer --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours_graphsage.py --dataset=amazon_computer --sampling_percent=0.7 --nhiddenlayer=2 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours_gat.py --dataset=amazon_computer --sampling_percent=0.7 --nhiddenlayer=1 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

############################### Evaluation for Ours ########################################  
Since our model is an unsupervised learning one, we provide the following code for evaluation after the model training.  

Ours(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=amazon_computer --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=amazon_computer --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=amazon_computer --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervised_graphsage.py --dataset=amazon_computer --sampling_percent=0.7 --nhiddenlayer=2 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervised_gat.py --dataset=amazon_computer --sampling_percent=0.7 --nhiddenlayer=1 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0    

**(6) On Amazon-Pho dataset**  
############################### Baselines ########################################  
GCN: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=amazon_photo --sampling_percent=1.0 --type=multigcn --nbaseblocklayer=0 --weight_decay=0.0    

ResGCN: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=amazon_photo --sampling_percent=1.0 --type=resgcn --nbaseblocklayer=1 --weight_decay=0.0      

JKNet: CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=amazon_photo --sampling_percent=1.0 --type=densegcn --nbaseblocklayer=1 --weight_decay=0.0     

GraphSage: CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_graphsage.py --dataset=amazon_photo --sampling_percent=1.0 --nhiddenlayer=2 --weight_decay=0.0     

GAT: CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_gat.py --dataset=amazon_photo --sampling_percent=1.0 --nhiddenlayer=1 --weight_decay=0.0     

DropEdge(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=amazon_photo --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --weight_decay=0.0     

DropEdge(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=amazon_photo --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1 --weight_decay=0.0     

DropEdge(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python train_new.py --dataset=amazon_photo --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1 --weight_decay=0.0     

DropEdge(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_graphsage.py --dataset=amazon_photo --sampling_percent=0.7 --nhiddenlayer=2  --weight_decay=0.0    

DropEdge(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python train_news_gat.py --dataset=amazon_photo --sampling_percent=0.7 --nhiddenlayer=1 --weight_decay=0.0     

DGI: CUDA_VISIBLE_DEVICES=your_gpu_num python train_DGI.py --dataset=amazon_photo --weight_decay=0.0     

############################### Ours ########################################  
Ours(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=amazon_photo --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=amazon_photo --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours.py --dataset=amazon_photo --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours_graphsage.py --dataset=amazon_photo --sampling_percent=0.7 --nhiddenlayer=2 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python train_Ours_gat.py --dataset=amazon_photo --sampling_percent=0.7 --nhiddenlayer=1 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

############################### Evaluation for Ours ########################################  
Since our model is an unsupervised learning one, we provide the following code for evaluation after the model training.  

Ours(GCN): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=amazon_photo --sampling_percent=0.7 --type=multigcn --nbaseblocklayer=0 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(ResGCN): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=amazon_photo --sampling_percent=0.7 --type=resgcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(JKNet): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervisd.py --dataset=amazon_photo --sampling_percent=0.7 --type=densegcn --nbaseblocklayer=1 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(GraphSage): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervised_graphsage.py --dataset=amazon_photo --sampling_percent=0.7 --nhiddenlayer=2 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0     

Ours(GAT): CUDA_VISIBLE_DEVICES=your_gpu_num python eval_unsupervised_gat.py --dataset=amazon_photo --sampling_percent=0.7 --nhiddenlayer=1 --nce_k=1024 --nce_t=0.1 --weight_decay=0.0   


**For codes of other methods such as DeepWalk, they have been provided by the authors and can be widely reached online, we do not repeat them here**  

**Thanks for your interest in our paper. We benefit a lot from codes provided by other researchers, we would like to thank them here.**  

**Also, this paper is under review now, we will provide more information later.**  
