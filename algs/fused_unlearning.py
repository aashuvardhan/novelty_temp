import time
import math
import pandas as pd
import torch

from models.Model_base import *
from models import LeNet_FashionMNIST, CNN_Cifar10, CNN_Cifar100
from utils import init_network, test_class_forget, test_client_forget
from dataset.data_utils import *
from algs.fl_base import Base
import torch.optim as optim
import copy
import logging
import matplotlib.pyplot as plt
from utils import *
import random
from models.Model_base import *
from fim_utils import compute_layer_sensitivity

class FUSED(Base):
    def __init__(self, args):
        super(FUSED, self).__init__(args)
        self.args = args
        self.log_dir = f"logs/fused_{self.args.data_name}_{self.args.alpha}"
        self.param_change_dict = {}
        self.param_size = {}

    def train_normal(self, global_model, client_all_loaders, test_loaders):
        checkpoints_ls = []
        result_list = []
        param_list = []
        for name, param in global_model.named_parameters():
            self.param_change_dict[name] = 0
            self.param_size[name] = 0

        for epoch in range(self.args.global_epoch):
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user*self.args.fraction), replace=False))
            select_client_loaders = [client_all_loaders[idx] for idx in selected_clients]

            # global_train_once now returns averaged model directly (online FedAvg)
            global_model = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)

            if self.args.forget_paradigm == 'sample':
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model,
                                                                                              self.args, test_loaders)
                print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero, avg_test_acc))
                result_list.extend(test_result_ls)
            elif self.args.forget_paradigm == 'client':
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args, test_loaders)
                print('Epoch={}, avg_f_acc={}, avg_r_acc={}'.format(epoch, avg_f_acc, avg_r_acc))
                result_list.extend(test_result_ls)
            elif self.args.forget_paradigm == 'class':
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args, test_loaders)
                print('Epoch={}, avg_f_acc={}, avg_r_acc={}'.format(epoch, avg_f_acc, avg_r_acc))
                result_list.extend(test_result_ls)

            if self.args.paradigm == 'fused':
                diff_ls = list(self.param_change_dict.values())
                name = list(self.param_change_dict.keys())
                diff_ls_ = [float(i) for i in diff_ls]
                param_list.append(diff_ls_)

        df = pd.DataFrame(param_list, columns=name)
        df.to_csv('./results/param_change_{}_distri_{}.csv'.format(self.args.data_name, self.args.alpha))

        torch.save(global_model.state_dict(), 'save_model/global_model_{}.pth'.format(self.args.data_name))

        if self.args.forget_paradigm == 'sample':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc'])
        elif self.args.forget_paradigm == 'client':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss'])
        elif self.args.forget_paradigm == 'class':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss'])
        if self.args.save_normal_result:
            df.to_csv('./results/Acc_loss_fl_{}_data_{}_distri_{}.csv'.format(self.args.forget_paradigm, self.args.data_name, self.args.alpha))

        return global_model, []


    def forget_client_train(self, global_model, client_all_loaders, test_loaders):
        global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(self.args.data_name)))
        avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, 1, global_model, self.args,
                                                                  test_loaders)
        print('FUSED-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format('xxxx', avg_r_acc,
                                                                                 avg_f_acc))

        # =====================================================================
        # FIM-GUIDED DYNAMIC LORA PLACEMENT
        # =====================================================================
        # Step 1: Build small clean and poison loaders for FIM analysis
        from torch.utils.data import TensorDataset, DataLoader as DL

        use_distilled = getattr(self.args, 'distill_data', False)

        if use_distilled:
            # ---- DISTILLED PATH: load pre-computed DM tensors ----
            import os
            print("[Phase 2] Loading DISTILLED data for FIM analysis ...")
            clean_x_parts, clean_y_parts = [], []
            poison_x_parts, poison_y_parts = [], []
            for client_id in range(self.args.num_user):
                pt_path = os.path.join('distilled_data', f'client_{client_id}.pt')
                if not os.path.exists(pt_path):
                    print(f"  WARNING: {pt_path} not found, skipping client {client_id}")
                    continue
                syn_img, syn_lbl = torch.load(pt_path, map_location=self.args.device)
                if client_id in self.args.forget_client_idx:
                    poison_x_parts.append(syn_img)
                    poison_y_parts.append(syn_lbl)
                else:
                    clean_x_parts.append(syn_img)
                    clean_y_parts.append(syn_lbl)
            clean_x = torch.cat(clean_x_parts, dim=0)
            clean_y = torch.cat(clean_y_parts, dim=0)
            poison_x = torch.cat(poison_x_parts, dim=0)
            poison_y = torch.cat(poison_y_parts, dim=0)
            print(f"  Distilled clean samples: {clean_x.shape[0]}, poison samples: {poison_x.shape[0]}")
        else:
            # ---- ORIGINAL PATH: aggregate raw client batches ----
            clean_data_list = []
            poison_data_list = []
            for idx in range(self.args.num_user):
                loader = client_all_loaders[idx]
                for data, target in loader:
                    if idx in self.args.forget_client_idx:
                        poison_data_list.append((data, target))
                    else:
                        clean_data_list.append((data, target))
            clean_x = torch.cat([d[0] for d in clean_data_list[:20]], dim=0)
            clean_y = torch.cat([d[1] for d in clean_data_list[:20]], dim=0)
            poison_x = torch.cat([d[0] for d in poison_data_list[:20]], dim=0)
            poison_y = torch.cat([d[1] for d in poison_data_list[:20]], dim=0)

        clean_loader_fim = DL(TensorDataset(clean_x, clean_y), batch_size=64, shuffle=False)
        poison_loader_fim = DL(TensorDataset(poison_x, poison_y), batch_size=64, shuffle=False)

        # ---- Build per-client distilled DataLoaders for the TRAINING LOOP ----
        # Only done when DM is enabled. The epoch loop below uses these small
        # synthetic loaders instead of the full real client_all_loaders, which
        # is what delivers the expected speedup.
        distilled_train_loaders = []
        if use_distilled:
            for client_id in range(self.args.num_user):
                if client_id in self.args.forget_client_idx:
                    continue  # skip forget clients
                pt_path = os.path.join('distilled_data', f'client_{client_id}.pt')
                if not os.path.exists(pt_path):
                    print(f"  WARNING: {pt_path} not found, skipping client {client_id}")
                    continue
                syn_img, syn_lbl = torch.load(pt_path, map_location='cpu')
                dl = DL(TensorDataset(syn_img, syn_lbl),
                        batch_size=self.args.local_batch_size, shuffle=True, drop_last=True)
                distilled_train_loaders.append(dl)
            print(f"[Phase 2] Training loop will use {len(distilled_train_loaders)} distilled client loaders "
                  f"({sum(len(dl.dataset) for dl in distilled_train_loaders)} total synthetic samples).")
        
        # Step 2: Run FIM sensitivity analysis
        target_modules, rank_map, sensitivity_scores = compute_layer_sensitivity(
            global_model,
            clean_loader_fim,
            poison_loader_fim,
            self.args.device,
            alpha=getattr(self.args, 'alpha_fim', 1.0),
            percentile=getattr(self.args, 'fim_percentile', 70),
            max_batches=getattr(self.args, 'fim_max_batches', 10),
        )
        
        # Step 3: Create DynamicLora with FIM-determined targets
        fused_model = DynamicLora(self.args, global_model, target_modules, rank_map)
        torch.save(fused_model.state_dict(), 'save_model/global_fusedmodel_{}.pth'.format(self.args.data_name))

        checkpoints_ls = []
        result_list = []
        consume_time = 0

        std_time = time.time()
        for epoch in range(self.args.global_epoch):
            fused_model.train()
            if use_distilled:
                # Use tiny synthetic loaders — this is what gives the speedup
                select_client_loaders = distilled_train_loaders
            else:
                selected_clients = [i for i in range(self.args.num_user)
                                    if i not in self.args.forget_client_idx]
                select_client_loaders = select_part_sample(
                    self.args, client_all_loaders, selected_clients)

            # global_train_once returns averaged model directly
            avg_model = self.global_train_once(epoch, fused_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            consume_time += end_time - std_time
            std_time = end_time  # reset for next epoch
            fused_model.load_state_dict(avg_model.state_dict())
            del avg_model

            fused_model.eval()

            avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, fused_model, self.args,
                                                                      test_loaders)

            result_list.extend(test_result_ls)

            print('FUSED-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format(epoch, avg_r_acc,
                                                                                    avg_f_acc))


        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss'])
        df['Comsume_time'] = consume_time

        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv('./results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm,
                                                                                              self.args.forget_paradigm,
                                                                                              self.args.data_name,
                                                                                              self.args.alpha,
                                                                                              len(self.args.forget_class_idx)))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(
                        self.args.forget_paradigm,
                        self.args.forget_paradigm,
                        self.args.data_name,
                        self.args.alpha,
                        len(self.args.forget_class_idx), self.args.cut_sample))
                
        
        # =====================================================================
        # NEW CODE: Retrieve and save the original Phase 1 model
        # =====================================================================
        print("Unlearning complete. Extracting the original Phase 1 base model...")
        
        # We use deepcopy so we don't accidentally destroy the fused_model we need to return
        temp_fused_model = copy.deepcopy(fused_model)
        
        # Drop the adapters and extract the frozen Phase 1 base model
        original_restored_model = temp_fused_model.lora_model.unload()
        
        # Save this restored model to the hard drive so you can inspect it later
        torch.save(original_restored_model.state_dict(), 'save_model/restored_phase1_model_{}.pth'.format(self.args.data_name))
        print("Original Phase 1 model successfully restored and saved to disk!")
        # =====================================================================

        # We return the FUSED model so the MIA and Relearning phases have the "cured" model to test
        

        return fused_model

    def forget_class(self, global_model, client_all_loaders, test_loaders):
        checkpoints_ls = []
        result_list = []
        consume_time = 0

        fused_model = Lora(self.args, global_model)
        for epoch in range(self.args.global_epoch):
            fused_model.train()
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user * self.args.fraction), replace=False))

            select_client_loaders = select_part_sample(self.args, client_all_loaders, selected_clients)
            std_time = time.time()

            client_models = self.global_train_once(epoch, fused_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            fused_model.load_state_dict(client_models.state_dict())  # client_models is now the avg model
            consume_time += end_time-std_time
            avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, fused_model, self.args, test_loaders)
            result_list.extend(test_result_ls)
            print('Epoch={}, Remember Test Acc={}, Forget Test Acc={}'.format(epoch, avg_r_acc, avg_f_acc))

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss'])
        df['Comsume_time'] = consume_time

        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm, self.args.forget_paradigm, self.args.data_name, self.args.alpha, len(self.args.forget_class_idx)))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                     self.args.forget_paradigm,
                                                                                     self.args.data_name,
                                                                                     self.args.alpha,
                                                                                     len(self.args.forget_class_idx), self.args.cut_sample))

        return fused_model

    def forget_sample(self, global_model, client_all_loaders, test_loaders):
        checkpoints_ls = []
        result_list = []
        consume_time = 0

        fused_model = Lora(self.args, global_model)
        for epoch in range(self.args.global_epoch):
            fused_model.train()
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user * self.args.fraction), replace=False))# 将需要遗忘的客户端排除在外

            self.select_forget_idx = list()
            select_client_loaders = list()
            record = -1
            for idx in selected_clients:
                select_client_loaders.append(client_all_loaders[idx])
                record += 1
                if idx in self.args.forget_client_idx:
                    self.select_forget_idx.append(record)
            std_time = time.time()
            client_models = self.global_train_once(epoch, fused_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            fused_model.load_state_dict(client_models.state_dict())  # client_models is now the avg model
            consume_time += end_time-std_time

            avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, fused_model, self.args, test_loaders)
            result_list.extend(test_result_ls)
            print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero, avg_test_acc))

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc'])
        df['Comsume_time'] = consume_time
        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm, self.args.forget_paradigm, self.args.data_name, self.args.alpha, len(self.args.forget_class_idx)))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                        self.args.forget_paradigm,
                                                                                        self.args.data_name,
                                                                                        self.args.alpha,
                                                                                        len(self.args.forget_class_idx), self.args.cut_sample))

        return fused_model

    def relearn_unlearning_knowledge(self, unlearning_model, client_all_loaders, test_loaders):
        checkpoints_ls = []
        all_global_models = list()
        all_client_models = list()
        global_model = unlearning_model
        result_list = []
        #pd.set_option('display.max_rows', None)

        all_global_models.append(global_model)
        std_time = time.time()
        for epoch in range(self.args.global_epoch):
            if self.args.forget_paradigm == 'client':
                select_client_loaders = list()
                
                for idx in self.args.forget_client_idx:
                    select_client_loaders.append(client_all_loaders[idx])
                '''
                select_client_loaders=client_all_loaders
                '''
                
            elif self.args.forget_paradigm == 'class':
                select_client_loaders = list()
                client_loaders = select_forget_class(self.args, copy.deepcopy(client_all_loaders))
                for v in client_loaders:
                    if v is not None:
                        select_client_loaders.append(v)
            elif self.args.forget_paradigm == 'sample':
                select_client_loaders = list()
                client_loaders = select_forget_sample(self.args, copy.deepcopy(client_all_loaders))
                for v in client_loaders:
                    if v is not None:
                        select_client_loaders.append(v)
            avg_model = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args,
                                                   checkpoints_ls)

            global_model.load_state_dict(avg_model.state_dict())
            del avg_model
            all_global_models.append(copy.deepcopy(global_model).to('cpu'))
            end_time = time.time()

            consume_time = end_time - std_time
            avg_f_acc, avg_r_acc = 0, 0

            if self.args.forget_paradigm == 'client':
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args,
                                                                          test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss',
                                           'Comsume_time'])
            elif self.args.forget_paradigm == 'class':
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args,
                                                                         test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss', 'Comsume_time'])
            elif self.args.forget_paradigm == 'sample':
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model, self.args, test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc', 'Comsume_time'])

            global_model.to('cpu')

            
            print("Relearn Round = {} Avg Forget Acc = {}, Avg Remember Acc = {}".format(epoch,avg_f_acc, avg_r_acc))
            #print(df)
        
        if self.args.cut_sample == 1.0:
            df.to_csv('./results/{}/relearn_data_{}_distri_{}_fnum_{}_algo_{}.csv'.format(self.args.forget_paradigm,
                                                                                          self.args.data_name,
                                                                                      self.args.alpha,
                                                                                      len(self.args.forget_class_idx),
                                                                                      self.args.paradigm), index=False)
        elif self.args.cut_sample < 1.0:
            df.to_csv('./results/{}/relearn_data_{}_distri_{}_fnum_{}_algo_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                          self.args.data_name,
                                                                                      self.args.alpha,
                                                                                      len(self.args.forget_class_idx),
                                                                                      self.args.paradigm, self.args.cut_sample), index=False)
        return
    

    def verify_restored_model(self, global_model, client_all_loaders, test_loaders):
        print("\n" + "="*50)
        print("VERIFYING RESTORED PHASE 1 MODEL")
        print("="*50)
        
        # 1. Load the exact frozen weights we extracted and saved via .unload()
        global_model.load_state_dict(torch.load('save_model/restored_phase1_model_{}.pth'.format(self.args.data_name)))
        global_model.eval()
        
        # 2. Immediate Test: This should EXACTLY match the final epoch of Phase 1
        if self.args.forget_paradigm == 'client':
            avg_f_acc, avg_r_acc, _ = test_client_forget(self, 0, global_model, self.args, test_loaders)
            print("INITIAL RESTORED STATE -> Avg Forget Acc: {:.4f}, Avg Remember Acc: {:.4f}".format(avg_f_acc, avg_r_acc))
            
        elif self.args.forget_paradigm == 'sample':
            avg_jingdu, avg_acc_zero, avg_test_acc, _ = test_backdoor_forget(self, 0, global_model, self.args, test_loaders)
            print("INITIAL RESTORED STATE -> Avg Forget Acc: {:.4f}, Avg Remember Acc: {:.4f}".format(avg_acc_zero, avg_test_acc))
            
        elif self.args.forget_paradigm == 'class':
            avg_f_acc, avg_r_acc, _ = test_class_forget(self, 0, global_model, self.args, test_loaders)
            print("INITIAL RESTORED STATE -> Avg Forget Acc: {:.4f}, Avg Remember Acc: {:.4f}".format(avg_f_acc, avg_r_acc))

        print("\nStarting Continual FL Training on Restored Model...")
        checkpoints_ls = []
        result_list = []
        std_time = time.time()
        
        # 3. The 50-Epoch Global Loop using ALL clients (Poisoned + Clean)
        for epoch in range(self.args.global_epoch):
            global_model.train()
            
            # Select ALL clients for standard federated training
            select_client_loaders = client_all_loaders 
            
            # global_train_once returns averaged model directly
            avg_model = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)
            global_model.load_state_dict(avg_model.state_dict())
            del avg_model
            
            global_model.eval()
            
            # Evaluate and log the trajectory
            if self.args.forget_paradigm == 'client':
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch+1, global_model, self.args, test_loaders)
                result_list.extend(test_result_ls)
                print('Restored Model - Epoch {}, Avg_r_acc: {:.4f}, Avg_f_acc: {:.4f}'.format(epoch+1, avg_r_acc, avg_f_acc))
                
            elif self.args.forget_paradigm == 'sample':
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch+1, global_model, self.args, test_loaders)
                result_list.extend(test_result_ls)
                print('Restored Model - Epoch {}, Avg_r_acc: {:.4f}, Avg_f_acc: {:.4f}'.format(epoch+1, avg_test_acc, avg_acc_zero))
                
            elif self.args.forget_paradigm == 'class':
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch+1, global_model, self.args, test_loaders)
                result_list.extend(test_result_ls)
                print('Restored Model - Epoch {}, Avg_r_acc: {:.4f}, Avg_f_acc: {:.4f}'.format(epoch+1, avg_r_acc, avg_f_acc))

        # 4. Save the trajectory to a CSV for graphing
        if self.args.forget_paradigm in ['client', 'class']:
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss'] if self.args.forget_paradigm == 'client' else ['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss'])
        else:
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc'])
            
        df['Consume_time'] = time.time() - std_time
        df.to_csv('./results/{}/restored_model_verification_{}_distri_{}.csv'.format(self.args.forget_paradigm, self.args.data_name, self.args.alpha), index=False)
        
        return global_model