from tqdm import trange, tqdm
import torch
from .traineval_utils import save_model


@torch.no_grad()
def validate(val_dataloader, model, params, epoch, n_best,
             max_answer_length, lowest_val_loss=None):
    
    epoch_loss = 0
    num_samples = 0
    
    model.eval()
    
    for batch in tqdm(val_dataloader, ncols = 100):
        batch =  tuple(t.to(params['device']) for t in batch)
        b_input_ids, b_attn_mask, b_start_indices, b_end_indices = batch
        output = model(input_ids=b_input_ids, 
                       attention_mask=b_attn_mask,
                       start_positions=b_start_indices,
                       end_positions=b_end_indices)
        loss = output.loss

        epoch_loss += loss.item()
        num_samples += b_input_ids.shape[0]
    
    epoch_val_loss = epoch_loss/num_samples
   
    if lowest_val_loss == None:
        lowest_val_loss = epoch_val_loss
        save_model(epoch, model, lowest_val_loss, params['model_save_path'] + '/lowest_loss_model.pt')
    else:
        if epoch_val_loss < lowest_val_loss:
            lowest_val_loss = epoch_val_loss
            save_model(epoch, model, lowest_val_loss, params['model_save_path'] + '/lowest_loss_model.pt')
    
    return lowest_val_loss



def train(model, train_dataloader, val_dataloader, optimizer, scheduler, params):
        print(params)
        
        start_epoch = 1
        train_loss_set = []
        print_every = 100
        print_loss = 0
        print_steps = 0
        
        n_best = params['n_best']
        
        lowest_eval_loss = None
        
        max_answer_length = params['max_answer_length']
        
        for i in trange(0, params['num_epochs'], desc="Epoch", ncols = 100):
            
            actual_epoch = start_epoch + i
            
            model.train()
            
            tr_loss = 0
            num_train_samples = 0

            optimizer.zero_grad()
            
            # TRAINING STEP
            for step, batch in enumerate(tqdm(train_dataloader, ncols = 100)):
                #pdb.set_trace()
                batch = tuple(t.to(params['device']) for t in batch)
                b_input_ids, b_attn_mask, b_start_ids, b_end_ids = batch
                
                
                output = model(input_ids=b_input_ids,
                               attention_mask=b_attn_mask,
                               start_positions=b_start_ids,
                               end_positions=b_end_ids)
                loss = output.loss
                loss = loss/params['num_grad_acc_step']
        
                tr_loss += loss.item()
                print_loss += loss.item()
                
                num_train_samples += b_input_ids.size(0)
                print_steps += b_input_ids.size(0)
                
                loss.backward()
                
                # if step % print_every == 0:
                #     print('Loss: ', print_loss / print_steps)
                #     print_loss = 0
                #     print_steps = 0
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if (step + 1) % params['num_grad_acc_step'] == 0 or (step + 1) == len(train_dataloader):
                    optimizer.step()
                    scheduler.step()

                    optimizer.zero_grad()
            
            epoch_train_loss = tr_loss / num_train_samples
            train_loss_set.append(epoch_train_loss)
        
        
            # VALIDATION STEP
            model.eval()
            with torch.no_grad():
                lowest_eval_loss = validate(val_dataloader, model, params, actual_epoch, n_best,
                                            max_answer_length, lowest_eval_loss)
            print("\nLoss after Epoch %d : %0.6f, %0.6f"%(actual_epoch, epoch_train_loss, lowest_eval_loss))
        print('\n')

