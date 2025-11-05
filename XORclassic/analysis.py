import matplotlib.pyplot as plt
import XORclassic.cnn
import XORclassic.fully_conn
import torch
from XORclassic.utils import load_variables, get_datasets
from sklearn.manifold import TSNE
from collections import Counter
import os

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# DISCLAIMER: These variables have to be the same that those appearing in cnn.py and fully_conn.py
hidden_channels_cnn = 28
hidden_channels_fully = 1024

def model_accuracy(test_data, train_data, network, classif = False):

    # dataset loaded
    _, test_dataloader = get_datasets(train_data, test_data)
    
    # variables loaded
    if classif: 
        B, W, V, b_W, b_V, C, b_C = load_variables(train_data, network, classif_data = test_data)
    else:
        B, W, V, b_W, b_V = load_variables(train_data, network)
        # classifier variables initialization
        if test_data == 'Random' or test_data == 'MNIST':
            b_C = torch.zeros(10, device=device)
            if network == 'CNN':
                C = torch.zeros(hidden_channels_cnn*7*7, 10, device = device)
            else:
                C = torch.zeros(hidden_channels_fully, 10, device = device)
        else:
            b_C = torch.zeros(47, device=device)
            if network == 'CNN':
                C = torch.zeros(hidden_channels_cnn*7*7, 47, device = device)
            else:
                C = torch.zeros(hidden_channels_fully, 47, device = device)

    # variables initialization
    correct_cl_test, correct_rec_test, total_cl_test, total_rec_test = 0, 0, 0, 0
    
    # calculate test accuracy
    if network == 'CNN':
        for inp_test, targ_test in test_dataloader:
            inp_test = inp_test.view(-1, 1, 28, 28).to(device)
            targ_test = targ_test.to(device)
            _, _, pred_rec_test, log_test = XORclassic.cnn.forward(inp_test, W, b_W, V, b_V, C, b_C)  
            pred_cl_test = torch.argmax(log_test, dim = 1)
            correct_cl_test += (targ_test == pred_cl_test).sum().item()
            total_cl_test += targ_test.size(0)
            correct_rec_test += (inp_test == pred_rec_test).sum().item()
            total_rec_test += inp_test.numel()    
    
    elif network == 'Fully':
        for inp_test, targ_test in test_dataloader:
            inp_test = inp_test.view(inp_test.size(0), -1).to(device)
            targ_test = targ_test.to(device)
            _, pred_rec_test, log_test = XORclassic.fully_conn.forward(inp_test, W, b_W, V, b_V, C, b_C)
            pred_cl_test = torch.argmax(log_test, dim = 1)
            correct_cl_test += (targ_test == pred_cl_test).sum().item()
            total_cl_test += targ_test.size(0)
            correct_rec_test += (inp_test == pred_rec_test).sum().item()
            total_rec_test += inp_test.numel()    

    else:
        raise TypeError('Error with network: You must choose an option between CNN and Fully')

    if classif:
        print('Accuracy of test dataset when reconstructing:', correct_rec_test/total_rec_test*100, '% \n')
        print('Accuracy of test dataset when classifying:', correct_cl_test/total_cl_test*100, '% \n')
    else:
        print('Accuracy of test dataset when reconstructing:', correct_rec_test/total_rec_test*100, '% \n')



def plot_tsne(train_data, network, test_data = None, save_fig = True):

    if test_data == None:
        test_data = train_data

     # dataset loaded
    _, test_dataloader = get_datasets(train_data, test_data)
    
    # variables loaded   
    B, W, V, b_W, b_V = load_variables(train_data, network)
    
    # classifier variables initialization
    if test_data == 'Random' or test_data == 'MNIST':
        b_C = torch.zeros(10, device=device)
        if network == 'CNN':
            C = torch.zeros(hidden_channels_cnn*7*7, 10, device = device)
        else:
            C = torch.zeros(hidden_channels_fully, 10, device = device)
    else:
        b_C = torch.zeros(47, device=device)
        if network == 'CNN':
            C = torch.zeros(hidden_channels_cnn*7*7, 47, device = device)
        else:
            C = torch.zeros(hidden_channels_fully, 47, device = device)

    hid_list = []
    labels_list = []
    
    if network == 'Fully':
        for inp, targ in test_dataloader:
            inp = inp.view(inp.size(0), -1).to(device)
            hid, _, _ = XORclassic.fully_conn.forward(inp, W, b_W, V, b_V, C, b_C)
            hid_flat = hid.view(hid.shape[0], -1).cpu()  # pasar a CPU
            hid_list.append(hid_flat)
            labels_list.append(targ.cpu())
    elif network == 'CNN':
        for inp, targ in test_dataloader:
            inp = inp.view(-1, 1, 28, 28).to(device)
            _, hid_cl, _, _ = XORclassic.cnn.forward(inp, W, b_W, V, b_V, C, b_C)
            hid_flat = hid_cl.view(hid_cl.shape[0], -1).cpu()  # pasar a CPU
            hid_list.append(hid_flat)
            labels_list.append(targ.cpu())
    else:
        raise TypeError("Error with network: You must choose an option between CNN and Fully")
    
    hid_all = torch.cat(hid_list, dim=0).numpy()
    labels_all = torch.cat(labels_list, dim=0).numpy()
    
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    hid_2d = tsne.fit_transform(hid_all)
    
    plt.figure(figsize=(8,8))
    scatter = plt.scatter(hid_2d[:,0], hid_2d[:,1], c=labels_all, cmap='tab10', s=10)
    plt.legend(*scatter.legend_elements(), title="Digits")
    plt.title("Latent space t-SNE")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    if save_fig:
        save_path = 'analysis_results_XORclassic/tSNE'
        if os.path.isdir(save_path):
            pass
        else:
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{network}_trained_with_{train_data}_tested_with_{test_data}.png'))

    plt.show()


def plot_reconstructions(train_data, network, test_data = None, save_fig = True):
    
    if test_data == None:
        test_data = train_data

     # dataset loaded
    _, test_dataloader = get_datasets(train_data, test_data)
    num_examples = 10  # examples to show
    examples_shown = 0
    
    # variables loaded   
    B, W, V, b_W, b_V = load_variables(train_data, network)
    
    # classifier variables initialization
    if test_data == 'Random' or test_data == 'MNIST':
        b_C = torch.zeros(10, device=device)
        if network == 'CNN':
            C = torch.zeros(hidden_channels_cnn*7*7, 10, device = device)
        else:
            C = torch.zeros(hidden_channels_fully, 10, device = device)
    else:
        b_C = torch.zeros(47, device=device)
        if network == 'CNN':
            C = torch.zeros(hidden_channels_cnn*7*7, 47, device = device)
        else:
            C = torch.zeros(hidden_channels_fully, 47, device = device)

    # calculate predictions and plot figure
    plt.figure(figsize=(num_examples * 2, 4))
    
    for inp, _ in test_dataloader:
        # calculate predictions depending on the network
        if network == 'Fully':
            inp = inp.view(inp.size(0), -1).to(device)
            _, pred_rec_test, _ = XORclassic.fully_conn.forward(inp, W, b_W, V, b_V, C, b_C)
        elif network == 'CNN':
            _, _, pred_rec_test, _ = XORclassic.cnn.forward(inp, W, b_W, V, b_V, C, b_C) 
        else:
            raise TypeError("Error with network: You must choose an option between CNN and Fully")
            
        for i in range(inp.size(0)):
            if examples_shown >= num_examples:
                break
    
            orig = inp[i].view(28, 28).detach().cpu()
            recon = pred_rec_test[i].view(28, 28).detach().cpu()
    
            # show original
            plt.subplot(2, num_examples, examples_shown + 1)
            plt.imshow(orig, cmap="gray")
            plt.axis("off")
            if examples_shown == 0:
                plt.title("Original")
    
            # show reconstructed
            plt.subplot(2, num_examples, num_examples + examples_shown + 1)
            plt.imshow(recon, cmap="gray")
            plt.axis("off")
            if examples_shown == 0:
                plt.title("Reconstructed")
    
            examples_shown += 1
    
        if examples_shown >= num_examples:
            break

    plt.suptitle(f"Reconstruction of {test_data} by {network} trained with {train_data}", fontsize=12)

    if save_fig:
        save_path = 'analysis_results_XORclassic/Reconstructions'
        if os.path.isdir(save_path):
            pass
        else:
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{network}_trained_with_{train_data}_tested_with_{test_data}.png'))

    plt.show()



def plot_classifications(train_data, network, test_data = None, save_fig = True):
    
    if test_data == None:
        test_data = train_data

     # dataset loaded
    _, test_dataloader = get_datasets(train_data, test_data)
    test = test_dataloader.dataset
    
    # variables loaded   
    B, W, V, b_W, b_V, C, b_C = load_variables(train_data, network, classif_data = test_data)
    if test_data == 'MNIST':
        classes, rows, cols, figsize = 10, 2, 5, (20, 8)
    else:
        classes, rows, cols, figsize = 47, 10, 5, (20, 40)

    # calculate the test elements of each class
    class_masks = [[] for _ in range(classes)]
    for i, (_, targ) in enumerate(test):
        class_masks[int(targ)].append(i)

    # calculate the predictions for each class
    preds = [[] for _ in range(classes)]
    
    for i in range(classes):
    
        correct, total = 0, len(class_masks[i])
        
        if network == 'Fully':
            for elem_class in class_masks[i]:
            
                inp, targ = test[elem_class]
                inp = inp.view(inp.size(0), -1).to(device)
            
                # feedforward step
                _, _, log = XORclassic.fully_conn.forward(inp, W, b_W, V, b_V, C, b_C) 
                pred_cl = torch.argmax(log, dim = 1)
                preds[i].append(int(pred_cl.cpu()))
                if int(pred_cl) == targ:
                    correct += 1
            
            print('Class {} accuracy: {} %'.format(i, correct/total*100))
            print('Number of elements of class {}: {}'.format(i, total), '\n')
            
        elif network == 'CNN':
            for elem_class in class_masks[i]:
            
                inp, targ = test[elem_class]
                inp = inp.view(-1, 1, 28, 28).to(device)
            
                # feedforward step
                _, _, _, log = XORclassic.cnn.forward(inp, W, b_W, V, b_V, C, b_C) 
                pred_cl = torch.argmax(log, dim = 1)
                preds[i].append(int(pred_cl.cpu()))
                if int(pred_cl) == targ:
                    correct += 1
            
            print('Class {} accuracy: {} %'.format(i, correct/total*100))
            print('Number of elements of class {}: {}'.format(i, total), '\n')
        else: 
            raise TypeError('Error with network: You must choose an option between CNN and Fully')

    # plot the results
    fig, axs = plt.subplots(rows, cols, figsize = figsize)  
    axs = axs.flatten() 
    
    for i in range(classes):
        conteo = Counter(preds[i])
        axs[i].bar(list(conteo.keys()), list(conteo.values()))
        axs[i].set_title(f'Clase {i}')
        axs[i].set_xlabel('Predicción')
        axs[i].set_ylabel('Frecuencia')
    
    plt.tight_layout()

    if save_fig:
        save_path = 'analysis_results_XORclassic/Classifications'
        if os.path.isdir(save_path):
            pass
        else:
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{network}_trained_with_{train_data}_tested_with_{test_data}.png'))

    plt.show()