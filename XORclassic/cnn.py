import torch
from XORclassic.utils import load_variables
import math

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# To make it work: kernel = 2*padding + 1
kernel = 3 #5
padding = 1
hidden_channels = 28 #64

# learning rates
eta_dec, eta_enc = 0.05, 0.02
eta_b_dec, eta_b_enc = 0.02, 0.01
eta_cls, eta_b_cls = 0.05, 0.02

# Training settings
epochs = 5
max_corrections = 3
p_ltd_required_consecutive = 3


def forward(inp, W, b_W, V, b_V, C, b_C):

    inp = inp.to(W.device)

    # CNN feedforward
    hid = ((torch.nn.functional.conv2d(inp, W, bias = None, padding = padding) + b_W) >= 0.0).float()
    mp = torch.nn.MaxPool2d(4, return_indices=True)
    hid_cl, idx = mp(hid)
    up = torch.nn.MaxUnpool2d(4)
    hid = up(hid_cl, idx)
    pred_ae = ((torch.nn.functional.conv2d(hid, V, bias = None, padding = padding) + b_V) >= 0.0).float()
    
    # to classify
    fl_hid = torch.flatten(hid_cl, start_dim = 1)
    pred_cl = fl_hid @ C.to(hid.device) + b_C.to(hid.device)
    
    return hid, hid_cl, pred_ae, pred_cl



def decoder_update(hid, xor_sign, kernel, padding, hidden_channels):

    # split input into sliding windows, transformed into columns
    hid_unf = torch.nn.functional.unfold(hid, kernel_size = kernel, padding = padding)

    # calculate updates and convert them into the original shape
    xor_sign_unf = xor_sign.view(hid.shape[0], 1, hid.shape[2]*hid.shape[3])
    dV_flat = torch.zeros(1, hidden_channels*kernel**2, device = hid.device)
    for b in range(hid.shape[0]):
        dV_flat += xor_sign_unf[b] @ hid_unf[b].transpose(0,1)
    dV = dV_flat.view(1, hidden_channels, kernel, kernel)
    
    return dV



def encoder_update(inp, enc_sign, kernel, padding, hidden_channels):

    # split input into sliding windows, transformed into columns
    inp_unf = torch.nn.functional.unfold(inp, kernel_size = kernel, padding = padding)
    enc_sign_unf = torch.nn.functional.unfold(enc_sign, kernel_size = 1, padding = 0)

    # calculate updates and convert them into the original shape
    dW = torch.zeros(hidden_channels, 1, kernel, kernel, device = inp.device)
    for b in range(inp.shape[0]):
        dW += (enc_sign_unf[b] @ inp_unf[b].transpose(0,1)).view(hidden_channels, 1, kernel, kernel)
    
    return dW


def training_cnn(train_dataloader, test_dataloader):
    
    # Weights initialization
    W = torch.randn(hidden_channels, 1, kernel, kernel, device=device) * 0.010
    b_W = torch.zeros(1, hidden_channels, 1, 1, device=device)
    V = torch.randn(1, hidden_channels, kernel, kernel, device=device) * 0.010
    b_V = torch.zeros(1, 1, 1, 1, device=device)
    
    # random feedback for the encoder update
    B = torch.randn(hidden_channels, 1, kernel, kernel, device=device) * (1.0 / math.sqrt(kernel*kernel)) 

    # To classify (when training cnn are not used)
    classes = 10
    C = torch.zeros(hidden_channels*7*7, classes, device=device)
    b_C = torch.zeros(classes, device=device)

    for epoch in range(epochs):
    
        num, den, num_test, den_test = 0, 0, 0, 0
        
        for inp, _ in train_dataloader:
    
            inp = inp.view(-1, 1, 28, 28).to(device)
            hid, _, pred_ae, _ = forward(inp, W, b_W, V, b_V, C, b_C)  
            count = 0
            
            while (inp != pred_ae).sum().item() > 0 and count < max_corrections:
            
                # calculate xor error with sign
                xor_sign = (inp - pred_ae)
                
                # update decoder weights
                V += (eta_dec*decoder_update(hid, xor_sign, kernel, padding, hidden_channels))
                b_V += eta_b_dec*xor_sign.mean(dim = (0,2,3), keepdim = True)
                
                # update encoder weights with the fixed random feedback matrix
                enc_sign = torch.sign(torch.nn.functional.conv2d(xor_sign, B, padding = padding))
                W += (eta_enc*encoder_update(inp, enc_sign, kernel, padding, hidden_channels))
                b_W += eta_b_enc*enc_sign.mean(dim = (0,2,3), keepdim = True)
                
                # get the prediction again
                hid, _, pred_ae, _ = forward(inp, W, b_W, V, b_V, C, b_C)
                count += 1
            num += (inp == pred_ae).sum().item()
            den += inp.numel()
        
        print('Epoch', epoch+1)
        print('Accuracy:', num/den*100, '%')
    
        # calculate accuracy of the test dataset
        for inp_test, _ in test_dataloader:
            inp_test = inp_test.view(-1, 1, 28, 28).to(device)
            _, _, pred_rec_test, _ = forward(inp_test, W, b_W, V, b_V, C, b_C)
            num_test += (inp_test == pred_rec_test).sum().item()
            den_test += inp_test.numel()
        print('Accuracy of test dataset:', num_test/den_test*100, '% \n')

    variables = {'B': B, 'W': W, 'V': V,
                 'b_W': b_W, 'b_V': b_V}
                 

    return variables



def training_cnn_classif(train_dataloader, test_dataloader, classif_data, train_data, network):
    
    # To classify (when training cnn are not used)
    if classif_data == 'MNIST':
        classes = 10
    elif classif_data == 'EMNIST':
        classes = 47
    else:
        raise TypeError("Error with classif_data: You must choose an option between MNIST and EMNIST")

    # variables loaded
    B, W, V, b_W, b_V = load_variables(train_data, network, classif_data = None)

    # classifier variables initialization
    C = torch.zeros(hidden_channels*7*7, classes, device=device)
    b_C = torch.zeros(classes, device=device)

    for epoch in range(epochs):
        
        correct, total, correct_test, total_test = 0, 0, 0, 0
       
        for inp, targ in train_dataloader:
    
            inp = inp.view(-1, 1, 28, 28).to(device)
            targ = targ.to(device)
            # feedforward
            _, hid_cl, _, log = forward(inp, W, b_W, V, b_V, C, b_C)  
            pred_cl = torch.argmax(log, dim = 1)
            targ_oh = torch.nn.functional.one_hot(targ, num_classes = classes)
            pred_cl_oh = torch.nn.functional.one_hot(pred_cl, num_classes = classes)
    
            # classifier weights update
            err_sign = (targ_oh - pred_cl_oh).to(torch.float)
            fl_hid = torch.flatten(hid_cl, start_dim = 1)
            C += eta_cls*(fl_hid.transpose(0,1) @ err_sign)
            b_C += eta_b_cls*err_sign.mean(dim = 0)
    
            # calculate accuracy
            _, _, _, log = forward(inp, W, b_W, V, b_V, C, b_C)  
            pred_cl = torch.argmax(log, dim = 1)
            correct += (pred_cl == targ).sum().item()
            total += targ.size(0)
        print('Epoch:' , epoch+1)
        print('Accuracy:', correct/total*100, '%')
    
        # calculate test accuracy
        for inp_test, targ_test in test_dataloader:
    
            inp_test = inp_test.view(-1, 1, 28, 28).to(device)
            targ_test = targ_test.to(device)
            _, _, _, log_test = forward(inp_test, W, b_W, V, b_V, C, b_C)  
            pred_cl_test = torch.argmax(log_test, dim = 1)
            correct_test += (targ_test == pred_cl_test).sum().item()
            total_test += targ_test.size(0)
        print('Accuracy of test dataset:', correct_test/total_test*100, '% \n')

    variables = {'C': C, 'b_C': b_C}
                 

    return variables
