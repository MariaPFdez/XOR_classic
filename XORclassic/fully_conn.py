import torch
import os
from XORclassic.utils import load_variables
import math

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# some variables initialization
input_dim = 28*28
hidden_dim = 1024

# learning rate parameters
eta_dec, eta_enc = 0.05, 0.02
eta_b_dec, eta_b_enc = 0.02, 0.01

# To classify (when training the fully connected are not used)
classes = 10
C = torch.zeros(hidden_dim, classes, device = device)
b_C = torch.zeros(classes, device = device)
eta_cls, eta_b_cls = 0.05, 0.02

# training settings
epochs = 5
max_corrections = 3
p_ltd_required_consecutive = 3



def forward(inp, W, b_W, V, b_V, C, b_C):

    inp = inp.to(W.device)
    
    # feedforward of the fully connected network
    hid = ((inp @ W + b_W) >= 0.0).float()
    pred_ae = ((hid @ V + b_V) >= 0.0).float()
    
    # to classify
    pred_cl = hid @ C.to(hid.device) + b_C.to(hid.device)
    
    return hid, pred_ae, pred_cl



def training_fully(train_dataloader, test_dataloader):

    # Weights initialization
    W = torch.randn(input_dim, hidden_dim, device = device) * 0.010
    b_W = torch.zeros(hidden_dim, device = device)
    V = torch.randn(hidden_dim, input_dim, device = device) * 0.010
    b_V = torch.zeros(input_dim, device = device)
    
    # random feedback for the encoder update
    B = torch.randn(input_dim, hidden_dim, device = device) * (1.0 / math.sqrt(input_dim)) 
    
    for epoch in range(epochs):

        num, den, num_test, den_test = 0, 0, 0, 0
        
        for inp, _ in train_dataloader:
        
            inp = inp.view(inp.size(0), -1).to(device)
            hid,  pred_ae, _ = forward(inp, W, b_W, V, b_V, C, b_C)       
            count = 0
            
            while (inp != pred_ae).sum().item() > 0 and count < max_corrections:
            
                # # calculate xor error with sign
                # xor_sign = (inp - pred_ae)

                # Logical error + direction
                e = (inp != pred_ae)  
                m_up = e & inp.bool() # push-up mask
                m_dn = e & (~inp.bool()) # push-down mask
                xor_sign = m_up.float() - m_dn.float() # logical equivalent of (x - y)
                
                # update decoder weights
                V += (eta_dec*(hid.transpose(0,1) @ xor_sign))
                b_V += eta_b_dec*xor_sign.mean(dim = 0)
                
                # update encoder weights with the fixed random feedback matrix
                enc_sign = torch.sign(xor_sign @ B)
                W += (eta_enc*(inp.transpose(0,1) @ enc_sign))
                b_W += eta_b_enc*enc_sign.mean(dim = 0)
                
                # get the prediction again
                hid,  pred_ae, _ = forward(inp, W, b_W, V, b_V, C, b_C)
                count += 1
            num += (inp == pred_ae).sum().item()
            den += inp.numel()

        print('Epoch', epoch+1)
        print('Accuracy:', num/den*100, '%')
    
        # calculate accuracy of the test dataset
        for inp_test, _ in test_dataloader:
            inp_test = inp_test.view(inp_test.size(0), -1).to(device)
            _,  pred_ae_test, _ = forward(inp_test, W, b_W, V, b_V, C, b_C)
            num_test += (inp_test == pred_ae_test).sum().item()
            den_test += inp_test.numel()
        print('Accuracy of test dataset:', num_test/den_test*100, '% \n')

    variables = {'B': B, 'W': W, 'V': V,
                 'b_W': b_W, 'b_V': b_V}
                 

    return variables



def training_fully_classif(train_dataloader, test_dataloader, classif_data, train_data, network):
    
    # To classify (when training cnn are not used)
    if classif_data == 'MNIST':
        classes = 10
    elif classif_data == 'EMNIST':
        classes = 47
    else:
        raise TypeError("You must choose an option between MNIST and EMNIST")

    # variables loaded
    B, W, V, b_W, b_V = load_variables(train_data, network, classif_data = None)

    # classifier variables initialization
    C = torch.zeros(hidden_dim, classes, device = device)
    b_C = torch.zeros(classes, device = device)


    for epoch in range(epochs):
        
        correct, total, correct_test, total_test = 0, 0, 0, 0
       
        for inp, targ in train_dataloader:
    
            # feedforward step
            inp = inp.view(inp.size(0), -1).to(device)
            targ = targ.to(device)
            hid, _, log = forward(inp, W, b_W, V, b_V, C, b_C)
            pred_cl = torch.argmax(log, dim = 1)
            targ_oh = torch.nn.functional.one_hot(targ, num_classes = classes)
            pred_cl_oh = torch.nn.functional.one_hot(pred_cl, num_classes = classes)
    
            # classifier weights update
            err_sign = (targ_oh - pred_cl_oh).to(torch.float)
            C += eta_cls*(hid.transpose(0,1) @ err_sign)
            b_C += eta_b_cls*err_sign.mean(dim = 0)
    
            # calculate accuracy
            _, _, log = forward(inp, W, b_W, V, b_V, C, b_C)
            pred_cl = torch.argmax(log, dim = 1)
            correct += (pred_cl == targ).sum().item()
            total += targ.size(0)
        print('Epoch:' , epoch+1)
        print('Accuracy:', correct/total)
    
        # calculate test accuracy
        for inp_test, targ_test in test_dataloader:
            
            inp_test = inp_test.view(inp_test.size(0), -1).to(device)
            targ_test = targ_test.to(device)
            _, _, log_test = forward(inp_test, W, b_W, V, b_V, C, b_C)
            pred_cl_test = torch.argmax(log_test, dim = 1)
            correct_test += (targ_test == pred_cl_test).sum().item()
            total_test += targ_test.size(0)
        print('Accuracy of test dataset:', correct_test/total_test*100, '% \n')

    variables = {'C': C, 'b_C': b_C}               

    return variables