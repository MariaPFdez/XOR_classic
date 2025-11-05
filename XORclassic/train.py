from XORclassic.utils import get_datasets, save_variables
from XORclassic.cnn import training_cnn, training_cnn_classif
from XORclassic.fully_conn import training_fully, training_fully_classif


def train_net(train_data, network, test_data = None, save_vars = True):

    if test_data == None:
        test_data = train_data
    
    train_dataloader, test_dataloader = get_datasets(train_data, test_data)

    if network == 'CNN':
        variables = training_cnn(train_dataloader = train_dataloader, test_dataloader = test_dataloader)

    elif network == 'Fully':
        variables = training_fully(train_dataloader = train_dataloader, test_dataloader = test_dataloader)

    else:
        raise TypeError("Error with network: You must choose an option between CNN and Fully") 

    if save_vars:
        save_variables(network, train_data, variables, classif_data = None)



def train_class(classif_data, train_data, network, save_vars = True):
    
    train_dataloader, test_dataloader = get_datasets(classif_data, classif_data)

    if network == 'CNN':
        variables = training_cnn_classif(train_dataloader = train_dataloader, test_dataloader = test_dataloader, classif_data = classif_data, train_data = train_data, network = network)

    elif network == 'Fully':
        variables = training_fully_classif(train_dataloader = train_dataloader, test_dataloader = test_dataloader, classif_data = classif_data, train_data = train_data, network = network)

    else:
        raise TypeError("Error with network: You must choose an option between CNN and Fully") 

    if save_vars:
        save_variables(network, train_data, variables, classif_data = classif_data)

