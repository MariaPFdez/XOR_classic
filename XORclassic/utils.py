from torchvision import datasets, transforms
import torch
import os

batch_size = 256

# to binarize datasets
tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (x > 0.5).float())])


def get_datasets(train_data, test_data):

    # to select the train dataset
    if train_data == 'MNIST':      
        train = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        
    elif train_data == 'EMNIST':
        train = datasets.EMNIST(root="data", split = 'balanced', train=True, download=True, transform=tfm)
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        
    elif train_data == 'Random':       
        train_dataloader = create_random_dataset(random_train = True, random_test = False)

    else:
        raise TypeError("Error with train_data: You must choose an option between MNIST, EMNIST and Random") 


    # to select the test dataset
    if test_data == 'MNIST':
        test = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    elif test_data == 'EMNIST':
        test = datasets.EMNIST(root="data", split = 'balanced', train=False, download=True, transform=tfm)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    elif test_data == 'Random':        
        test_dataloader = create_random_dataset(random_train = False, random_test = True)

    else: 
        raise TypeError("Error with test_data: You must choose an option between MNIST, EMNIST and Random") 


    print('Train and test datasets loaded \n')
    
    return train_dataloader, test_dataloader





def create_random_dataset(random_train, random_test):

    # Load MNIST to get the b&w proportion
    train = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
    
    # Join MNIST datasets and normalize (train + test)
    all_data = torch.cat([train.data, test.data], dim=0)
    all_data = all_data.float() / 255.0  # normalizar a [0,1]

    # Get the b&w proportion
    threshold = 0.5
    num_pixels = all_data.numel()
    num_white = (all_data > threshold).sum().item()
    num_black = num_pixels - num_white
    p_white = num_white / num_pixels
    p_black = 1 - p_white 
    # print(f"Proporción de píxeles blancos: {p_white:.4f}")
    # print(f"Proporción de píxeles negros: {p_black:.4f}")

    train_size = len(train)
    test_size = len(test)
    img_shape = train.data.shape[1:]  # (28, 28)

    if random_train:
        
        # Random images (1=white, 0=black) using p_white
        train_images = (torch.rand((train_size, *img_shape)) < p_white).float()

        # Add channel (N, 1, 28, 28)
        train_images = train_images.unsqueeze(1)

        # Dummy labels to use them while training
        train_labels = torch.zeros(train_size, dtype=torch.long)

        # Create datasets and dataloaders
        train_random = torch.utils.data.TensorDataset(train_images, train_labels)
        dataloader = torch.utils.data.DataLoader(train_random, batch_size=batch_size, shuffle=True)


    if random_test:

        # Random images (1=white, 0=black) using p_white
        test_images = (torch.rand((test_size, *img_shape)) < p_white).float()

        # Add channel (N, 1, 28, 28)
        test_images = test_images.unsqueeze(1)

        # Dummy labels to use them while training
        test_labels = torch.zeros(test_size, dtype=torch.long)

        # Create datasets and dataloaders
        test_random = torch.utils.data.TensorDataset(test_images, test_labels)
        dataloader = torch.utils.data.DataLoader(test_random, batch_size=batch_size, shuffle=False)

    return dataloader

# # Histograma de intensidades reales de MNIST
# plt.hist(all_data.flatten().numpy(), bins=50)
# plt.title("Histograma de intensidad de píxeles (MNIST)")
# plt.xlabel("Valor del píxel (0=negro, 1=blanco)")
# plt.ylabel("Frecuencia")
# plt.show()

# # Ejemplo de imagen MNIST real
# plt.imshow(train.data[0], cmap="gray")
# plt.title("Ejemplo MNIST real")
# plt.show()

# # Ejemplo de imagen aleatoria con misma distribución
# plt.imshow(train_images[0, 0], cmap="gray")
# plt.title("Ejemplo de imagen aleatoria con distribución MNIST")
# plt.show()



def save_variables(network, train_data, variables, classif_data = None):

    if classif_data:
        save_path = os.path.join('saved_variables_XORclassic', network, train_data, 'Classifier', classif_data)
    else:
        save_path = os.path.join('saved_variables_XORclassic', network, train_data)
                             
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
        
    for key, value in variables.items():
        torch.save(value, os.path.join(save_path, f'{key}.pt'))

    print('Variables saved in:', save_path)



def load_variables(train_data, network, classif_data = None):

    
    load_path = os.path.join('saved_variables_XORclassic', network, train_data)
    if os.path.isdir(load_path) == False:
        raise TypeError("The model you want to select was not saved")
    
    if classif_data:
        load_path_class = os.path.join('saved_variables_XORclassic', network, train_data, 'Classifier', classif_data)
        if os.path.isdir(load_path_class) == False:
            raise TypeError("The classifier model you want to select was not saved")
        
    variables = ['B', 'W', 'V', 'b_W', 'b_V']
    loaded_vars = {}
    
    for name in variables:
        tensor = torch.load(os.path.join(load_path, f'{name}.pt'))
        loaded_vars[name] = tensor

    if classif_data: 
        for name in ['C', 'b_C']:
            tensor = torch.load(os.path.join(load_path_class, f'{name}.pt'))
            loaded_vars[name] = tensor
        return loaded_vars['B'], loaded_vars['W'], loaded_vars['V'], loaded_vars['b_W'], loaded_vars['b_V'], loaded_vars['C'], loaded_vars['b_C']
    else:
        return loaded_vars['B'], loaded_vars['W'], loaded_vars['V'], loaded_vars['b_W'], loaded_vars['b_V']


