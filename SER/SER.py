from SER_Process_Data import read_dataset
from SER_Module import learning, valid


if __name__ == '__main__':
    # Define paths for source & target domain
    source_path = './Musical_Instruments.json'
    target_path = './Patio_Lawn_and_Garden.json'

    iteration = 300

    path = source_path[2:-5] + '_plus_' + target_path[2:-5]
    print('Source & Target domain: ', path)

    save = './' + path + '.pth'
    write_file = './Performance_' + path + '.txt'

    s_data, s_dict, t_train, t_valid, t_test, t_dict, w_embed = read_dataset(source_path, target_path)

    for i in range(iteration):
        # After 1 epoch of training -> load trained parameter
        if i > 0:
            learning(s_data, s_dict, t_train, t_dict, w_embed, save, 1)
        # First training
        else:
            learning(s_data, s_dict, t_train, t_dict, w_embed, save, 0)

        # Validation and Test
        valid(t_valid, t_test, t_dict, w_embed, save, write_file)
