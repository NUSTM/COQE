from torch.utils.data import Dataset, DataLoader


class MyData(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list[0])

    def __getitem__(self, index):
        return_list = []
        for i in range(len(self.data_list)):
            return_list.append(list(self.data_list[i][index]))

        return return_list


def collate_fn(data):
    """
    :param data: data denote [(), ()], length is the number of parameters
    :return: a list of list
    """
    data_tuple_list, res_list = list(zip(*data)), []
    for i in range(len(data_tuple_list)):
        res_list.append(list(data_tuple_list[i]))
    return res_list


def get_loader(data_list, batch_size):
    data_set = MyData(data_list)
    data_loader = DataLoader(data_set, batch_size, shuffle=False, collate_fn=collate_fn)
    return data_loader


def create_first_data_loader(data_dict, batch_size, mask_index=None):
    feature_data = [data_dict['input_ids'], data_dict['attn_mask'], data_dict['start_index'], data_dict['end_index'],
                    data_dict['comparative_label'], data_dict['multi_label'], data_dict['result_label']]

    print("feature size: ", len(feature_data))

    if mask_index is not None:
        for index in range(len(feature_data)):
            feature_data[index] = feature_data[index][mask_index]
    print("data loader size: ", len(feature_data[0]))
    return get_loader(feature_data, batch_size)
