def file_reader(file_path):
    try:
        if not file_path.endswith('.txt'):
            raise ValueError("The provided path '%s' does not correspond to a text file" % file_path)
        with open(file_path, 'r') as f:
            contents = f.readlines()
        contents = [i.split()[0] for i in contents]
        return contents
    except FileNotFoundError as e:
        print("The path '%s' does not exist : " % file_path, e)


def check_path(path):
    if not(os.path.exists(path)):
        os.mkdir(path)


def gray2rgb(img):
    img_rgb = np.zeros((*img.shape, 3))
    img_rgb[:, :, 0] = img
    img_rgb[:, :, 1] = img
    img_rgb[:, :, 2] = img
    return img_rgb


def file_writer_normal_list(file_path, list_values):
    try:
        if not file_path.endswith('.txt'):
            raise ValueError("The provided path '%s' does not correspond to a text file" % file_path)
        if not isinstance(list_values, list):
            raise TypeError("The values to be written into the file do not correspond to a list : ", type(list_values))
        with open(file_path, 'w+') as f:
            for item in list_values:
                f.write(str(item) + '\n')
    except Exception as e:
        print("Error in 'file_writer_normal_list' :", e)