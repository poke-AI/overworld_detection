import numpy as np
import pandas as pd
import os
import sys
import glob
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for member in root.findall('object'):
            value = (
                os.path.splitext(os.path.abspath(xml_file))[0] + ".jpg",
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text),
                member[0].text,
            )
            xml_list.append(value)
            
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    xml_df = pd.DataFrame(xml_list, columns = column_name)
    return xml_df


if __name__ == "__main__":
    data_folder_path = os.path.abspath(sys.argv[1])
    train_path = data_folder_path + "/train/"
    validation_path = data_folder_path + "/validation/"

    train_df = xml_to_csv(train_path)
    train_df.to_csv(data_folder_path + "/train.csv", index = False, header=False)

    validation_df = xml_to_csv(validation_path)
    validation_df.to_csv(data_folder_path + "/validation.csv", index = False, header=False)