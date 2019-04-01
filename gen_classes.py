import os
import json


def gen_classes():
    label_dirs = {'train': 'bdd/labels/train/', 'val': 'bdd/labels/val/'}
    classes = set()

    for mode in ['train', 'val']:
        print('Scanning "{0}" files'.format(mode))
        label_names = os.listdir(label_dirs[mode])

        for i, label_name in enumerate(label_names):
            if i > 0 and i % 5000 == 0:
                print('Scanned', i, 'files')
            with open(label_dirs[mode] + label_name) as input_file:
                for obj in json.load(input_file)['frames'][0]['objects']:
                    if 'box2d' not in obj or obj['category'] in classes:
                        continue

                    print('Found category:', obj['category'])
                    classes.add(obj['category'])
        print('Scanned', len(label_names), 'files')
    return classes


classes = list(gen_classes())
print('Found classes: ', classes)
