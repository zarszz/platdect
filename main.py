import os.path
import argparse

from detector import detect


def load_model():
    dirname = os.path.dirname(__file__)
    cfg = os.path.join(dirname, 'input/plate.cfg')
    weights = os.path.join(dirname, 'input/plate.weights')
    label = ['Plate']
    return cfg, weights, label


def main():
    cfg, weights, classes = load_model()
    parser = argparse.ArgumentParser(description="Deteksi plat nomor pada kendaraan")
    sub_parser = parser.add_subparsers(title="daftar perintah", dest="command")

    parser_detect = sub_parser.add_parser('detect', help='Deteksi plat nomor')
    parser_detect.add_argument('input', type=str)

    args = parser.parse_args()
    image = args.input

    detect(image, cfg, weights, classes)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
