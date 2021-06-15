import os.path

from detector import detect, load_model
from flask import Flask, render_template, request

app = Flask(__name__)
cfg, weights, classes = load_model()


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    file = request.files['photo']

    # save image file to directory
    file_path = os.path.join(os.getcwd(), 'images', file.filename)
    file.save(file_path)

    # perform object detection
    confidences, no_of_detected, class_ids = detect(file_path, cfg, weights, classes, save_img=True)
    if no_of_detected > 0:
        accuracy = float("{:.2f}".format(confidences[0])) * 100
        return render_template('upload.html', filename=file.filename, accuracy=accuracy, success_detected=True)
    else:
        return render_template('upload.html', filename=file.filename, success_detected=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run("0.0.0.0", 5000, debug=True)
