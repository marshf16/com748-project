from PIL import ImageFont
import visualkeras
import os
from tensorflow.keras.models import load_model


def visualize_cnn(model):
    model.add(visualkeras.SpacingDummyLayer(spacing=100))
    visualkeras.layered_view(model, to_file='src/self_driving_car/data/cnn_visualization.png', legend=True).show()  # font is optional!


def main():
    model = load_model(os.path.abspath('src/self_driving_car/data/sign_detection_model.h5'), compile=False)
    visualize_cnn(model)


if __name__ == '__main__':
	main()