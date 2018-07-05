import os
from result_predictor import generate_predicted_json


if __name__ == '__main__':
    generate_predicted_json('/app/test/data/test_images', '/Users/victor/Desktop/model/data/models/adaptive-17.pkl',
                            '/Users/victor/Desktop/model/data/annotations/vocab.pkl', '/app/test/result.json', 224,
                            256, False)