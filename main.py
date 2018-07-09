from app.result_predictor import generate_predicted_json


if __name__ == '__main__':
    generate_predicted_json('/Users/victor/Desktop/test_images', '/Users/victor/Desktop/model/data/models/adaptive-20.pkl',
                            '/Users/victor/Desktop/model/data/annotations/vocab.pkl', '/Users/victor/Desktop/result.json', 224,
                            256, False)