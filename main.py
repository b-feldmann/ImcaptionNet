from app.result_predictor import generate_predicted_json


if __name__ == '__main__':
    generate_predicted_json('/test/data', '/model/imcaption_net.pkl',
                            '/vocab/vocab.pkl', '/result/result.json', 224,
                            256, False)
