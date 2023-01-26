import json
from flask import Flask, request
from recommendation_system import RecommendationSystem
from env import Env

app = Flask(__name__)


@app.route('/train_model', methods=['POST'])
def train_model():
    inputs = request.get_json()
    env = Env()
    rs = RecommendationSystem(env=env, **inputs)
    rs.run()


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
