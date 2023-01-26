from env import  Env
from recommendation_system import RecommendationSystem

if __name__ == '__main__':
    env = Env()
    rs = RecommendationSystem(env, n_neighbors=10)
    rs.run()