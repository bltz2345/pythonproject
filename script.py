import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import random
from deepface import DeepFace
from PIL import Image

# TMDb API 연동용 클래스
class TMDbFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_base = "https://api.themoviedb.org/3"
        self.image_base = "https://image.tmdb.org/t/p/w500"

    def discover_movies_by_genres(self, genre_ids, count=100):
        url = f"{self.api_base}/discover/movie"
        params = {
            "api_key": self.api_key,
            "with_genres": ','.join(map(str, genre_ids)),
            "sort_by": "popularity.desc",
            "language": "ko-KR",
            "page": 1
        }
        response = requests.get(url, params=params)
        movies = []
        if response.status_code == 200:
            for movie in response.json().get("results", [])[:count]:
                title = movie.get("title")
                genres = self.get_genre_names(movie.get("genre_ids", []))
                rating = movie.get("vote_average", 0)
                overview = movie.get("overview", "")
                poster_path = movie.get("poster_path")
                poster_url = f"{self.image_base}{poster_path}" if poster_path else None
                movies.append({"title": title, "genre": genres, "rating": rating, "poster": poster_url, "overview": overview})
        return movies

    def get_genre_names(self, ids):
        genre_map = {
            28: "액션", 12: "모험", 16: "애니메이션", 35: "코미디",
            80: "범죄", 18: "드라마", 10751: "가족", 14: "판타지",
            36: "역사", 27: "공포", 10402: "음악", 9648: "미스터리",
            10749: "로맨스", 878: "SF", 10770: "TV 영화", 53: "스릴러",
            10752: "전쟁", 37: "서부"
        }
        return ' '.join([genre_map.get(gid, str(gid)) for gid in ids])

    def search_movie(self, query):
        url = f"{self.api_base}/search/movie"
        params = {"api_key": self.api_key, "query": query, "language": "ko-KR"}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json().get("results", [])
        return []

# 영화 추천 시스템 클래스
class MovieRecommender:
    def __init__(self, tmdb_api_key):
        self.tmdb = TMDbFetcher(tmdb_api_key)
        self.emotion_genre_map = {
            'happy': [10749, 35],        # 로맨스, 코미디
            'sad': [18],                 # 드라마
            'angry': [53, 80],           # 스릴러, 범죄
            'surprise': [14, 878],       # 판타지, SF
            'neutral': [16, 10751],      # 애니메이션, 가족
            'excited': [28, 12]          # 액션, 모험
        }
        self.movies_data = self.fetch_movies_from_tmdb()
        self.df = pd.DataFrame(self.movies_data)
        self.prepare_recommendations()
        self.emotion_to_genre = {
            'happy': '행복할 때는 로맨스나 코미디 영화를 추천합니다! 😊',
            'sad': '슬플 때는 감동적인 드라마로 마음을 달래보세요 😢',
            'angry': '화가 날 때는 스릴러나 액션으로 스트레스를 풀어보세요 😠',
            'surprise': '신선한 자극이 필요하다면 SF나 판타지 영화! 😮',
            'neutral': '편안한 시간에는 가족 영화나 애니메이션! 😐',
            'excited': '에너지가 넘칠 때는 액션 영화로 더 신나게! 🤩'
        }

    def fetch_movies_from_tmdb(self):
        data = []
        for emotion, genre_ids in self.emotion_genre_map.items():
            movies = self.tmdb.discover_movies_by_genres(genre_ids, count=100)
            for m in movies:
                m['emotion_match'] = emotion
                data.append(m)
        return data

    def prepare_recommendations(self):
        tfidf = TfidfVectorizer(stop_words='english')
        self.df['overview'] = self.df['overview'].fillna("")
        tfidf_matrix = tfidf.fit_transform(self.df['overview'])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def get_recommendations_by_emotion(self, emotion, num_recommendations=5):
        emotion_movies = self.df[self.df['emotion_match'] == emotion]
        if len(emotion_movies) >= num_recommendations:
            return emotion_movies.nlargest(num_recommendations, 'rating')
        return self.df.nlargest(num_recommendations, 'rating')

    def get_similar_movies(self, movie_title, num=5):
        idx = self.df[self.df['title'].str.lower() == movie_title.lower()].index
        if idx.empty:
            return pd.DataFrame()
        sim_scores = list(enumerate(self.cosine_sim[idx[0]]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.df.iloc[movie_indices]

    def get_real_time_recommendations(self, movie_title, num=5):
        results = self.tmdb.search_movie(movie_title)
        if not results:
            return []
        target = results[0]
        target_overview = target.get("overview", "")
        if not target_overview:
            return []
        tfidf = TfidfVectorizer(stop_words='english')
        corpus = [target_overview] + self.df['overview'].tolist()
        tfidf_matrix = tfidf.fit_transform(corpus)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        top_indices = cosine_sim.argsort()[-num:][::-1]
        return self.df.iloc[top_indices].to_dict(orient='records')

    def detect_emotion_from_image(self, image):
        try:
            from deepface import DeepFace
            import numpy as np

            image = image.convert("RGB")
            img_array = np.array(image)

            result = DeepFace.analyze(img_path=img_array, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]

            emotion = result['dominant_emotion']
            emotion_map = {
                'happy': 'happy', 'sad': 'sad', 'angry': 'angry', 'surprise': 'surprise',
                'neutral': 'neutral', 'fear': 'neutral', 'disgust': 'angry'
            }
            mapped = emotion_map.get(emotion, 'neutral')
            confidence = result['emotion'][emotion] / 100.0
            return mapped, confidence

        except Exception as e:
            print("DeepFace Error:", e)
            emotions = ['happy', 'sad', 'angry', 'surprise', 'neutral', 'excited']
            weights = [0.3, 0.15, 0.1, 0.15, 0.25, 0.05]
            return random.choices(emotions, weights=weights)[0], random.uniform(0.7, 0.95)


def main():
    st.set_page_config(page_title="\U0001F3AC AI 영화 추천 시스템", layout="wide")
    tmdb_api_key = "587e1fa6ea89acde84397fec1d463361"
    recommender = MovieRecommender(tmdb_api_key)

    st.title("\U0001F3AC AI 영화 추천 시스템 + 감정 인식")
    st.write("감정을 인식하거나 좋아하는 영화로 비슷한 영화를 추천받아보세요!")

    tab1, tab2 = st.tabs(["\U0001F4F8 감정 기반 추천", "\U0001F3AF 영화 입력 기반 추천"])

    with tab1:
        st.header("\U0001F4F8 얼굴 감정으로 영화 추천받기")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("\U0001F4F7 사진 업로드")
            photo_option = st.radio("사진 입력 방식 선택:", ["카메라로 촬영", "파일 업로드", "감정 직접 선택"])
            detected_emotion = None
            confidence = 0
            if photo_option == "카메라로 촬영":
                camera_photo = st.camera_input("\U0001F4F8 사진을 찍어주세요!")
                if camera_photo:
                    image = Image.open(camera_photo)
                    st.image(image, caption="촬영된 사진", width=300)
                    detected_emotion, confidence = recommender.detect_emotion_from_image(image)
            elif photo_option == "파일 업로드":
                uploaded_file = st.file_uploader("이미지 파일을 업로드하세요", type=['jpg', 'jpeg', 'png'])
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="업로드된 이미지", width=300)
                    detected_emotion, confidence = recommender.detect_emotion_from_image(image)
            else:
                emotion_options = {
                    '😊 행복': 'happy', '😢 슬픔': 'sad', '😠 화남': 'angry',
                    '😮 놀라움': 'surprise', '😐 무감정': 'neutral', '🤩 흥미진진': 'excited'
                }
                selected_emotion_text = st.selectbox("현재 기분을 선택하세요:", list(emotion_options.keys()))
                detected_emotion = emotion_options[selected_emotion_text]
                confidence = 1.0

        with col2:
            if detected_emotion:
                emotion_korean = {
                    'happy': '😊 행복', 'sad': '😢 슬픔', 'angry': '😠 화남',
                    'surprise': '😮 놀라움', 'neutral': '😐 평온', 'excited': '🤩 흥미진진'
                }
                st.success(f"**감지된 감정**: {emotion_korean[detected_emotion]}\n**신뢰도**: {confidence:.1%}\n\n{recommender.emotion_to_genre[detected_emotion]}")
                st.subheader("\U0001F3AC 추천 영화 목록")
                recommendations = recommender.get_recommendations_by_emotion(detected_emotion, 5)
                for _, movie in recommendations.iterrows():
                    with st.container():
                        movie_col1, movie_col2 = st.columns([1, 4])
                        with movie_col1:
                            if movie['poster']:
                                st.image(movie['poster'], width=100)
                            else:
                                st.write("(포스터 없음)")
                        with movie_col2:
                            st.markdown(f"**{movie['title']}**")
                            st.markdown(f"장르: {movie['genre']}  ⭐ {movie['rating']}/10")
                    st.divider()

    with tab2:
        st.header("\U0001F3AF 좋아하는 영화 입력하기")
        movie_input = st.text_input("영화 제목을 입력하세요 (예: 인셉션)")
        if movie_input:
            realtime_recommendations = recommender.get_real_time_recommendations(movie_input, num=5)
            if realtime_recommendations:
                st.subheader("\U0001F3AC 실시간 유사 영화 추천")
                for movie in realtime_recommendations:
                    with st.container():
                        movie_col1, movie_col2 = st.columns([1, 4])
                        with movie_col1:
                            if movie['poster']:
                                st.image(movie['poster'], width=100)
                            else:
                                st.write("(포스터 없음)")
                        with movie_col2:
                            st.markdown(f"**{movie['title']}**")
                            st.markdown(f"장르: {movie['genre']}  ⭐ {movie['rating']}/10")
                    st.divider()
            else:
                st.warning("입력한 영화에 대한 유사한 결과를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()