import joblib
from sklearn.feature_extraction.text import CountVectorizer  # 피체 벡터화
from sklearn.metrics.pairwise import cosine_similarity  # 코사인 유사도
import numpy as np
import pandas as pd
import os

class cos_similarity():
    def __init__(self,datas,idx):
        # 사용 데이터 추출
        self.store = datas[idx].groupby('가게명',as_index=False).max().loc[:,['가게명','업종']] # ,'메뉴'

        # 훈련 데이터 컬럼 생성
        self.store['가게명+업종'] = self.store['가게명']+' '+self.store['업종']

        # 유사도 측정 객체 생성
        count_vect_category = CountVectorizer(min_df=0, ngram_range=(1,2))

        # cos유사도 훈련 및 변환
        place_category = count_vect_category.fit_transform(self.store['가게명+업종'].dropna(axis=0, inplace=False)) 

        # 단어별 문자 유사도 테이블 생성
        place_simi_cate = cosine_similarity(place_category, place_category) 

        # 유사도 순 인덱스 정렬
        self.place_simi_cate_sorted_ind = place_simi_cate.argsort()[:, ::-1]


    # 유사업체 출력 함수
    # 검색가게명, 출력항목 수
    def __call__(self, title_name, top_n=10):
        
        # 검색 가게명 저장
        place_title = self.store[self.store['가게명'] == title_name]
        # print(place_title)

        # 검색 가게명 인덱스 저장
        place_index = place_title.index.values
        # print(place_index)

        # 검색 가게명과 유사한 순서로 인덱스 정렬 후
        similar_indexes = self.place_simi_cate_sorted_ind[place_index, :(top_n)].reshape(-1)
        # print(similar_indexes)

        # 데이터에서 인덱스를 정렬된 순서로 리턴
        return self.store.iloc[similar_indexes][['가게명','업종']].iloc[1:,:] # ,'메뉴'
    

