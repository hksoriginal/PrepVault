from newsapi import NewsApiClient


class NewsAPI:

    def __init__(self, api_key):
        self.newsapi = NewsApiClient(api_key='663d6475ca1148699cfd8bc1bf9f2cdc')
        self.base_url = "https://newsapi.org/v2/top-headlines"
