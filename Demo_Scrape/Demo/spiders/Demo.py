

import scrapy
import json
from nltk import tokenize 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from Demo.items import DemoItem
import re 
# Command: 
# scrapy crawl demo -o ../FinalProject/Test/demo.csv
class DemoSpider(scrapy.Spider):
    name = 'demo'
    allowed_domains = ['yelp.com']
    start_urls = ['https://www.yelp.com/biz/buttermilk-kitchen-atlanta']

    def parse(self, response):
        out = DemoItem()
        cleanr = re.compile('<.*?>')

        for i in response.xpath('//p[@lang = "en"]').extract():
            i = re.sub(cleanr, ' ', i)
            i = i.replace(u'\xa0',u'')
            out['review'] = i.encode('utf-8').strip()
            yield out
